import yaml
import json
import pickle
import os
import glob
import random
import numpy as np
import sys
import argparse
from tqdm import tqdm
import time
import multiprocessing
from functools import partial
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
import torch
torch.set_warn_always(False)
from torch.nn import functional as F
from transformers import EncoderDecoderModel
from torch.cuda import is_available as cuda_available
from aria.data.midi import MidiDict
from aria.tokenizer import AbsTokenizer

from corruptions import DataCorruption
from generation import generate
from infill import infill
# from classify_genre import get_genre_probabilities

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.utils import flatten, unflatten_corrupted, parse_generation, unflatten_for_aria, Segment_Novelty, convert_midi_to_wav

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=os.path.normpath("configs/configs_style_transfer.yaml"),
                    help="Path to the config file")
parser.add_argument("--experiment_name", type=str, default="all",
                    help="Name of the experiment")
args = parser.parse_args()

# Load config file
with open(args.config, 'r') as f:
    configs = yaml.safe_load(f)
    
artifact_folder = configs["raw_data"]["artifact_folder"]

# Get tokenizer
tokenizer_filepath = os.path.join(artifact_folder, "style_transfer", "vocab_corrupted.pkl")
# Load the tokenizer dictionary
with open(tokenizer_filepath, "rb") as f:
    tokenizer = pickle.load(f)

# Reverse the tokenizer
decode_tokenizer = {v: k for k, v in tokenizer.items()}

# Load the fusion model
fusion_model = EncoderDecoderModel.from_pretrained(os.path.join(artifact_folder, "style_transfer", "fine_tuned_model"))
fusion_model.eval()
fusion_model.to("cuda" if cuda_available() else "cpu")
# print("Fusion model loaded")

# Open pkl file
with open(os.path.join(artifact_folder, "style_transfer", "fine_tuning_valid.pkl"), "rb") as f:
    valid_sequences = pickle.load(f)

# Create new directory if it does not exist
eval_folder = configs["raw_data"]["eval_folder"]
if not os.path.exists(eval_folder):
    os.makedirs(eval_folder)


def generate_batch_on_gpu(gpu_id, midi_file_paths, convert_to, context_before, context_after, 
                          t_segment_start, corruption_passes, corruption_name, corruption_rate, 
                          pass_number, fusion_model, configs, novel_peaks_pct, tokenizer, decode_tokenizer, experiment_name, 
                          counter, total_files, write_intermediate_passes):
    torch.cuda.set_device(gpu_id)
    for i, item in enumerate(midi_file_paths):
        midi_file_path = item
        audio_file_path = midi_file_path.replace(".mid", ".wav")
        output_folder = os.path.join(*midi_file_path.split("/")[:-1], experiment_name, "target_style_"+convert_to, "corruption_name_"+corruption_name, "corruption_rate_"+str(corruption_rate), "context_"+str(context_before)+"_"+str(context_after) , "pass_"+str(pass_number))
        print(output_folder)
        # time.sleep(30)
        try:
            generate(midi_file_path, audio_file_path, fusion_model, configs, novel_peaks_pct, 
                     t_segment_start, convert_to, context_before, context_after, corruption_passes,
                 tokenizer, decode_tokenizer, output_folder, save_original=False, quiet=True, write_intermediate_passes=write_intermediate_passes)
        except Exception as e:
            print(f"Error: {e}")
            print(f"Error in processing file: {midi_file_path}")
        
        # Safely increment the counter
        with counter.get_lock():
            counter.value += 1
        tqdm.write(f"Progress: {counter.value}/{total_files} files processed.")


def infill_batch_on_gpu(gpu_id, midi_file_paths, convert_to, context_before, context_after, context_infilling,
                          t_segment_start, corruption_passes, corruption_name, corruption_rate, 
                          pass_number, fusion_model, configs, novel_peaks_pct, tokenizer, decode_tokenizer, experiment_name, 
                          counter, total_files, write_intermediate_passes, temperature, save_infilling_only):
    torch.cuda.set_device(gpu_id)
    for i, item in enumerate(midi_file_paths):
        midi_file_path = item
        audio_file_path = midi_file_path.replace(".mid", ".wav")
        output_folder = os.path.join(*midi_file_path.split("/")[:-1], experiment_name, "target_style_"+convert_to, "pass_"+str(pass_number))
        print(output_folder)
        # time.sleep(30)
        try:
            infill(midi_file_path, audio_file_path, fusion_model, configs, novel_peaks_pct,
             t_segment_start, convert_to, context_before, context_after, context_infilling,
             corruption_passes, tokenizer, decode_tokenizer, output_folder, 
             save_original=True, quiet=False, write_intermediate_passes=write_intermediate_passes, 
             temperature=temperature, save_infilling_only=save_infilling_only)
        except Exception as e:
            print(f"Error: {e}")
            print(f"Error in processing file: {midi_file_path}")
        
        # Safely increment the counter
        with counter.get_lock():
            counter.value += 1
        tqdm.write(f"Progress: {counter.value}/{total_files} files processed.")


def run_parallel_generation(midi_file_paths, convert_to, context_before, context_after,
                            t_segment_start, corruption_passes, corruption_name, corruption_rate, 
                            pass_number, fusion_model, configs, novel_peaks_pct, tokenizer, decode_tokenizer, 
                            experiment_name, max_processes_per_gpu, write_intermediate_passes):
    
    available_gpus = list(range(torch.cuda.device_count()))
    midi_files_split = np.array_split(midi_file_paths, len(available_gpus) * max_processes_per_gpu)
    
    # Shared counter
    counter = multiprocessing.Value('i', 0)
    processes = []
    for gpu_id in available_gpus:
        fusion_model.to(f"cuda:{gpu_id}")
        # fusion_model.share_memory()
        for i in range(max_processes_per_gpu):
            p = multiprocessing.Process(target=generate_batch_on_gpu, 
                                         args=(gpu_id, midi_files_split[gpu_id * max_processes_per_gpu + i], 
                                               convert_to, context_before, context_after, 
                                               t_segment_start, corruption_passes, corruption_name, 
                                               corruption_rate, pass_number, fusion_model, 
                                               configs, novel_peaks_pct, tokenizer, decode_tokenizer, experiment_name, 
                                               counter, len(midi_file_paths), write_intermediate_passes))
            processes.append(p)
            p.start()

    for p in processes:
        p.join()

def run_parallel_infill(midi_file_paths, convert_to, context_before, context_after, context_infilling,
                            t_segment_start, corruption_passes, corruption_name, corruption_rate, 
                            pass_number, fusion_model, configs, novel_peaks_pct, tokenizer, decode_tokenizer, 
                            experiment_name, max_processes_per_gpu, write_intermediate_passes, temperature, save_infilling_only):
    
    available_gpus = list(range(torch.cuda.device_count()))
    midi_files_split = np.array_split(midi_file_paths, len(available_gpus) * max_processes_per_gpu)
    
    # Shared counter
    counter = multiprocessing.Value('i', 0)
    processes = []
    for gpu_id in available_gpus:
        fusion_model.to(f"cuda:{gpu_id}")
        # fusion_model.share_memory()
        for i in range(max_processes_per_gpu):
            p = multiprocessing.Process(target=infill_batch_on_gpu, 
                                         args=(gpu_id, midi_files_split[gpu_id * max_processes_per_gpu + i], 
                                               convert_to, context_before, context_after, context_infilling,
                                               t_segment_start, corruption_passes, corruption_name, 
                                               corruption_rate, pass_number, fusion_model, 
                                               configs, novel_peaks_pct, tokenizer, decode_tokenizer, experiment_name, 
                                               counter, len(midi_file_paths), write_intermediate_passes, temperature, save_infilling_only))
            processes.append(p)
            p.start()

    for p in processes:
        p.join()

def run_generation(midi_file_paths, convert_to, context, t_segment_start, 
                       corruptions, corruption_rates, n_passes, fusion_model, configs, novel_peaks_pct, 
                       tokenizer, decode_tokenizer, experiment_name, max_processes_per_gpu, write_intermediate_passes):
    print(f"Running experiment {experiment_name}")
    for corruption_name, corruption_function in corruptions.items():
        print(f"Corruption name: {corruption_name}")
        for corruption_rate in corruption_rates:
            print(f"Corruption rate: {corruption_rate}")
            corruption_passes = {'pass_1': {'corruption_rate': corruption_rate, 'corruption_type': corruption_name}}
            for context_length in context:
                print(f"Context: {context_length}")
                for pass_number in n_passes:
                    print(f"Number of passes: {pass_number}")
                    for n in range(2, pass_number + 1):
                        corruption_passes[f'pass_{n}'] = {'corruption_rate': corruption_rate, 'corruption_type': corruption_name}
                    run_parallel_generation(midi_file_paths, convert_to, context_length, context_length, 
                                            t_segment_start, corruption_passes, corruption_name, corruption_rate, 
                                            pass_number, fusion_model, configs, novel_peaks_pct, tokenizer, decode_tokenizer, experiment_name, 
                                            max_processes_per_gpu, write_intermediate_passes)
    print(f"Experiment {experiment_name} completed")


################ Write original midi files to evaluations folder ################
# Check if eval_folder has midi and wav files within them
# If it does, skip this step
# Otherwise, write the original midi files to the eval_folder
if len(glob.glob(os.path.join(eval_folder, "*", "*", "original_*.mid"))) > 0:
    # print("Original MIDI files already saved to evaluations folder")
    original_midi_file_paths = glob.glob(os.path.join(eval_folder, "*", "*", "original_*.mid"))
    original_wav_file_paths = glob.glob(os.path.join(eval_folder, "*", "*", "original_*.wav"))
else:
    aria_tokenizer = AbsTokenizer()
    for i, item in tqdm(enumerate(valid_sequences)):
        tokenized_sequence, genre = item
        # Save the original MIDI file
        mid_dict = aria_tokenizer.detokenize(tokenized_sequence)
        original_mid = mid_dict.to_midi()
        tmp_folder = os.path.join(eval_folder, genre, str(i))
        if not os.path.exists(tmp_folder):
            os.makedirs(tmp_folder)
        original_mid.save(os.path.join(tmp_folder, "original_" + str(i) + ".mid"))

    print("Original MIDI files saved to evaluations folder")

    original_midi_file_paths = glob.glob(os.path.join(eval_folder, "*", "*", "original_*.mid"))
    # Write wav files for original midi files
    original_wav_file_paths = convert_midi_to_wav(original_midi_file_paths, "/homes/kb658/fusion/artifacts/soundfont.sf", max_workers=6)
    print("Wav files saved to evaluations folder")


if __name__ == '__main__':

    # Set the start method
    multiprocessing.set_start_method('spawn')
    # fusion_model.share_memory()

    if args.experiment_name == "experiment_1" or args.experiment_name == "all":
        # ################ Experiment 1: Multiple Passes over different corruption rates with specific corruptions ################
        data_corruption_obj = DataCorruption()
        corruptions = data_corruption_obj.corruption_functions
        corruptions["random"] = None
        corruption_rates = [1.0, 0.75, 0.5, 0.25]
        n_passes = [10]
        context = [5]
        t_segment_start = 2
        novel_peaks_pct = 0.05
        convert_to = "jazz"
        experiment_name = "experiment_1"
        write_intermediate_passes = True
        max_processes_per_gpu = 4

        # Example usage:
        filtered_midi_file_paths = [file for file in original_midi_file_paths if convert_to not in file and "generated" not in file and "pop" not in file]
        print(f"Number of files to process: {len(filtered_midi_file_paths)}")
        run_generation(filtered_midi_file_paths, convert_to, context, t_segment_start, 
                    corruptions, corruption_rates, n_passes, fusion_model, configs, novel_peaks_pct, tokenizer, decode_tokenizer, 
                    experiment_name, max_processes_per_gpu, write_intermediate_passes=write_intermediate_passes)

        data_corruption_obj = DataCorruption()
        corruptions = data_corruption_obj.corruption_functions
        corruptions["random"] = None
        corruption_rates = [1.0, 0.75, 0.5, 0.25]
        n_passes = [10]
        context = [5]
        t_segment_start = 2
        novel_peaks_pct = 0.05
        convert_to = "classical"
        experiment_name = "experiment_1"
        write_intermediate_passes = True
        max_processes_per_gpu = 4

        # Example usage:
        filtered_midi_file_paths = [file for file in original_midi_file_paths if convert_to not in file and "generated" not in file and "pop" not in file]
        print(f"Number of files to process: {len(filtered_midi_file_paths)}")
        run_generation(filtered_midi_file_paths, convert_to, context, t_segment_start, 
                    corruptions, corruption_rates, n_passes, fusion_model, configs, novel_peaks_pct, tokenizer, decode_tokenizer, 
                    experiment_name, max_processes_per_gpu, write_intermediate_passes=write_intermediate_passes)
        

    if args.experiment_name == "experiment_2" or args.experiment_name == "all":
        # ################ Experiment 2: Multiple Passes over different contexts with random corruptions ################
        data_corruption_obj = DataCorruption()
        corruptions = dict()
        corruptions["random"] = None
        corruption_rates = [1.0]
        n_passes = [10]
        context = [5,4,3,2,1]
        t_segment_start = 2
        novel_peaks_pct = 0.05
        convert_to = "jazz"
        experiment_name = "experiment_2"
        write_intermediate_passes = True
        max_processes_per_gpu = 4

        # Example usage:
        filtered_midi_file_paths = [file for file in original_midi_file_paths if convert_to not in file and "generated" not in file and "pop" not in file]
        print(f"Number of files to process: {len(filtered_midi_file_paths)}")
        run_generation(filtered_midi_file_paths, convert_to, context, t_segment_start, 
                    corruptions, corruption_rates, n_passes, fusion_model, configs, novel_peaks_pct, tokenizer, decode_tokenizer, 
                    experiment_name, max_processes_per_gpu, write_intermediate_passes=write_intermediate_passes)

        data_corruption_obj = DataCorruption()
        corruptions = dict()
        corruptions["random"] = None
        corruption_rates = [1.0]
        n_passes = [10]
        context = [5,4,3,2,1]
        t_segment_start = 2
        novel_peaks_pct = 0.05
        convert_to = "classical"
        experiment_name = "experiment_2"
        write_intermediate_passes = True
        max_processes_per_gpu = 4

        # Example usage:
        filtered_midi_file_paths = [file for file in original_midi_file_paths if convert_to not in file and "generated" not in file and "pop" not in file]
        print(f"Number of files to process: {len(filtered_midi_file_paths)}")
        run_generation(filtered_midi_file_paths, convert_to, context, t_segment_start, 
                    corruptions, corruption_rates, n_passes, fusion_model, configs, novel_peaks_pct, tokenizer, decode_tokenizer, 
                    experiment_name, max_processes_per_gpu, write_intermediate_passes=write_intermediate_passes)
        
    if args.experiment_name == "experiment_3" or args.experiment_name == "all":
        # ################ Experiment 3: Variations. Multiple Passes over different corruption rates with specific corruptions ################
        data_corruption_obj = DataCorruption()
        corruptions = data_corruption_obj.corruption_functions
        corruptions["random"] = None
        corruption_rates = [1.0, 0.75, 0.5, 0.25]
        n_passes = [10]
        context = [5]
        t_segment_start = 2
        novel_peaks_pct = 0.05
        convert_to = "jazz"
        experiment_name = "experiment_3"
        write_intermediate_passes = True
        max_processes_per_gpu = 4

        # Example usage:
        filtered_midi_file_paths = [file for file in original_midi_file_paths if convert_to in file and "generated" not in file and "pop" not in file]
        print(f"Number of files to process: {len(filtered_midi_file_paths)}")
        run_generation(filtered_midi_file_paths, convert_to, context, t_segment_start, 
                    corruptions, corruption_rates, n_passes, fusion_model, configs, novel_peaks_pct, tokenizer, decode_tokenizer, 
                    experiment_name, max_processes_per_gpu, write_intermediate_passes=write_intermediate_passes)

        data_corruption_obj = DataCorruption()
        corruptions = data_corruption_obj.corruption_functions
        corruptions["random"] = None
        corruption_rates = [1.0, 0.75, 0.5, 0.25]
        n_passes = [10]
        context = [5]
        t_segment_start = 2
        novel_peaks_pct = 0.05
        convert_to = "classical"
        experiment_name = "experiment_3"
        write_intermediate_passes = True
        max_processes_per_gpu = 4

        # Example usage:
        filtered_midi_file_paths = [file for file in original_midi_file_paths if convert_to in file and "generated" not in file and "pop" not in file]
        print(f"Number of files to process: {len(filtered_midi_file_paths)}")
        run_generation(filtered_midi_file_paths, convert_to, context, t_segment_start, 
                    corruptions, corruption_rates, n_passes, fusion_model, configs, novel_peaks_pct, tokenizer, decode_tokenizer, 
                    experiment_name, max_processes_per_gpu, write_intermediate_passes=write_intermediate_passes)
    
    if args.experiment_name == "experiment_4" or args.experiment_name == "all":
        # ################ Experiment 4: Variations. Multiple Passes over different contexts with random corruptions ################
        data_corruption_obj = DataCorruption()
        corruptions = dict()
        corruptions["random"] = None
        corruption_rates = [1.0]
        n_passes = [10]
        context = [5,4,3,2,1]
        t_segment_start = 2
        novel_peaks_pct = 0.05
        convert_to = "jazz"
        experiment_name = "experiment_4"
        write_intermediate_passes = True
        max_processes_per_gpu = 4

        # Example usage:
        filtered_midi_file_paths = [file for file in original_midi_file_paths if convert_to in file and "generated" not in file and "pop" not in file]
        print(f"Number of files to process: {len(filtered_midi_file_paths)}")
        run_generation(filtered_midi_file_paths, convert_to, context, t_segment_start, 
                    corruptions, corruption_rates, n_passes, fusion_model, configs, novel_peaks_pct, tokenizer, decode_tokenizer, 
                    experiment_name, max_processes_per_gpu, write_intermediate_passes=write_intermediate_passes)

        data_corruption_obj = DataCorruption()
        corruptions = dict()
        corruptions["random"] = None
        corruption_rates = [1.0]
        n_passes = [10]
        context = [5,4,3,2,1]
        t_segment_start = 2
        novel_peaks_pct = 0.05
        convert_to = "classical"
        experiment_name = "experiment_4"
        write_intermediate_passes = True
        max_processes_per_gpu = 4

        # Example usage:
        filtered_midi_file_paths = [file for file in original_midi_file_paths if convert_to in file and "generated" not in file and "pop" not in file]
        print(f"Number of files to process: {len(filtered_midi_file_paths)}")
        run_generation(filtered_midi_file_paths, convert_to, context, t_segment_start, 
                    corruptions, corruption_rates, n_passes, fusion_model, configs, novel_peaks_pct, tokenizer, decode_tokenizer, 
                    experiment_name, max_processes_per_gpu, write_intermediate_passes=write_intermediate_passes)
        
    if args.experiment_name == "experiment_5" or args.experiment_name == "all":
        ################# Experiment 5: Infilling ################
        corruption_name = "whole_mask"
        corruption_rate = 1.0
        pass_number = 1
        corruption_passes = {'pass_1': {'corruption_rate': 1.0, 'corruption_type': "whole_mask"}}
        context_before = 4
        context_after = 4
        context_infilling = 2
        t_segment_start = 5
        novel_peaks_pct = 0
        convert_to = "jazz"
        experiment_name = "experiment_5"
        write_intermediate_passes = True
        max_processes_per_gpu = 1
        temperature = 0.97
        save_infilling_only = True

        # Example usage:
        filtered_midi_file_paths = [file for file in original_midi_file_paths if convert_to in file and "generated" not in file and "pop" not in file]
        print(f"Number of files to process: {len(filtered_midi_file_paths)}")
        run_parallel_infill(filtered_midi_file_paths, convert_to, context_before, context_after, context_infilling,
                            t_segment_start, corruption_passes, corruption_name, corruption_rate, 
                            pass_number, fusion_model, configs, novel_peaks_pct, tokenizer, decode_tokenizer, 
                            experiment_name, max_processes_per_gpu, write_intermediate_passes, temperature, save_infilling_only)
        
        corruption_name = "whole_mask"
        corruption_rate = 1.0
        pass_number = 1
        corruption_passes = {'pass_1': {'corruption_rate': 1.0, 'corruption_type': "whole_mask"}}
        context_before = 4
        context_after = 4
        context_infilling = 2
        t_segment_start = 5
        novel_peaks_pct = 0
        convert_to = "classical"
        experiment_name = "experiment_5"
        write_intermediate_passes = True
        max_processes_per_gpu = 1
        temperature = 0.97
        save_infilling_only = True

        # Example usage:
        filtered_midi_file_paths = [file for file in original_midi_file_paths if convert_to in file and "generated" not in file and "pop" not in file]
        print(f"Number of files to process: {len(filtered_midi_file_paths)}")
        run_parallel_infill(filtered_midi_file_paths, convert_to, context_before, context_after, context_infilling,
                            t_segment_start, corruption_passes, corruption_name, corruption_rate, 
                            pass_number, fusion_model, configs, novel_peaks_pct, tokenizer, decode_tokenizer, 
                            experiment_name, max_processes_per_gpu, write_intermediate_passes, temperature, save_infilling_only)
        
    if args.experiment_name == "experiment_6" or args.experiment_name == "all":
        ################# Experiment 5: Prompt Continuation ################
        corruption_name = "whole_mask"
        corruption_rate = 1.0
        pass_number = 1
        corruption_passes = {'pass_1': {'corruption_rate': 1.0, 'corruption_type': "whole_mask"}}
        context_before = 4
        context_after = 0
        context_infilling = 2
        t_segment_start = 5
        novel_peaks_pct = 0
        convert_to = "jazz"
        experiment_name = "experiment_6"
        write_intermediate_passes = True
        max_processes_per_gpu = 1
        temperature = 0.97
        save_infilling_only = True

        # Example usage:
        filtered_midi_file_paths = [file for file in original_midi_file_paths if convert_to in file and "generated" not in file and "pop" not in file]
        print(f"Number of files to process: {len(filtered_midi_file_paths)}")
        run_parallel_infill(filtered_midi_file_paths, convert_to, context_before, context_after, context_infilling,
                            t_segment_start, corruption_passes, corruption_name, corruption_rate, 
                            pass_number, fusion_model, configs, novel_peaks_pct, tokenizer, decode_tokenizer, 
                            experiment_name, max_processes_per_gpu, write_intermediate_passes, temperature, save_infilling_only)
        
        corruption_name = "whole_mask"
        corruption_rate = 1.0
        pass_number = 1
        corruption_passes = {'pass_1': {'corruption_rate': 1.0, 'corruption_type': "whole_mask"}}
        context_before = 4
        context_after = 0
        context_infilling = 2
        t_segment_start = 5
        novel_peaks_pct = 0
        convert_to = "classical"
        experiment_name = "experiment_6"
        write_intermediate_passes = True
        max_processes_per_gpu = 1
        temperature = 0.97
        save_infilling_only = True

        # Example usage:
        filtered_midi_file_paths = [file for file in original_midi_file_paths if convert_to in file and "generated" not in file and "pop" not in file]
        print(f"Number of files to process: {len(filtered_midi_file_paths)}")
        run_parallel_infill(filtered_midi_file_paths, convert_to, context_before, context_after, context_infilling,
                            t_segment_start, corruption_passes, corruption_name, corruption_rate, 
                            pass_number, fusion_model, configs, novel_peaks_pct, tokenizer, decode_tokenizer, 
                            experiment_name, max_processes_per_gpu, write_intermediate_passes, temperature, save_infilling_only)