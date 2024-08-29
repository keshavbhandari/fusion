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
import torch.multiprocessing as mp
from functools import partial
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
import torch
torch.set_warn_always(False)
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as dist
from transformers import EncoderDecoderModel
from torch.cuda import is_available as cuda_available
from aria.data.midi import MidiDict
from aria.tokenizer import AbsTokenizer

from corruptions import DataCorruption
from generate import generate
from classify_genre import get_genre_probabilities

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.utils import flatten, unflatten_corrupted, parse_generation, unflatten_for_aria, Segment_Novelty, convert_midi_to_wav

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=os.path.normpath("configs/configs_style_transfer.yaml"),
                    help="Path to the config file")
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
eval_folder = os.path.join("evaluations")
if not os.path.exists(eval_folder):
    os.makedirs(eval_folder)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def generate_batch_on_gpu(rank, world_size, midi_file_paths, convert_to, context_before, context_after, 
                          t_segment_start, corruption_passes, corruption_name, corruption_rate, 
                          pass_number, fusion_model, configs, tokenizer, decode_tokenizer, experiment_name, 
                          counter, total_files):
    # torch.cuda.set_device(gpu_id)
    setup(rank, world_size)
    model = DDP(fusion_model.to(rank), device_ids=[rank])
    for i, item in enumerate(midi_file_paths):
        midi_file_path = item
        audio_file_path = midi_file_path.replace(".mid", ".wav")
        output_folder = os.path.join(*midi_file_path.split("/")[:-1], experiment_name, "target_style_"+convert_to, "corruption_name_"+corruption_name, "corruption_rate_"+str(corruption_rate), "pass_"+str(pass_number))
        print(output_folder)
        # time.sleep(30)
        try:
            generate(midi_file_path, audio_file_path, model, configs, t_segment_start,
                 convert_to, context_before, context_after, corruption_passes,
                 tokenizer, decode_tokenizer, output_folder, save_original=False, quiet=True)
        except Exception as e:
            print(f"Error: {e}")
            print(f"Error in processing file: {midi_file_path}")
        
        # Safely increment the counter
        with counter.get_lock():
            counter.value += 1
        tqdm.write(f"Progress: {counter.value}/{total_files} files processed.")

    # Clean up
    cleanup()


# def run_parallel_generation(midi_file_paths, convert_to, context_before, context_after,
#                             t_segment_start, corruption_passes, corruption_name, corruption_rate, 
#                             pass_number, fusion_model, configs, tokenizer, decode_tokenizer, 
#                             experiment_name, max_processes_per_gpu=2):
    
#     available_gpus = list(range(torch.cuda.device_count()))
#     midi_files_split = np.array_split(midi_file_paths, len(available_gpus) * max_processes_per_gpu)
    
#     # Shared counter
#     counter = multiprocessing.Value('i', 0)
#     processes = []
#     for gpu_id in available_gpus:
#         fusion_model.to(f"cuda:{gpu_id}")
#         fusion_model.share_memory()
#         for i in range(max_processes_per_gpu):
#             p = multiprocessing.Process(target=generate_batch_on_gpu, 
#                                          args=(gpu_id, midi_files_split[gpu_id * max_processes_per_gpu + i], 
#                                                convert_to, context_before, context_after, 
#                                                t_segment_start, corruption_passes, corruption_name, 
#                                                corruption_rate, pass_number, fusion_model, 
#                                                configs, tokenizer, decode_tokenizer, experiment_name, counter, len(midi_file_paths)))
#             processes.append(p)
#             p.start()

#     for p in processes:
#         p.join()

def run_parallel_generation(midi_file_paths, convert_to, context_before, context_after,
                            t_segment_start, corruption_passes, corruption_name, corruption_rate, 
                            pass_number, fusion_model, configs, tokenizer, decode_tokenizer, 
                            experiment_name, max_processes_per_gpu=2):

    available_gpus = torch.cuda.device_count()
    world_size = available_gpus * max_processes_per_gpu
    midi_files_split = np.array_split(midi_file_paths, world_size)
    counter = mp.Value('i', 0)

    mp.spawn(generate_batch_on_gpu, 
             args=(world_size, midi_files_split, convert_to, context_before, context_after, 
                   t_segment_start, corruption_passes, corruption_name, corruption_rate, 
                   pass_number, fusion_model, configs, tokenizer, decode_tokenizer, 
                   experiment_name, counter, len(midi_file_paths)),
             nprocs=world_size)


def run_evaluation(midi_file_paths, convert_to, context_before, context_after, t_segment_start, 
                       corruptions, corruption_rates, n_passes, fusion_model, configs, tokenizer, decode_tokenizer, 
                       experiment_name, max_processes_per_gpu):
    print(f"Running experiment {experiment_name}")
    for corruption_name, corruption_function in corruptions.items():
        print(f"Corruption name: {corruption_name}")
        for corruption_rate in corruption_rates:
            print(f"Corruption rate: {corruption_rate}")
            corruption_passes = {'pass_1': {'corruption_rate': corruption_rate, 'corruption_type': corruption_name}}
            for pass_number in n_passes:
                print(f"Number of passes: {pass_number}")
                for n in range(2, pass_number + 1):
                    corruption_passes[f'pass_{n}'] = {'corruption_rate': corruption_rate, 'corruption_type': corruption_name}
                run_parallel_generation(midi_file_paths, convert_to, context_before, context_after, 
                                        t_segment_start, corruption_passes, corruption_name, corruption_rate, 
                                        pass_number, fusion_model, configs, tokenizer, decode_tokenizer, experiment_name, 
                                        max_processes_per_gpu=max_processes_per_gpu)
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
    mp.set_start_method('spawn')
    # fusion_model.share_memory()

    ################ Experiment 1: Single Pass with Specific Corruptions ################
    data_corruption_obj = DataCorruption()
    corruptions = data_corruption_obj.corruption_functions
    corruption_rates = [1.0]
    n_passes = [1,3,5,10]
    context_before = 5
    context_after = 5
    t_segment_start = 2
    convert_to = "jazz"
    experiment_name = "experiment_1"
    max_processes_per_gpu = 4  # Limit to n processes per GPU

    # Example usage:
    filtered_midi_file_paths = [file for file in original_midi_file_paths if convert_to not in file and "generated" not in file]
    print(f"Number of files to process: {len(filtered_midi_file_paths)}")
    run_evaluation(filtered_midi_file_paths, convert_to, context_before, context_after, t_segment_start, 
                   corruptions, corruption_rates, n_passes, fusion_model, configs, tokenizer, decode_tokenizer, 
                   experiment_name, max_processes_per_gpu)