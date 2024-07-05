import yaml
import os
import argparse
import pickle
import glob
import numpy as np
import json
from aria.data.midi import MidiDict
from aria.tokenizer import AbsTokenizer
from tqdm import tqdm
import random
from copy import deepcopy
import sys
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.utils import flatten

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=os.path.normpath("configs/configs_os.yaml"),
                    help="Path to the config file")
args = parser.parse_args()

# Load config file
with open(args.config, 'r') as f:
    configs = yaml.safe_load(f)

artifact_folder = configs["raw_data"]["artifact_folder"]
mono_folder = configs["raw_data"]["mono_folder"]
json_folder = configs["raw_data"]["json_folder"]
raw_data_folders = configs["raw_data"]["raw_data_folders"]
classical_data_path = raw_data_folders['classical']['folder_path']
# pop_data_path = raw_data_folders['pop']['folder_path']
jazz_data_path = raw_data_folders['jazz']['folder_path']

# Build the vocabulary
vocab = {}
# MIDI velocity range from 0 to 127
velocity = [0, 15, 30, 45, 60, 75, 90, 105, 120, 127]
# MIDI pitch range from 0 to 127
midi_pitch = list(range(0, 128))
# Onsets are quantized in 10 milliseconds up to 5 seconds
onset = list(range(0, 5001, 10))
duration = list(range(0, 5001, 10))

# Add the tokens to the vocabulary
for v in velocity:
    for p in midi_pitch:
        vocab[("piano", p, v)] = len(vocab) + 1
for o in onset:
    vocab[("onset", o)] = len(vocab) + 1
for d in duration:
    vocab[("dur", d)] = len(vocab) + 1

# Chord frequency ratio tokens
# 0.0 to 1.0 in 0.05 increments
for i in np.arange(0.0, 1.05, 0.05):
    vocab[("cfr", round(i, 2))] = len(vocab) + 1

# Chord density tokens
# 0.0 to 8.0 in 0.25 increments
for i in np.arange(0.0, 8.25, 0.25):
    vocab[("cd", round(i, 2))] = len(vocab) + 1

# Genre tokens
vocab["classical"] = len(vocab) + 1
# vocab["pop"] = len(vocab) + 1
vocab["jazz"] = len(vocab) + 1

# Accompaniment type tokens
# vocab["no_bridge"] = len(vocab) + 1

# Special tokens
vocab[('prefix', 'instrument', 'piano')] = len(vocab) + 1
vocab["<T>"] = len(vocab) + 1
vocab["<D>"] = len(vocab) + 1
vocab["<U>"] = len(vocab) + 1
vocab["<S>"] = len(vocab) + 1
vocab["<E>"] = len(vocab) + 1
vocab["SEP"] = len(vocab) + 1

# Print the vocabulary length
print(f"Vocabulary length: {len(vocab)}")

# Save the vocabulary
# Directory path
fusion_folder = os.path.join(artifact_folder, "fusion")
# Make directory if it doesn't exist
os.makedirs(fusion_folder, exist_ok=True)
vocab_path = os.path.join(fusion_folder, "vocab.json")
with open(os.path.join(fusion_folder, "vocab.pkl"), 'wb') as f:
    pickle.dump(vocab, f)


# Get all the midi file names in the data path recursively
classical_file_list = glob.glob(classical_data_path + '/**/*.midi', recursive=True)
# pop_file_list = glob.glob(pop_data_path + '/**/*.mid', recursive=True)
# pop_file_list = [file for file in pop_file_list if "versions" not in file]
jazz_file_list = glob.glob(jazz_data_path + '/**/*.midi', recursive=True)

file_dict = {}
file_dict.update({file: "classical" for file in classical_file_list})
# file_dict.update({file: "pop" for file in pop_file_list})
file_dict.update({file: "jazz" for file in jazz_file_list})

print(f"Number of Classical MIDI files: {sum(value == 'classical' for value in file_dict.values())}")
# print(f"Number of Pop MIDI files: {sum(value == 'pop' for value in file_dict.values())}")
print(f"Number of Jazz MIDI files: {sum(value == 'jazz' for value in file_dict.values())}")
print(f"Number of MIDI files: {len(file_dict)}")

# Shuffle the file_dict
file_list = list(file_dict.keys())
random.shuffle(file_list)

# Split the data into train and validation sets
train_data = {k: v for k, v in file_dict.items() if k in file_list[:int(0.9 * len(file_list))]}
val_data = {k: v for k, v in file_dict.items() if k in file_list[int(0.9 * len(file_list)):]}
print(f"Number of training files: {len(train_data)}")
print(f"Number of validation files: {len(val_data)}")

# Save the train and validation file lists as JSON
with open(os.path.join(fusion_folder, "train_file_list.json"), "w") as f:
    json.dump(train_data, f)

with open(os.path.join(fusion_folder, "valid_file_list.json"), "w") as f:
    json.dump(val_data, f)


def store_files_as_json(file_dict, pickle_folder, file_name="train"):
    aria_tokenizer = AbsTokenizer()
    list_of_tokens = []
    
    for idx, file_path in tqdm(enumerate(file_dict.keys()), total=len(file_dict), desc="Processing files"):
        genre = file_dict[file_path]
        # if genre == "pop":
        #     midi_dict = MidiDict.from_midi(file_path)
        #     final_seq = [midi_dict, genre]
        #     list_of_tokens.append(final_seq)
        # else:
        mid = MidiDict.from_midi(file_path)
        tokenized_sequence = aria_tokenizer.tokenize(mid)
        final_seq = [tokenized_sequence, genre]
        list_of_tokens.append(final_seq)
    
    # Save the list_of_tokens as a pickle file
    pickle_file_path = os.path.join(pickle_folder, f"{file_name}.pkl")
    with open(pickle_file_path, "wb") as f:
        pickle.dump(list_of_tokens, f)

# Store the training and validation files as json
store_files_as_json(train_data, fusion_folder, file_name="train")
store_files_as_json(val_data, fusion_folder, file_name="valid")