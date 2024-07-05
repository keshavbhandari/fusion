import yaml
import jsonlines
import glob
import random
import os
import sys
import pickle
import json
import argparse
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset
import torch
from torch.nn import functional as F
from aria.data.midi import MidiDict
from aria.tokenizer import AbsTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.utils import flatten, unflatten, skyline, get_conditions, separate_list, interleave_conditions


class Fusion_Dataset(Dataset):
    def __init__(self, configs, data_list, mode="train", shuffle = False):
        self.mode = mode
        self.data_list = data_list
        if shuffle:
            random.shuffle(self.data_list)

        # Artifact folder
        self.artifact_folder = configs['raw_data']['artifact_folder']
        # Load encoder tokenizer json file dictionary
        tokenizer_filepath = os.path.join(self.artifact_folder, "fusion", "vocab.pkl")

        self.aria_tokenizer = AbsTokenizer()
        # Load the pickled tokenizer dictionary
        with open(tokenizer_filepath, 'rb') as f:
            self.tokenizer = pickle.load(f)

        # Get the maximum sequence length
        self.encoder_max_sequence_length = configs['model']['fusion_model']['encoder_max_sequence_length']
        self.decoder_max_sequence_length = configs['model']['fusion_model']['decoder_max_sequence_length']

        self.use_skyline_threshold = configs['training']['fusion']['use_skyline_threshold']
        self.use_conditions = configs['training']['fusion']['use_conditions']

        # Print length of dataset
        print("Length of dataset: ", len(self.data_list))

    def __len__(self):
        return len(self.data_list)
    
    def augmentation(self, sequence, change_pitch_by=5, static_velocity=True):
        # Apply pitch augmentation to the sequence
        sequence_copy = deepcopy(sequence)
        for n, note in enumerate(sequence_copy):
            if note[0] == "piano":
                new_pitch = note[1] + change_pitch_by
                if static_velocity:
                    velocity = 90
                else:
                    velocity = note[2]
                new_note = "<U>" if new_pitch < 0 or new_pitch > 127 else ("piano", new_pitch, velocity)
                sequence_copy[n] = new_note               

        return sequence_copy
    
    def remove_initial_tokens(self, data):
        result = []
        tokens_started = False
        
        for item in data:
            if isinstance(item, tuple) and item[0] == 'piano':
                result.append(item)
                tokens_started = True
            elif not tokens_started:
                if item != "<T>":
                    result.append(item)
            else:
                result.append(item)

        return result
    

    def get_melody_extracted_sequences(self, target_sequence):
        # Take the 3rd token as the start token until the 2nd last token
        target_sequence = target_sequence[2:-1]

        # Get random crop of the sequence of length max_sequence_length
        piano_token_indices = [i for i in range(len(target_sequence)) if target_sequence[i][0] == "piano"]
        # Exclude the last token of piano_token_indices to generate at least one token
        piano_token_indices = piano_token_indices[:-1]

        if len(piano_token_indices) > 0:
            # Choose the start index randomly
            start_idx = random.choice(piano_token_indices)
        else:
            print("No piano tokens found in the sequence for file")
            assert len(piano_token_indices) > 0

        # Crop the sequence
        end_idx = start_idx + self.decoder_max_sequence_length - 2
        target_sequence = target_sequence[start_idx:end_idx]

        # Call the flatten function
        flattened_sequence = flatten(target_sequence, add_special_tokens=True)
        
        if self.use_skyline_threshold:
            min_pitch_threshold = random.randint(55, 65)
        else: 
            min_pitch_threshold = 0
        # Call the skyline function
        melody, _ = skyline(flattened_sequence, diff_threshold=30, static_velocity=True, pitch_threshold=min_pitch_threshold)

        # Get melody to total ratio
        if len(melody) > 0:
            if self.use_conditions:
                separated_flattened_sequence = separate_list(flattened_sequence)
                cfr_list, cd_list = get_conditions(separated_flattened_sequence)
                conditioned_flattened_sequence = interleave_conditions(melody, cfr_list, cd_list)
                melody = unflatten(conditioned_flattened_sequence)
            else:
                melody = unflatten(melody)                

        target_sequence = flatten(target_sequence, add_special_tokens=True)
        target_sequence = unflatten(target_sequence)

        return target_sequence, melody
        

    def __getitem__(self, idx):
        sequence_info = self.data_list[idx]
        genre = sequence_info[-1]
        tokenized_sequence = sequence_info[0]

        meta_tokens = [genre]

        if genre in ["classical", "jazz"]:
            # Apply augmentations
            pitch_aug_function = self.aria_tokenizer.export_pitch_aug(12)
            tokenized_sequence = pitch_aug_function(tokenized_sequence)


            tokenized_sequence, melody = self.get_melody_extracted_sequences(tokenized_sequence)

        # Add the meta tokens to the input tokens
        input_tokens = meta_tokens + ["SEP"] + melody
        # input_tokens = melody

        # Trim the sequences if it is longer than expected
        if len(tokenized_sequence) >= self.decoder_max_sequence_length-2:
            tokenized_sequence = tokenized_sequence[-self.decoder_max_sequence_length-2:]
        # Add the start and end tokens
        tokenized_sequence = ["<S>"] + tokenized_sequence + ["<E>"]

        # Tokenize the melody and harmony sequences
        input_tokens = [self.tokenizer[tuple(token)] if isinstance(token, list) else self.tokenizer[token] for token in input_tokens]
        tokenized_sequence = [self.tokenizer[tuple(token)] if isinstance(token, list) else self.tokenizer[token] for token in tokenized_sequence]

        # Pad the sequences
        if len(tokenized_sequence) < self.decoder_max_sequence_length:
            tokenized_sequence = F.pad(torch.tensor(tokenized_sequence), (0, self.decoder_max_sequence_length - len(tokenized_sequence))).to(torch.int64)
        else:
            tokenized_sequence = torch.tensor(tokenized_sequence[-self.decoder_max_sequence_length:]).to(torch.int64)
        if len(input_tokens) < self.encoder_max_sequence_length:
            input_tokens = F.pad(torch.tensor(input_tokens), (0, self.encoder_max_sequence_length - len(input_tokens))).to(torch.int64)
        else:
            input_tokens = torch.tensor(input_tokens[-self.encoder_max_sequence_length:]).to(torch.int64)

        # Attention mask based on non-padded tokens of the phrase
        attention_mask = torch.where(input_tokens != 0, 1, 0).type(torch.bool)

        train_data = {"input_ids": input_tokens, "labels": tokenized_sequence, "attention_mask": attention_mask}

        return train_data





if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.normpath("configs/configs_os.yaml"),
                        help="Path to the config file")
    args = parser.parse_args()

    # Load config file
    with open(args.config, 'r') as f:
        configs = yaml.safe_load(f)
        
    artifact_folder = configs["raw_data"]["artifact_folder"]
    raw_data_folders = configs["raw_data"]["raw_data_folders"]

    # Get tokenizer
    tokenizer_filepath = os.path.join(artifact_folder, "fusion", "vocab.pkl")
    # Load the tokenizer dictionary
    with open(tokenizer_filepath, "rb") as f:
        tokenizer = pickle.load(f)
    # Reverse the tokenizer
    decode_tokenizer = {v: k for k, v in tokenizer.items()}


    # Open the train, validation, and test set files
    with open(os.path.join(artifact_folder, "fusion", "train.pkl"), "rb") as f:
        train_sequences = pickle.load(f)
    # with open(os.path.join(artifact_folder, "fusion", "valid.pkl"), "rb") as f:
    #     valid_sequences = pickle.load(f)
    
    # # Open the train, validation, and test set files
    # with open(os.path.join(artifact_folder, "fusion", "train.json"), "r") as f:
    #     train_sequences = json.load(f)
    # # with open(os.path.join(artifact_folder, "fusion", "valid.json"), "r") as f:
    # #     valid_sequences = json.load(f)
        
    # train_sequences = [s for s in train_sequences if s[-1] == "jazz"]
    
    # Call the Fusion_Dataset class
    data_loader = Fusion_Dataset(configs, train_sequences, mode="val", shuffle=True)
    # Get the first item
    for n, data in enumerate(data_loader):
        print(data["input_ids"], '\n')
        print(data["labels"], '\n')
        print(data["input_ids"].shape)
        print(data["labels"].shape)
        print(data["attention_mask"].shape)
        if n == 1000:
            break

    # # Write the generated sequences to a MIDI file
    # instrument_token = ('prefix', 'instrument', 'piano')
    # input_ids = data["input_ids"].tolist()
    # input_ids = [decode_tokenizer[token] for token in input_ids if token != 0]
    # special_words = ["SEP", "no_bridge", "classical", "pop", "jazz", "same_onset_ratio"]
    # input_ids = [token for token in input_ids if not any(special_word in token for special_word in special_words)]
    # input_ids = [instrument_token, "<S>"] + input_ids + ["<E>"]
    # labels = data["labels"].tolist()
    # labels = [decode_tokenizer[token] for token in labels if token != 0]
    # labels = [instrument_token] + labels

    # mid_dict = data_loader.aria_tokenizer.detokenize(input_ids)
    # mid = mid_dict.to_midi()
    # filename = "debug_input.mid"
    # mid.save(filename)

    # mid_dict = data_loader.aria_tokenizer.detokenize(labels)
    # mid = mid_dict.to_midi()
    # filename = "debug_labels.mid"
    # mid.save(filename)

    # input_ids = [instrument_token, "<S>"] + data["input_ids"] + ["<E>"]
    # input_ids = [decode_tokenizer[token] for token in input_ids if token != 0]
    # input_ids = [token for token in input_ids if token not in ["SEP", "no_bridge", "classical", "pop", "jazz"] or token[0] != "same_onset_ratio"]
    # labels = data["labels"]
    # labels = [instrument_token] + labels
    # labels = [decode_tokenizer[token] for token in labels if token != 0]

    # mid_dict = data_loader.aria_tokenizer.detokenize(input_ids)
    # mid = mid_dict.to_midi()
    # filename = "debug_input.mid"
    # mid.save(filename)

    # mid_dict = data_loader.aria_tokenizer.detokenize(labels)
    # mid = mid_dict.to_midi()
    # filename = "debug_labels.mid"
    # mid.save(filename)