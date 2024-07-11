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

from corruptions import DataCorruption

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.utils import flatten, unflatten, skyline, unflatten_corrupted


class Fusion_Dataset(Dataset):
    def __init__(self, configs, data_list, mode="train", shuffle = False):
        self.mode = mode
        self.data_list = data_list
        if shuffle:
            random.shuffle(self.data_list)

        # Artifact folder
        self.artifact_folder = configs['raw_data']['artifact_folder']
        # Load encoder tokenizer json file dictionary
        tokenizer_filepath = os.path.join(self.artifact_folder, "fusion", "vocab_corrupted.pkl")

        self.aria_tokenizer = AbsTokenizer()
        # Load the pickled tokenizer dictionary
        with open(tokenizer_filepath, 'rb') as f:
            self.tokenizer = pickle.load(f)

        self.corruption_obj = DataCorruption()

        # Get the maximum sequence length
        self.encoder_max_sequence_length = configs['model']['fusion_model']['encoder_max_sequence_length']
        self.decoder_max_sequence_length = configs['model']['fusion_model']['decoder_max_sequence_length']

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
    
    
    def get_corrupted_sequence(self, sequence, meta_data):
        # Take the 3rd token as the start token until the 2nd last token
        sequence = sequence[2:-1]

        # Call the flatten function
        flattened_sequence = flatten(sequence, add_special_tokens=True)

        # Corrupt the flattened sequence
        context_before = random.randint(1, 5)
        context_after = 1
        output_dict = self.corruption_obj.apply_random_corruption(flattened_sequence, 
                                                                  context_before=context_before, 
                                                                  context_after=context_after, 
                                                                  meta_data=meta_data, 
                                                                  t_segment_ind=None,
                                                                  inference=False,
                                                                  corruption_type=None)
        corrupted_sequence = output_dict['corrupted_sequence']
        original_segment = output_dict['original_segment']

        corrupted_sequence, original_segment = unflatten_corrupted(corrupted_sequence), unflatten(original_segment)

        return corrupted_sequence, original_segment


    def __getitem__(self, idx):
        sequence_info = self.data_list[idx]
        genre = sequence_info[-1]
        tokenized_sequence = sequence_info[0]

        meta_tokens = [genre]

        if genre in ["classical", "jazz"]:
            # Apply augmentations
            pitch_aug_function = self.aria_tokenizer.export_pitch_aug(12)
            tokenized_sequence = pitch_aug_function(tokenized_sequence)

            input_tokens, original = self.get_corrupted_sequence(tokenized_sequence, meta_tokens)
        
        # Add the start and end tokens
        original = ["<S>"] + original + ["<E>"]

        # Tokenize the sequences
        input_tokens = [self.tokenizer[tuple(token)] if isinstance(token, list) else self.tokenizer[token] for token in input_tokens]
        original = [self.tokenizer[tuple(token)] if isinstance(token, list) else self.tokenizer[token] for token in original]

        # Pad the sequences
        if len(original) < self.decoder_max_sequence_length:
            original = F.pad(torch.tensor(original), (0, self.decoder_max_sequence_length - len(original))).to(torch.int64)
        else:
            original = torch.tensor(original[0:self.decoder_max_sequence_length]).to(torch.int64)
        if len(input_tokens) < self.encoder_max_sequence_length:
            input_tokens = F.pad(torch.tensor(input_tokens), (0, self.encoder_max_sequence_length - len(input_tokens))).to(torch.int64)
        else:
            input_tokens = torch.tensor(input_tokens[0:self.encoder_max_sequence_length]).to(torch.int64)

        # Attention mask based on non-padded tokens of the phrase
        attention_mask = torch.where(input_tokens != 0, 1, 0).type(torch.bool)

        train_data = {"input_ids": input_tokens, "labels": original, "attention_mask": attention_mask}

        return train_data





if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.normpath("configs/configs_style_transfer.yaml"),
                        help="Path to the config file")
    args = parser.parse_args()

    # Load config file
    with open(args.config, 'r') as f:
        configs = yaml.safe_load(f)
        
    artifact_folder = configs["raw_data"]["artifact_folder"]
    raw_data_folders = configs["raw_data"]["raw_data_folders"]

    # # Get tokenizer
    # tokenizer_filepath = os.path.join(artifact_folder, "fusion", "vocab_corrupted.pkl")
    # # Load the tokenizer dictionary
    # with open(tokenizer_filepath, "rb") as f:
    #     tokenizer = pickle.load(f)
    # # Reverse the tokenizer
    # decode_tokenizer = {v: k for k, v in tokenizer.items()}


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