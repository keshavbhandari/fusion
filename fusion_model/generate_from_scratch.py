import yaml
import json
import pickle
import os
import random
import numpy as np
import sys
import argparse
import torch
from torch.nn import functional as F
from transformers import EncoderDecoderModel
from torch.cuda import is_available as cuda_available
from aria.data.midi import MidiDict
from aria.tokenizer import AbsTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.utils import flatten, skyline


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
data_path = raw_data_folders['classical']['folder_path']
output_folder = os.path.join("/homes/kb658/fusion/output")
# Create output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get the encoder and decoder max sequence length
encoder_max_sequence_length = configs['model']['fusion_model']['encoder_max_sequence_length']
decoder_max_sequence_length = configs['model']['fusion_model']['decoder_max_sequence_length']

# Get tokenizer
tokenizer_filepath = os.path.join(artifact_folder, "fusion", "vocab.pkl")
# Load the tokenizer dictionary
with open(tokenizer_filepath, "rb") as f:
    tokenizer = pickle.load(f)

# Reverse the tokenizer
decode_tokenizer = {v: k for k, v in tokenizer.items()}

# Load the fusion model
fusion_model = EncoderDecoderModel.from_pretrained(os.path.join(artifact_folder, "fusion", "model"))
fusion_model.eval()
fusion_model.to("cuda" if cuda_available() else "cpu")
print("Fusion model loaded")


file_path = os.path.join("/homes/kb658/yinyang/output/yin_yang/NLB075160_01.mid")
# file_path = os.path.join("/import/c4dm-datasets/maestro-v3.0.0/2008/MIDI-Unprocessed_07_R2_2008_01-05_ORIG_MID--AUDIO_07_R2_2008_wav--2.midi")
mid = MidiDict.from_midi(file_path)
aria_tokenizer = AbsTokenizer()
tokenized_sequence = aria_tokenizer.tokenize(mid)
instrument_token = tokenized_sequence[0]
tokenized_sequence = tokenized_sequence[2:-1]
# Call the flatten function
# flattened_sequence = flatten(tokenized_sequence, add_special_tokens=True)
# Call the skyline function
# tokenized_sequence, harmony = skyline(flattened_sequence, diff_threshold=30, static_velocity=True)

n_t_tokens = len([token for token in tokenized_sequence if token == "<T>"])
same_onset_ratio_seq = []
for i in range(n_t_tokens+1):
    random_onset_ratio = np.random.choice([0.5, 0.55, 0.6, 0.65, 0.7, 0.75])
    # random_onset_ratio = np.random.choice([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    same_onset_ratio_seq.append(("same_onset_ratio", random_onset_ratio))
print("Same onset ratio sequence:", same_onset_ratio_seq)

tokenized_sequence = ["pop"] + same_onset_ratio_seq + ["SEP"] + tokenized_sequence
# tokenized_sequence = ["pop"] + ["SEP"] + tokenized_sequence

# Tokenize the sequence
tokenized_sequence = [tokenizer[tuple(token)] if isinstance(token, list) else tokenizer[token] for token in tokenized_sequence]
# Pad the sequences
if len(tokenized_sequence) < encoder_max_sequence_length:
    tokenized_sequence = F.pad(torch.tensor(tokenized_sequence), (0, encoder_max_sequence_length - len(tokenized_sequence))).to(torch.int64)
else:
    tokenized_sequence = torch.tensor(tokenized_sequence[0:decoder_max_sequence_length:]).to(torch.int64)

# Generate the sequence
input_ids = tokenized_sequence.unsqueeze(0).to("cuda" if cuda_available() else "cpu")
# Encode the input
encoder_outputs = fusion_model.encoder(input_ids)
# Prepare the initial decoder input
decoder_input_ids = torch.tensor([[tokenizer["<S>"]]]).to("cuda" if cuda_available() else "cpu")
# Set the temperature
temperature = 0.75
# Generation loop
max_length = decoder_max_sequence_length  # Maximum length of the generated sequence

for _ in range(max_length):
    # Pass decoder inputs and encoder outputs to the model
    outputs = fusion_model.decoder(decoder_input_ids, encoder_hidden_states=encoder_outputs.last_hidden_state)
    # Get the logits and apply temperature scaling
    logits = outputs.logits[:, -1, :] / temperature

    # Apply softmax to get probabilities and sample the next token
    probs = torch.nn.functional.softmax(logits, dim=-1)
    next_token_id = torch.multinomial(probs, num_samples=1)
    
    # Append the next token to the decoder input
    decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=-1)
    
    # Break if the end token is generated
    if next_token_id == tokenizer["<E>"]:
        break    

output = decoder_input_ids
# output = fusion_model.generate(input_ids, decoder_start_token_id=tokenizer["<S>"], max_length=decoder_max_sequence_length, num_beams=1, do_sample=True, early_stopping=False, temperature=0.8) # 0.75

# Decode the generated sequences
generated_sequences = [decode_tokenizer[token] for token in output[0].tolist()]
# Remove special tokens
generated_sequences = [token for token in generated_sequences if token not in ["<S>", "<E>", "<SEP>"]]
generated_sequences = [instrument_token, "<S>"] + generated_sequences + ["<E>"]

# Print the generated sequences
print("Generated sequences:", generated_sequences)

# Write the generated sequences to a MIDI file
mid_dict = aria_tokenizer.detokenize(generated_sequences)
mid = mid_dict.to_midi()
filename = os.path.basename(file_path)
mid.save(os.path.join(output_folder, filename))