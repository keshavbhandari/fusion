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
fusion_model = EncoderDecoderModel.from_pretrained(os.path.join(artifact_folder, "fusion", "fusion_model_static_skyline"))
# fusion_model = EncoderDecoderModel.from_pretrained(os.path.join(artifact_folder, "fusion", "fusion_model"))
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

# Add cfr and cd conditions to the tokenized sequence
conditioned_sequence = []
conditioned_sequence.append(("cfr", 0.85))
conditioned_sequence.append(("cd", 2.0))
for i in range(len(tokenized_sequence)):
    if tokenized_sequence[i] == "<T>":
        random_cfr = np.random.choice([0.7, 0.75, 0.8, 0.85, 0.9])
        random_cd = np.random.choice([2.0, 2.5, 3.0, 3.5, 4.0, 4.5]) #5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0
        print(random_cfr, random_cd)
        conditioned_sequence.append(tokenized_sequence[i])
        conditioned_sequence.append(("cfr", random_cfr))
        conditioned_sequence.append(("cd", random_cd))
    else:
        conditioned_sequence.append(tokenized_sequence[i])

tokenized_sequence = ["classical"] + ["SEP"] + conditioned_sequence
# tokenized_sequence = conditioned_sequence

# Tokenize the sequence
tokenized_sequence = [tokenizer[tuple(token)] if isinstance(token, list) else tokenizer[token] for token in tokenized_sequence]
# Pad the sequences
if len(tokenized_sequence) < encoder_max_sequence_length:
    tokenized_sequence = F.pad(torch.tensor(tokenized_sequence), (0, encoder_max_sequence_length - len(tokenized_sequence))).to(torch.int64)
else:
    tokenized_sequence = torch.tensor(tokenized_sequence[0:decoder_max_sequence_length:]).to(torch.int64)

# Generate the sequence
input_ids = tokenized_sequence.unsqueeze(0).to("cuda" if cuda_available() else "cpu")


def generate_sequence(model, tokenizer, input_ids, max_length, temperature=1.0, top_k=0, top_p=0.0):
    # Set the model to evaluation mode
    model.eval()
    
    # Move input to the same device as the model
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attn_mask = torch.where(input_ids != 0, 1, 0).to(device)
    # Prepare the initial decoder input
    decoder_input_ids = torch.tensor([[tokenizer["<S>"]]]).to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get model output
            batch = {'input_ids': input_ids, 'attention_mask': attn_mask}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], decoder_input_ids=decoder_input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < top_k_logits[:, [-1]]] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[:, indices_to_remove] = float('-inf')
            
            # Convert logits to probabilities
            probs = F.softmax(next_token_logits, dim=-1) 
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1) # check shape
            
            # Append the next token to the sequence
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)
            
            # Check if we've generated an EOS token
            if next_token.item() == tokenizer["<E>"]:
                break
    
    return decoder_input_ids


# Generate the sequence

output = generate_sequence(fusion_model, tokenizer, input_ids, decoder_max_sequence_length, temperature=0.0001, top_k=50, top_p=1.0)
# output = fusion_model.generate(input_ids, decoder_start_token_id=tokenizer["<S>"], max_length=decoder_max_sequence_length, num_beams=1, do_sample=True, early_stopping=False, temperature=0.0001, top_k=50, top_p=1.0) # 0.75

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