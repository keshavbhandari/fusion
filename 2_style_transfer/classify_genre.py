from aria.data.midi import MidiDict
from aria.tokenizer import AbsTokenizer
import pickle
import argparse
import os
import yaml
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification
from data_loader import Genre_Classifier_Dataset
from corruptions import DataCorruption


def get_genre_probabilities(midi_file_path, tokenizer, model, dataset_obj, encoder_max_sequence_length, aria_tokenizer):
    # Read a midi file
    mid = MidiDict.from_midi(midi_file_path)
    # Tokenize the midi file
    tokens = aria_tokenizer.tokenize(mid)

    corruption_obj = DataCorruption()
    separated_sequence = corruption_obj.seperateitems(tokens)
    # Get the indices of the novelty tokens
    novelty_segments = [n for n, i in enumerate(separated_sequence) if '<N>' in i]
    all_segment_indices, _, _, _ = corruption_obj.get_segment_to_corrupt(separated_sequence, t_segment_ind=2, exclude_idx=novelty_segments)

    t_segment_ind = 0
    n_iterations = len(all_segment_indices) - 1
    jump_every = 1
    # Initialize tqdm
    progress_bar = tqdm(total=n_iterations)
    genre_probs = {'classical': [], 'pop': [], 'jazz': []}

    while t_segment_ind < n_iterations:
        # Get the cropped sequence
        input_tokens = dataset_obj.get_cropped_sequence(tokens, meta_data=None, t_segment_ind=t_segment_ind)

        # Tokenize the sequences
        input_tokens = [tokenizer[tuple(token)] if isinstance(token, list) else tokenizer[token] for token in input_tokens]

        # Pad the sequences
        if len(input_tokens) < encoder_max_sequence_length:
            input_tokens = F.pad(torch.tensor(input_tokens), (0, encoder_max_sequence_length - len(input_tokens))).to(torch.int64)
        else:
            input_tokens = torch.tensor(input_tokens[0:encoder_max_sequence_length]).to(torch.int64)

        # Attention mask based on non-padded tokens of the phrase
        attention_mask = torch.where(input_tokens != 0, 1, 0).type(torch.bool)

        # Get the prediction
        outputs = model(input_tokens.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
        logits = outputs.logits
        # Get probabilities from the logits
        probs = F.softmax(logits, dim=1)
        genre_probs['classical'].append(probs[0][0].item())
        genre_probs['pop'].append(probs[0][1].item())
        genre_probs['jazz'].append(probs[0][2].item())

        t_segment_ind += jump_every
        # Update the progress bar
        progress_bar.update(jump_every)

    progress_bar.close()

    # Print average probabilities by genre
    print(f"Average probability for classical: {sum(genre_probs['classical'])/len(genre_probs['classical'])}")
    print(f"Average probability for pop: {sum(genre_probs['pop'])/len(genre_probs['pop'])}")
    print(f"Average probability for jazz: {sum(genre_probs['jazz'])/len(genre_probs['jazz'])}")

    return genre_probs


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

    # Get the encoder max sequence length
    encoder_max_sequence_length = configs['classifier_model']['encoder_max_sequence_length']

    # Get tokenizer
    tokenizer_filepath = os.path.join(artifact_folder, "style_transfer", "vocab_corrupted.pkl")
    # Load the tokenizer dictionary
    with open(tokenizer_filepath, "rb") as f:
        tokenizer = pickle.load(f)

    aria_tokenizer = AbsTokenizer()

    # Open the test set files
    with open(os.path.join(artifact_folder, "style_transfer", "fine_tuning_valid.pkl"), "rb") as f:
        valid_sequences = pickle.load(f)

    # Load the model
    model_path = os.path.join(artifact_folder, "style_transfer", "classifier_model")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    print("Model loaded")

    # Load the dataset
    dataset_obj = Genre_Classifier_Dataset(configs, valid_sequences, mode="eval", shuffle=False)


    generated_midi_file_path = "/homes/kb658/fusion/output/generated_debussy-clair-de-lune.mid"
    original_midi_file_path = "/homes/kb658/fusion/output/original_debussy-clair-de-lune.mid"

    # generated_midi_file_path = "/homes/kb658/fusion/output/generated_schumann_kinderszenen_15_7_(c)harfesoft.mid"
    # original_midi_file_path = "/homes/kb658/fusion/output/original_schumann_kinderszenen_15_7_(c)harfesoft.mid"

    genre_probs = get_genre_probabilities(original_midi_file_path, tokenizer, model, dataset_obj, encoder_max_sequence_length, aria_tokenizer)
    genre_probs = get_genre_probabilities(generated_midi_file_path, tokenizer, model, dataset_obj, encoder_max_sequence_length, aria_tokenizer)

    # # Save the genre probabilities
    # with open("genre_probs.pkl", "wb") as f:
    #     pickle.dump(genre_probs, f)