from aria.data.midi import MidiDict
from aria.tokenizer import AbsTokenizer
import pickle
import argparse
import os
import yaml
from tqdm import tqdm
import torch
import torch.nn.functional as F
import librosa
import numpy as np
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
from transformers import AutoProcessor, ClapModel
from transformers import AutoModelForSequenceClassification
from data_loader import Genre_Classifier_Dataset
from corruptions import DataCorruption
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.utils import convert_midi_to_wav

def get_genre_probabilities(midi_file_path, tokenizer, model, dataset_obj, encoder_max_sequence_length, aria_tokenizer, verbose=True):
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

    if verbose:
        # Print average probabilities by genre
        print(f"Average probability for classical: {sum(genre_probs['classical'])/len(genre_probs['classical'])}")
        print(f"Average probability for pop: {sum(genre_probs['pop'])/len(genre_probs['pop'])}")
        print(f"Average probability for jazz: {sum(genre_probs['jazz'])/len(genre_probs['jazz'])}")

    return genre_probs


def generate_ssm(wav_file):

    y, sr = librosa.load(wav_file, sr=None, mono=True)

    # Duration of the audio file in minutes and seconds
    duration = librosa.get_duration(y=y, sr=sr)
    duration = f"{int(duration // 60)} minutes and {int(duration % 60)} seconds"

    #### Chromagram using librosa ####
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

    ## norm of chroma ##
    nrow, ncol = chroma.shape
    resolution = len(y) / nrow / sr
    chroma_norm = np.linalg.norm(chroma, axis=0)

    # Convert to torch tensor
    chroma = torch.from_numpy(chroma).float().cuda()
    chroma_norm = torch.from_numpy(chroma_norm).float().cuda()

    # Compute dot products
    dot_products = torch.matmul(chroma.T, chroma)

    # Compute norms product
    norms_product = chroma_norm[:, None] * chroma_norm[None, :]

    # Compute cosine similarity
    ssm = dot_products / norms_product

    #### normalize SSM ####
    ssm_normalized = (ssm - torch.min(ssm)) / (torch.max(ssm) - torch.min(ssm))

    #### Flip up-side-down ####
    ssm_flipped = torch.flipud(ssm_normalized)

    # Rescale ssm_normalized
    ssm_rescaled = F.interpolate(ssm_flipped.unsqueeze(0).unsqueeze(0), size=(512, 512), mode='bilinear').squeeze().cpu().numpy()

    return ssm_rescaled, duration


def compare_similarity_matrices(original_wav_file, generated_wav_file):
    # Generate the similarity matrices
    original_ssm, original_duration = generate_ssm(original_wav_file)
    generated_ssm, generated_duration = generate_ssm(generated_wav_file)

    # Ensure the matrices are numpy arrays
    original_ssm = np.array(original_ssm)
    generated_ssm = np.array(generated_ssm)

    # Check if the matrices have the same shape
    if original_ssm.shape != generated_ssm.shape:
        raise ValueError("Matrices must have the same dimensions")

    # Calculate Mean Squared Error (MSE)
    mse_value = mean_squared_error(original_ssm, generated_ssm)

    # Calculate Structural Similarity Index (SSIM)
    ssim_value = ssim(original_ssm, generated_ssm, data_range=generated_ssm.max() - generated_ssm.min())

    # Calculate Pearson Correlation Coefficient
    pearson_corr, _ = pearsonr(original_ssm.flatten(), generated_ssm.flatten())

    # Calculate Frobenius Norm
    frobenius_norm = np.linalg.norm(original_ssm - generated_ssm, 'fro')

    print(f"Mean Squared Error: {mse_value}")
    print(f"SSIM: {ssim_value}")
    print(f"Pearson Correlation: {pearson_corr}")
    print(f"Frobenius Norm: {frobenius_norm}")

    return {
        'MSE': mse_value,
        'SSIM': ssim_value,
        'Pearson Correlation': pearson_corr,
        'Frobenius Norm': frobenius_norm
    }


def compare_dtw_chroma(original_wav_file, generated_wav_file, hop_length = 1024):
    x_1, fs = librosa.load(original_wav_file)
    x_2, fs = librosa.load(generated_wav_file)
    x_1_chroma = librosa.feature.chroma_cqt(y=x_1, sr=fs,
                                         hop_length=hop_length)
    x_2_chroma = librosa.feature.chroma_cqt(y=x_2, sr=fs,
                                         hop_length=hop_length)
    
    D, wp = librosa.sequence.dtw(X=x_1_chroma, Y=x_2_chroma, metric='cosine')
    wp_s = librosa.frames_to_time(wp, sr=fs, hop_length=hop_length)

    dtw_cost = D[-1, -1]  # The final cost is at the bottom-right corner of the cost matrix

    print(f"DTW cost: {dtw_cost}")

    return dtw_cost


def clap_similarity_score(audio_sample, text, model, processor):
    # Preprocess inputs
    text_input = processor(text=[text], return_tensors="pt")
    audio_input = processor(audios=audio_sample, return_tensors="pt", sampling_rate=48000)

    # Generate embeddings
    text_embeddings = model.get_text_features(**text_input)
    audio_embeddings = model.get_audio_features(**audio_input)

    # Normalize embeddings
    text_embeddings = text_embeddings / torch.norm(text_embeddings, dim=1, keepdim=True)
    audio_embeddings = audio_embeddings / torch.norm(audio_embeddings, dim=1, keepdim=True)

    # Compute similarity score
    similarity_score = torch.mm(text_embeddings, audio_embeddings.T)

    return similarity_score


def compare_clap_similarity_score(wav_filepath, model, processor):
    audio_sample, sr = librosa.load(wav_filepath, sr=48000, mono=True)
    total_duration = audio_sample.shape[0] / sr

    segment_duration = 30  # Duration of each segment in seconds
    segment_samples = int(segment_duration * sr)  # Number of samples in each segment

    classical_similarity_scores = []
    jazz_similarity_scores = []
    for i in tqdm(range(0, int(total_duration / segment_duration)), desc="Processing segments", disable=False):
        start_sample = i * segment_samples
        end_sample = start_sample + segment_samples

        segment_audio = audio_sample[start_sample:end_sample]

        classical_similarity_score = clap_similarity_score(segment_audio, "Classical piano music", model, processor)
        jazz_similarity_score = clap_similarity_score(segment_audio, "Jazz piano music", model, processor)
        classical_similarity_scores.append(classical_similarity_score)
        jazz_similarity_scores.append(jazz_similarity_score)
    
    # Average the similarity score values with numpy
    classical_similarity_scores = np.mean(torch.cat(classical_similarity_scores).detach().numpy())
    jazz_similarity_scores = np.mean(torch.cat(jazz_similarity_scores).detach().numpy())
    print(f"Average similarity score for classical music: {classical_similarity_scores}")
    print(f"Average similarity score for jazz music: {jazz_similarity_scores}")

    return classical_similarity_scores, jazz_similarity_scores
        
    



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
    print("Genre classifier model loaded")

    clap_model = ClapModel.from_pretrained("laion/larger_clap_music_and_speech")
    clap_model.eval()
    processor = AutoProcessor.from_pretrained("laion/larger_clap_music_and_speech")
    print("CLAP model loaded")

    # Load the dataset
    dataset_obj = Genre_Classifier_Dataset(configs, valid_sequences, mode="eval", shuffle=False)

    original_midi_file_path = "/homes/kb658/fusion/evaluations/classical/19/original_19.mid"
    generated_midi_file_path = "/homes/kb658/fusion/evaluations/classical/19/experiment_2/target_style_jazz/corruption_name_skyline/corruption_rate_1.0/pass_10/generated_original_19.mid"

    origianl_genre_probs = get_genre_probabilities(original_midi_file_path, tokenizer, model, dataset_obj, encoder_max_sequence_length, aria_tokenizer, verbose=True)
    generated_genre_probs = get_genre_probabilities(generated_midi_file_path, tokenizer, model, dataset_obj, encoder_max_sequence_length, aria_tokenizer, verbose=True)

    # Convert the midi files to wav
    original_wav_file = original_midi_file_path.replace(".mid", ".wav")
    generated_wav_file = convert_midi_to_wav([generated_midi_file_path], "/homes/kb658/fusion/artifacts/soundfont.sf", 1)
    generated_wav_file = generated_wav_file[0]

    # Get the similarity matrices
    ssm_scores = compare_similarity_matrices(original_wav_file, generated_wav_file)

    # Get the DTW cost
    dtw_cost = compare_dtw_chroma(original_wav_file, generated_wav_file)

    # Get the CLAP similarity scores
    original_clap_scores = compare_clap_similarity_score(original_wav_file, clap_model, processor)
    generated_clap_scores = compare_clap_similarity_score(generated_wav_file, clap_model, processor)

    # Delete the generated wav file
    os.remove(generated_wav_file)

    # # Save the genre probabilities
    # with open("genre_probs.pkl", "wb") as f:
    #     pickle.dump(genre_probs, f)