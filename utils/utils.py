import math
import pickle
import random
import copy
import numpy as np
import miditoolkit
import pandas as pd
import yaml
import os
from tqdm import tqdm
import subprocess
from concurrent.futures import ProcessPoolExecutor
from ssmnet.core import SsmNetDeploy


# Define a function to flatten the tokenized sequence
def flatten(sequence, add_special_tokens=True):
    flattened_sequence = []
    note_info = []
    for i in range(len(sequence)):
        if add_special_tokens:
            if sequence[i] == "<T>" or sequence[i] == "<D>":
                flattened_sequence.append(sequence[i])
        if sequence[i][0] == "piano":
            note_info.append(sequence[i][1])
            note_info.append(sequence[i][2])
        elif sequence[i][0] == "onset":
            note_info.append(sequence[i][1])
        elif sequence[i][0] == "dur":
            note_info.append(sequence[i][1])
            flattened_sequence.append(note_info) 
            note_info = []

    return flattened_sequence

def parse_generation(sequence, add_special_tokens=True):
    flattened_sequence = []
    note_info = {}
    for i in range(len(sequence)):
        if add_special_tokens:
            if sequence[i] == "<T>" or sequence[i] == "<D>":
                flattened_sequence.append(sequence[i])
        if sequence[i][0] == "piano":
            note_info['pitch'] = sequence[i][1]
            note_info['velocity'] = sequence[i][2]
            # Arrange the note info in the following order: [pitch, velocity, onset, duration]
            if len(note_info) == 4:
                note_info = [note_info['pitch'], note_info['velocity'], note_info['onset'], note_info['duration']]
                flattened_sequence.append(note_info) 
                note_info = {}
        elif sequence[i][0] == "onset":
            note_info['onset'] = sequence[i][1]
        elif sequence[i][0] == "dur":
            note_info['duration'] = sequence[i][1]

    sequence = copy.deepcopy(flattened_sequence)
    unflattened_sequence = []
    for i in range(len(sequence)):
        if sequence[i] == "<T>" or sequence[i] == "<D>":
            unflattened_sequence.append(sequence[i])
            continue
        elif type(sequence[i]) == tuple:
            unflattened_sequence.append(sequence[i])
        else:
            note_info = ("piano", sequence[i][0], sequence[i][1])
            unflattened_sequence.append(note_info)
            note_info = ("onset", sequence[i][2])
            unflattened_sequence.append(note_info)
            note_info = ("dur", sequence[i][3])
            unflattened_sequence.append(note_info)            
            note_info = []

    return unflattened_sequence

# Reverse the flattened function
def unflatten(sequence, static_velocity=False):
    unflattened_sequence = []
    for i in range(len(sequence)):
        if sequence[i] == "<T>" or sequence[i] == "<D>":
            unflattened_sequence.append(sequence[i])
            continue
        elif type(sequence[i]) == tuple:
            unflattened_sequence.append(sequence[i])
        else:
            note_info = ("onset", sequence[i][2])
            unflattened_sequence.append(note_info)
            note_info = ("dur", sequence[i][3])
            unflattened_sequence.append(note_info)
            if static_velocity:
                note_info = ("piano", sequence[i][0], 90)
            else:
                note_info = ("piano", sequence[i][0], sequence[i][1])
            unflattened_sequence.append(note_info)
            note_info = []

    return unflattened_sequence

def unflatten_for_aria(sequence):
    unflattened_sequence = []
    for i in range(len(sequence)):
        if type(sequence[i]) == str:
            unflattened_sequence.append(sequence[i])
            continue
        elif type(sequence[i]) == tuple:
            unflattened_sequence.append(sequence[i])
        else:
            note_info = ("piano", sequence[i][0], sequence[i][1])
            unflattened_sequence.append(note_info)
            note_info = ("onset", sequence[i][2])
            unflattened_sequence.append(note_info)
            note_info = ("dur", sequence[i][3])
            unflattened_sequence.append(note_info)            
            note_info = []

    return unflattened_sequence

# Reverse the corrupted flattened function
def unflatten_corrupted(sequence, static_velocity=False):
    unflattened_sequence = []
    for i in range(len(sequence)):
        if type(sequence[i]) == str:
            unflattened_sequence.append(sequence[i])
            continue
        elif type(sequence[i]) == tuple:
            unflattened_sequence.append(sequence[i])
        else:
            if type(sequence[i][2]) == int:
                note_info = ("onset", sequence[i][2])
            else:
                note_info = 'O'
            unflattened_sequence.append(note_info)
            if type(sequence[i][3]) == int:
                note_info = ("dur", sequence[i][3])
            else:
                note_info = 'D'
            unflattened_sequence.append(note_info)
            if type(sequence[i][0]) == int:
                if static_velocity:
                    note_info = ("piano", sequence[i][0], 90)
                else:
                    note_info = ("piano", sequence[i][0], sequence[i][1])
            else:
                note_info = 'PVM'                
            unflattened_sequence.append(note_info)
            note_info = []

    return unflattened_sequence


class Segment_Novelty:
    def __init__(self, config_file, audio_file):
        
        with open(config_file, "r", encoding="utf-8") as fid:
            config_d = yaml.safe_load(fid)

        self.ssmnet = SsmNetDeploy(config_d)
        self.audio_file = audio_file

    def m_get_features(self, audio_file):
        return self.ssmnet.m_get_features(audio_file)

    def m_get_ssm_novelty(self, feat_3m):
        return self.ssmnet.m_get_ssm_novelty(feat_3m)

    def m_get_boundaries(self, hat_novelty_np, time_sec_v):
        return self.ssmnet.m_get_boundaries(hat_novelty_np, time_sec_v)

    def m_plot(self, hat_ssm_np, hat_novelty_np, hat_boundary_frame_v, output_pdf_file):
        return self.ssmnet.m_plot(hat_ssm_np, hat_novelty_np, hat_boundary_frame_v, output_pdf_file)

    def m_export_csv(self, hat_boundary_sec_v, output_csv_file):
        return self.ssmnet.m_export_csv(hat_boundary_sec_v, output_csv_file)
    
    def max_items(self, values, indices, n):
        sorted_items = np.argsort(values)[::-1]
        top_items = sorted_items[:n]
        return indices[top_items]
    
    def find_novelty_timestamps(self, top_peak_indices, timestampsarray):
        return timestampsarray[top_peak_indices]
    
    def locate_peak_timestamps(self, indices, values, n_peaks):
        top_novelty_indices = self.max_items(values, indices, n_peaks)
        _, time_sec_v = self.m_get_features(self.audio_file)
        timestamps = self.find_novelty_timestamps(top_novelty_indices, time_sec_v)
        sorted_timestamps = np.sort(timestamps)
        return sorted_timestamps

    def get_peak_timestamps(self, audio_file, n_peaks):
        feat_3m, time_sec_v = self.m_get_features(audio_file)
        _, hat_novelty_np = self.m_get_ssm_novelty(feat_3m)
        _, hat_boundary_frame_v = self.m_get_boundaries(hat_novelty_np, time_sec_v)
        all_novelty_values = hat_novelty_np[hat_boundary_frame_v]
        return self.locate_peak_timestamps(hat_boundary_frame_v, all_novelty_values, n_peaks)



# Skyline function for separating melody and harmony from the tokenized sequence
def skyline(sequence: list, diff_threshold=50, static_velocity=True, pitch_threshold=None):
    
    if pitch_threshold is None:
        pitch_threshold = 0
    
    melody = []
    harmony = []
    pointer_pitch = sequence[0][0]
    pointer_velocity = sequence[0][1]
    pointer_onset = sequence[0][2]
    pointer_duration = sequence[0][3]
    i = 0

    for i in range(1, len(sequence)):
        if type(sequence[i]) != str:
            current_pitch = sequence[i][0]
            current_velocity = sequence[i][1]
            current_onset = sequence[i][2]
            current_duration = sequence[i][3]

            if type(sequence[i-1]) == str and type(sequence[i-2]) == str:
                diff_curr_prev_onset = 5000
            elif type(sequence[i-1]) == str and type(sequence[i-2]) != str:
                diff_curr_prev_onset = abs(current_onset - sequence[i-2][2])
            else:
                diff_curr_prev_onset = abs(current_onset - sequence[i-1][2])
            
            # Check if the difference between the current onset and the previous onset is greater than the threshold and the pitch is greater than the threshold
            if diff_curr_prev_onset > diff_threshold:

                if pointer_pitch > pitch_threshold:
                    # Append the previous note
                    if static_velocity:
                        melody.append([pointer_pitch, 90, pointer_onset, pointer_duration])                        
                    else:
                        melody.append([pointer_pitch, pointer_velocity, pointer_onset, pointer_duration])
                
                # Update the pointer
                pointer_pitch = current_pitch
                pointer_velocity = current_velocity
                pointer_onset = current_onset
                pointer_duration = current_duration            
            else:
                if current_pitch > pointer_pitch:
                    # Append the previous note
                    harmony.append(("piano", pointer_pitch, pointer_velocity))
                    harmony.append(("onset", pointer_onset))
                    harmony.append(("dur", pointer_duration))
                    # Append <t> based on condition
                    if current_onset < pointer_onset:
                        harmony.append("<T>")
                    # Update the pointer
                    pointer_pitch = current_pitch
                    pointer_velocity = current_velocity
                    pointer_onset = current_onset
                    pointer_duration = current_duration
                else:
                    # Append the previous note
                    harmony.append(("piano", current_pitch, current_velocity))
                    harmony.append(("onset", current_onset))
                    harmony.append(("dur", current_duration))
                    # Append <t> based on condition
                    if current_onset < pointer_onset:
                        harmony.append("<T>")
                    continue

            # Append the last note
            if i == len(sequence) - 1: 
                if diff_curr_prev_onset > diff_threshold:
                    if pointer_pitch > pitch_threshold:
                        if static_velocity:
                            melody.append([pointer_pitch, 90, pointer_onset, pointer_duration])
                        else:
                            melody.append([pointer_pitch, pointer_velocity, pointer_onset, pointer_duration])
                else:
                    if current_pitch > pointer_pitch:
                        if current_pitch > pitch_threshold:
                            if static_velocity:
                                melody.append(["piano", current_pitch, 90, current_onset, current_duration])
                            else:
                                melody.append(["piano", current_pitch, current_velocity, current_onset, current_duration])
                    else:
                        harmony.append(("piano", current_pitch, current_velocity))
                        harmony.append(("onset", current_onset))
                        harmony.append(("dur", current_duration))

        if sequence[i-1] == "<T>":
            melody.append("<T>")
        
        if sequence[i] == "<D>":
            melody.append("<D>")

    return melody, harmony


# Define a function to round a value to the nearest 05
def round_to_nearest_n(input_value, round_to=0.05):
    rounded_value = round(round(input_value / round_to) * round_to, 2)
    return rounded_value


def get_chord_info(chunk):
    if len(chunk) < 2:
        return 0, 0, pd.DataFrame()
    
    df = pd.DataFrame(chunk, columns=["pitch", "velocity", "onset", "duration"])
    df['previous_onset'] = df['onset'].shift(1).fillna(0).astype(int)
    df['next_onset'] = df['onset'].shift(-1).fillna(0).astype(int)
    df['same_onset_previous'] = np.where((abs(df['onset'] - df['previous_onset']) <= 30), 1, 0)
    df['same_onset_next'] = np.where((abs(df['onset'] - df['next_onset']) <= 30), 1, 0)
    df['same_onset'] = np.where((df['same_onset_previous'] == 0) & (df['same_onset_next'] == 1), 1, 0)

    counter = 0
    group = 0
    new_column = []

    for value in df['same_onset']:
        if value == 1:
            counter += 1
            group = counter
        new_column.append(group)

    df['new_same_onset'] = np.where((df['same_onset_previous'] == 0) & (df['same_onset_next'] == 0), 0, new_column)

    len_df = len(df)
    df.fillna(0, inplace=True)
    df_filtered = df.loc[df['new_same_onset']!=0]
    
    if len(df_filtered) == 0:
        return 0, 0, pd.DataFrame()
    
    cfr = len(df_filtered) / len_df
    cd = df_filtered['new_same_onset'].mean()
    cd = 8 if cd > 8 else cd

    cfr = round_to_nearest_n(cfr, round_to=0.05)
    cd = round_to_nearest_n(cd, round_to=0.25)

    return cfr, cd, df


def get_conditions(separated_list):
    cfr_list = []
    cd_list = []
    for i in range(len(separated_list)):
        cfr, cd, df = get_chord_info(separated_list[i])
        cfr_list.append(("cfr", cfr))
        cd_list.append(("cd", cd))
    return cfr_list, cd_list


# Separate the list of lists based on the <T> token
def separate_list(sequence):
    separated_list = []
    sublist = []
    for i in range(len(sequence)):
        if sequence[i] == "<T>":
            separated_list.append(sublist)
            sublist = []
        elif type(sequence[i]) == list:
            sublist.append(sequence[i])
    if sublist:
        separated_list.append(sublist)
    return separated_list


def interleave_conditions(flattened_sequence, cfr_list, cd_list):
    conditioned_flattened_sequence = []
    for n, i in enumerate(flattened_sequence):
        if n == 0:
            cfr_condition = cfr_list.pop(0)
            cd_condition = cd_list.pop(0)
            conditioned_flattened_sequence.append(cfr_condition)
            conditioned_flattened_sequence.append(cd_condition)
            conditioned_flattened_sequence.append(i)
        elif i == "<T>":
            conditioned_flattened_sequence.append(i)
            if len(cfr_list) > 0:
                cfr_condition = cfr_list.pop(0)
                cd_condition = cd_list.pop(0)
                conditioned_flattened_sequence.append(cfr_condition)
                conditioned_flattened_sequence.append(cd_condition)
        else:
            conditioned_flattened_sequence.append(i)

    if len(cfr_list) > 0:
        conditioned_flattened_sequence.append(cfr_list.pop(0))
        conditioned_flattened_sequence.append(cd_list.pop(0))

    return conditioned_flattened_sequence


def save_wav(filepath, soundfont_path="soundfont.sf"):
    # Extract the directory and the stem (filename without extension)
    directory = os.path.dirname(filepath)
    stem = os.path.splitext(os.path.basename(filepath))[0]

    # Construct the full paths for MIDI and WAV files
    midi_filepath = os.path.join(directory, f"{stem}.mid")
    wav_filepath = os.path.join(directory, f"{stem}.wav")

    # Run the fluidsynth command to convert MIDI to WAV
    process = subprocess.Popen(
        f"fluidsynth -r 48000 {soundfont_path} -g 1.0 --quiet --no-shell {midi_filepath} -T wav -F {wav_filepath} > /dev/null",
        shell=True
    )
    process.wait()

    return wav_filepath


def convert_midi_to_wav(filepaths, soundfont_path="../artifacts/soundfont.sf", max_workers=32, verbose=True):
    if verbose:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Use tqdm to track progress
            results = list(tqdm(executor.map(save_wav, filepaths, [soundfont_path]*len(filepaths)), total=len(filepaths), desc="Converting MIDI to WAV"))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(save_wav, filepaths, [soundfont_path]*len(filepaths)))
    return results

# # Find all notes which have the same onset time before n notes and print the average ratio of same note onsets
# def chord_density_ratio(flattened_sequence):
#     same_onset_notes = []
#     total_notes = 0
#     previous_note = None
#     match = False
#     averages = []
#     for i in range(len(flattened_sequence)):
#         # if i % time_frame == 0 and i != 0:
#         if flattened_sequence[i] == "<T>":
#             # Calculate the average number of notes with the same onset time
#             if total_notes == 0:
#                 continue
#             average = len(same_onset_notes) / total_notes
#             average = round_to_nearest_n(average)
#             average = ("same_onset_ratio", average)
#             averages.append(average)
#             same_onset_notes = []
#             total_notes = 0
#             continue
#         elif type(flattened_sequence[i]) == str:
#             continue
#         else:
#             if i == 0:
#                 previous_note = flattened_sequence[i]
#             elif i < len(flattened_sequence)-1:
#                 onset_time = flattened_sequence[i][2]
#                 if previous_note is None:
#                     previous_note = flattened_sequence[i]
#                     total_notes += 1
#                     continue
#                 # Look at previous note and check if the onset time is the same. If it is then add the previous note to the same_onset_notes list
#                 elif abs(onset_time - previous_note[2]) <= 30:
#                     same_onset_notes.append(previous_note)
#                     previous_note = flattened_sequence[i]
#                     total_notes += 1
#                     match = True
#                 else:
#                     if match:
#                         same_onset_notes.append(previous_note)
#                     previous_note = flattened_sequence[i]
#                     total_notes += 1
#                     match = False
#             else:
#                 onset_time = flattened_sequence[i][2]
#                 if previous_note is None:
#                     previous_note = flattened_sequence[i]   
#                 elif abs(onset_time - previous_note[2]) <= 30:
#                     same_onset_notes.append(previous_note)
#                     total_notes += 1
#                 else:
#                     if match:
#                         same_onset_notes.append(previous_note)
#                     total_notes += 1

#     # Calculate the average number of notes with the same onset time for the last time frame
#     if len(same_onset_notes) > 0:
#         average = len(same_onset_notes) / total_notes
#         average = round_to_nearest_n(average)
#         average = ("same_onset_ratio", average)
#         averages.append(average)

#     return averages


# def get_note_lengths(flattened_sequence):    
#     notes_before_t_token = []
#     tracker = []
#     for note in flattened_sequence:
#         if note == "<T>":
#             notes_before_t_token.append(len(tracker))
#             tracker = []
#         else:
#             tracker.append(note)
#     return notes_before_t_token

# def get_mtr(melody_notes, melody_harmony_notes):
#     if len(melody_notes) == 0:
#         return []
#     elif len(melody_notes) > len(melody_harmony_notes):
#         return []

#     ratios = []
#     for idx, n_melody in enumerate(melody_notes):
#         n_melody_harmony = melody_harmony_notes[idx]
#         n_melody_harmony = 1 if n_melody_harmony == 0 else n_melody_harmony

#         # Get the ratio
#         ratio = round_to_nearest_n(n_melody / n_melody_harmony, 0.05)
#         if ratio > 1:
#             ratio = 1.0
#         ratios.append(("melody_density_ratio", ratio))
#     return ratios

# def get_mtr_ratio(melody_flattened_notes, flattened_notes):
#     melody_notes = get_note_lengths(melody_flattened_notes)
#     melody_harmony_notes = get_note_lengths(flattened_notes)
#     ratios = get_mtr(melody_notes, melody_harmony_notes)
#     return ratios