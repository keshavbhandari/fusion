import math
import pickle
import random
import copy
import numpy as np
import miditoolkit
from music21 import stream, meter, note, metadata, tempo, converter, instrument, chord, key
import pandas as pd
import yaml
from ssmnet.core import SsmNetDeploy

def find_beats_in_bar(time_signature):
    if time_signature == "null" or time_signature is None:
        time_signature = "4/4"
    numerator = int(time_signature.split("_")[1].split("/")[0])
    denominator = int(time_signature.split("_")[1].split("/")[1])
    if denominator == 4:
        beats_in_bar = numerator * (denominator / 8) * 2
    elif denominator == 8:
        beats_in_bar = numerator * (denominator / 8) / 2
    elif denominator == 2:
        beats_in_bar = numerator * (denominator / 8) * 8
    elif denominator == 1:
        beats_in_bar = numerator * (denominator / 8) * 32
    return beats_in_bar

def annotation_to_encoding(annotation_file):
    key_signature = annotation_file['features']['tonic'][0]
    major_or_minor = annotation_file['features']['mode'][0]

    time_signature = annotation_file['features']['timesignature'][0]
    if time_signature == "null" or time_signature is None:
        time_signature = "4/4"
    numerator = int(time_signature.split("/")[0])
    denominator = int(time_signature.split("/")[1])
    if denominator == 4:
        beats_in_bar = numerator * (denominator / 8) * 2
    elif denominator == 8:
        beats_in_bar = numerator * (denominator / 8) / 2
    elif denominator == 2:
        beats_in_bar = numerator * (denominator / 8) * 8
    elif denominator == 1:
        beats_in_bar = numerator * (denominator / 8) * 32
    else: 
        return [], time_signature, key_signature, major_or_minor
    
    pitches = annotation_file['features']['midipitch']
    durations = annotation_file['features']['duration']
    next_note_rest_value = annotation_file['features']['restduration_frac']

    encoding = []
    bar = 0
    onset = 0
    for idx, pitch_value in enumerate(pitches):
        note_info = []
        if idx == 0:
            note_info.append([bar, onset, 0, pitches[idx], durations[idx], 91])
        else:
            # Check if previous note was a rest
            prev_rest = next_note_rest_value[idx-1]
            if prev_rest is None:
                rest = 0
            else:            
                if "/" in prev_rest:
                    rest = float(int(prev_rest.split("/")[0]) / int(prev_rest.split("/")[1]))
                else:
                    rest = int(prev_rest)
            
            onset += durations[idx-1] + rest  
            
            if onset >= beats_in_bar:
                previous_onset = encoding[-1][1]
                onset = (previous_onset + durations[idx-1] + rest) % beats_in_bar
                bar += 1
            note_info.append([bar, onset, 0, pitches[idx], durations[idx], 91])
        encoding+=note_info
    
    return encoding, time_signature, key_signature, major_or_minor


def encoding_to_midi(encoding, tempo_dict, time_signature, sharps_flats=None, midi_file_path="output.mid", write_midi=True, piece_name="Yin-Yang Prelude"):
    time_signature = time_signature.split("_")[1]

    # Create a Score
    score = stream.Score()
    score.metadata = metadata.Metadata()
    score.metadata.title = piece_name

    # Create a Part for the instrument
    part = stream.Part()

    # Set the initial tempo
    initial_tempo = tempo_dict.get(0, 120)
    part.append(tempo.MetronomeMark(number=initial_tempo))

    # Set the time signature
    time_signature = meter.TimeSignature(time_signature)
    # Add the time signature to the Part
    part.append(time_signature)

    # Set the key signature if sharps_flats is not None
    if sharps_flats is not None:
        key_signature = key.KeySignature(sharps_flats)
        part.append(key_signature)

    # Iterate through the MIDI data and create Note objects
    for entry in encoding:
        bar_number, onset_position, instrument_number, pitch, duration, velocity = entry[:6]

        # Create a Note
        n = note.Note(pitch, quarterLength=duration)
        n.volume.velocity = velocity

        # Calculate the offset position
        offset_position = bar_number * time_signature.barDuration.quarterLength + onset_position

        # Add the Note to the Part at the calculated offset position
        part.insert(offset_position, n)

        # Check if there is a tempo change for the next bar
        next_tempo = tempo_dict.get(bar_number + 1, None)
        if next_tempo is not None:
            part.append(tempo.MetronomeMark(number=next_tempo))

    # Add the Part to the Score
    score.append(part)

    # Write the Score to a MIDI file
    # midi_file_path = "output.mid"
    if write_midi:
        score.write('midi', fp=midi_file_path)
    else:
        score.write('musicxml', fp=midi_file_path)


pos_resolution = 4 # 16  # per beat (quarter note)
bar_max = 32
velocity_quant = 4
tempo_quant = 12  # 2 ** (1 / 12)
min_tempo = 16
max_tempo = 256
duration_max = 4  # 2 ** 8 * beat
max_ts_denominator = 6  # x/1 x/2 x/4 ... x/64
max_notes_per_bar = 1  # 1/64 ... 128/64 #
beat_note_factor = 4  # In MIDI format a note is always 4 beats
deduplicate = True
filter_symbolic = False
filter_symbolic_ppl = 16
trunc_pos = 2 ** 16  # approx 30 minutes (1024 measures)
sample_len_max = 1024  # window length max
sample_overlap_rate = 1.5
ts_filter = True
pool_num = 200
max_inst = 127
max_pitch = 127
max_velocity = 127
tracks_start = [16, 144, 997, 5366, 6921, 10489]
tracks_end = [143, 996, 5365, 6920, 10488, 11858]


inst_to_row = { '80':0, '32':1, '128':2,  '25':3, '0':4, '48':5, '129':6}
prog_to_abrv = {'0':'P','25':'G','32':'B','48':'S','80':'M','128':'D'}
track_name = ['lead', 'bass', 'drum', 'guitar', 'piano', 'string']

root_dict = {'C': 0, 'C#': 1, 'D': 2, 'Eb': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'Ab': 8, 'A': 9, 'Bb': 10, 'B': 11}
kind_dict = {'null': 0, 'm': 1, '+': 2, 'dim': 3, 'seven': 4, 'maj7': 5, 'm7': 6, 'm7b5': 7}
root_list = list(root_dict.keys())
kind_list = list(kind_dict.keys())

_CHORD_KIND_PITCHES = {
    'null': [0, 4, 7],
    'm': [0, 3, 7],
    '+': [0, 4, 8],
    'dim': [0, 3, 6],
    'seven': [0, 4, 7, 10],
    'maj7': [0, 4, 7, 11],
    'm7': [0, 3, 7, 10],
    'm7b5': [0, 3, 6, 10],
}

ts_dict = dict()
ts_list = list()
for i in range(0, max_ts_denominator + 1):  # 1 ~ 64
    for j in range(1, ((2 ** i) * max_notes_per_bar) + 1):
        ts_dict[(j, 2 ** i)] = len(ts_dict)
        ts_list.append((j, 2 ** i))
dur_enc = list()
dur_dec = list()
for i in range(duration_max):
    for j in range(pos_resolution):
        dur_dec.append(len(dur_enc))
        for k in range(2 ** i):
            dur_enc.append(len(dur_dec) - 1)

tokens_to_ids = {}
ids_to_tokens = []
pad_index = None
empty_index = None


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