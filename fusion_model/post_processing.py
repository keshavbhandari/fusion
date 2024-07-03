import os
import sys
from aria.tokenizer import AbsTokenizer
from aria.data.midi import MidiDict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.utils import flatten, skyline


output_folder = os.path.join("/homes/kb658/fusion/output/")
file_path = "/homes/kb658/fusion/output/NLB075160_01.mid"
# file_path = os.path.join("/import/c4dm-datasets/maestro-v3.0.0/2008/MIDI-Unprocessed_07_R2_2008_01-05_ORIG_MID--AUDIO_07_R2_2008_wav--2.midi")
mid = MidiDict.from_midi(file_path)
aria_tokenizer = AbsTokenizer()
tokenized_sequence = aria_tokenizer.tokenize(mid)
instrument_token = tokenized_sequence[0]
generated_sequences = tokenized_sequence[2:-1]


flattened_sequence = flatten(generated_sequences, add_special_tokens=True)
print(len(flattened_sequence))
print(flattened_sequence, '\n')

def quantize_to_nearest_tenth(value):
    return round(value / 10) * 10

# Define the arpeggios for the queue
def get_arpeggios(queue, next_onset_time, arpeggiation_type="up"):
    """
    Arpeggiate the queue
    :param queue: The queue
    :param arpeggiation_type: The type of arpeggiation
    :return: The arpeggiated queue
    """
    if len(queue) <= 2:
        return queue

    # Get minimum onset time in queue
    min_onset_time = min([note[2] for note in queue])
    onset_diff = next_onset_time - min_onset_time if next_onset_time > min_onset_time else 4990 - min_onset_time
    # onset_delta = quantize_to_nearest_tenth(onset_diff / len(queue))
    onset_delta = 250
    n_arpeggiate_notes = onset_diff // onset_delta
    if onset_delta < 100:
        return queue
    
    if arpeggiation_type == "up":
        queue = sorted(queue, key=lambda x: x[0])
    elif arpeggiation_type == "down":
        queue = sorted(queue, key=lambda x: x[0], reverse=True)
    
    arpeggiated_queue = []
    if arpeggiation_type == "up":
        # Pop out last element of queue
        melody_note = queue.pop()
    else:
        # Pop out first element of queue
        melody_note = queue.pop(0)        
    arpeggiated_queue.append(tuple(melody_note))

    for i, note in enumerate(queue):
        if i < n_arpeggiate_notes:
            note = list(note)
            next_onset_time = min_onset_time + i * onset_delta
            note[2] = next_onset_time
            arpeggiated_queue.append(tuple(note))
        else:
            note = list(note)
            note[2] = next_onset_time
            arpeggiated_queue.append(tuple(note))
    
    return arpeggiated_queue


# Define the arpeggiate function
def arpeggiate(flattened_sequence, arpeggiation_type="up"):
    """
    Arpeggiate the sequence
    :param flattened_sequence: The flattened sequence
    :param arpeggiation_type: The type of arpeggiation
    :return: The arpeggiated sequence
    """
    arpeggiated_sequence = []
    queue = []
    onset_in_queue = None
    threshold=50
    for n, note in enumerate(flattened_sequence):
        if isinstance(note, list):
            if len(queue) == 0:
                queue.append(note)
                onset_in_queue = note[2]
                continue
            else:
                # Check if the note's onset is within the threshold
                # If the note's onset is within the threshold, add it to the queue
                # If the note's onset is not within the threshold, arpeggiate the existing queue if there are any notes and append the notes to the arpeggiated sequence and initialize the queue again
                
                if abs(note[2] - onset_in_queue) < threshold:
                    queue.append(note)
                else:
                    if arpeggiation_type == "up":
                        queue = get_arpeggios(queue, note[2], "up")
                    elif arpeggiation_type == "down":
                        queue = get_arpeggios(queue, note[2], "down")

                    for chord_note in queue:
                        arpeggiated_sequence.append(chord_note)
                    queue = []
                    queue.append(note)
                    onset_in_queue = note[2]
        
        # If the last note is reached, arpeggiate the queue
        if len(queue) > 0 and n == len(flattened_sequence) - 1:
            if arpeggiation_type == "up":
                queue = sorted(queue, key=lambda x: x[0])

            elif arpeggiation_type == "down":
                queue = sorted(queue, key=lambda x: x[0], reverse=True)
            for chord_note in queue:
                arpeggiated_sequence.append(chord_note)

    return arpeggiated_sequence

# Reverse the flattened function
def unflatten(sequence):
    unflattened_sequence = []
    for i in range(len(sequence)):
        note_info = ("piano", sequence[i][0], sequence[i][1])
        unflattened_sequence.append(note_info)
        note_info = ("onset", sequence[i][2])
        unflattened_sequence.append(note_info)
        note_info = ("dur", sequence[i][3])
        unflattened_sequence.append(note_info)
        note_info = []
        
        if i < len(sequence)-1:
            if sequence[i+1][2] < sequence[i][2]:
                unflattened_sequence.append("<T>")

    return unflattened_sequence

arpeggiated_sequence = arpeggiate(flattened_sequence, arpeggiation_type="down")
unflattened_sequence = unflatten(arpeggiated_sequence)
print(len(unflattened_sequence))
print(unflattened_sequence)


unflattened_sequence = [('prefix', 'instrument', 'piano'), "<S>"] + unflattened_sequence + ["<E>"]

# # Print the generated sequences
# print("Generated sequences:", unflattened_sequence)

# Write the generated sequences to a MIDI file
aria_tokenizer = AbsTokenizer()
mid_dict = aria_tokenizer.detokenize(unflattened_sequence)
mid = mid_dict.to_midi()

filename = os.path.basename(file_path)
mid.save(os.path.join(output_folder, f"arpeggiated_{filename}"))
# mid.save('test_file.mid')