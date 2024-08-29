import numpy as np
import ast
import random
import copy
from typing import List, Tuple, Union, Dict

class DataCorruption:
    def __init__(self):

        self.corruption_functions = {
            'pitch_velocity_mask': self.pitch_velocity_mask,
            'onset_duration_mask': self.onset_duration_mask,
            'whole_mask': self.whole_mask,
            'permute_pitches': self.permute_pitches,
            'permute_pitch_velocity': self.permute_pitch_velocity,
            'fragmentation': self.fragmentation,
            'incorrect_transposition': self.incorrect_transposition,
            'skyline': self.skyline,
            'note_modification': self.note_modification
        }

    @staticmethod
    def read_data_from_file(file_path: str) -> List[Union[str, List[Union[str, int]]]]:
        """
        Read the data from a file and parse it into a list.
        """
        with open(file_path, 'r') as file:
            corruptions_string = file.read()
        data = ast.literal_eval(corruptions_string)
        return data

    def seperateitems(self, data: List) -> List[Union[str, List[Union[str, int]]]]:
        """
        Split the list so all items between a set of <T> tokens are individual.
        """
        items = []
        current_item = []

        for element in data:
            if element in ['<T>']:
                if current_item:
                    items.append(current_item)
                    current_item = []
                items.append(element)
            else:
                current_item.append(element)

        if current_item:
            items.append(current_item)

        return items

    def get_segment_to_corrupt(self, data: List, t_segment_ind = None, exclude_idx = []) -> Tuple[List[int], List[Union[str, int]], bool]:
        """
        Get a random segment to corrupt from the data.
        """
        indices_to_corrupt = [n for n, i in enumerate(data) if type(i) == list and n not in exclude_idx]
        if t_segment_ind is not None:
            assert t_segment_ind < len(indices_to_corrupt), "t_segment_ind should be less than the number of segments in the data"
            random_index = indices_to_corrupt[t_segment_ind]            
        else:
            random_index = random.choice(indices_to_corrupt)
        last_idx_flag = False if random_index < len(data) - 1 else True

        return indices_to_corrupt, random_index, data[random_index], last_idx_flag

    def pitch_velocity_mask(self, data: List, meta_data: List, **kwargs) -> Tuple[List[Union[str, List[Union[str, int]]]], str]:
        """
        Apply pitch and velocity mask to the data segment.
        """
        for n, note in enumerate(data):
            if type(data[n]) == list:
                data[n] = ['P', 'V', data[n][2], data[n][3]]

        return ['pitch_velocity_mask'] + meta_data + data, 'pitch_velocity_mask'

    def onset_duration_mask(self, data: List, meta_data: List, **kwargs) -> Tuple[List[Union[str, List[Union[str, int]]]], str]:
        """
        Apply onset and duration mask to the data segment.
        """
        for n, note in enumerate(data):
            if type(data[n]) == list:
                data[n] = [data[n][0], data[n][1], 'O', 'D']

        return ['onset_duration_mask'] + meta_data + data, 'onset_duration_mask'

    def whole_mask(self, data: List, meta_data: List, **kwargs) -> Tuple[List[Union[str, List[Union[str, int]]]], str]:
        """
        Apply a general mask to the data segment.
        """
        str_elements = [i for i in data if type(i) == str]
        output = ['mask'] + str_elements

        return ['whole_mask'] + meta_data + output, 'whole_mask'

    def permute_pitches(self, data: List, meta_data: List, **kwargs) -> Tuple[List[Union[str, List[Union[str, int]]]], str]:
        """
        Permute the pitches in the data segment.
        """
        if not kwargs.get('inference'):
            pitches = [note[0] for note in data if type(note) == list]
            random.shuffle(pitches)

            for note in data:
                if type(note) == list:
                    note[0] = pitches.pop(0)

        return ['pitch_permutation'] + meta_data + data, 'pitch_permutation'

    def permute_pitch_velocity(self, data: List, meta_data: List, **kwargs) -> Tuple[List[Union[str, List[Union[str, int]]]], str]:
        """
        Permute the pitches and velocities in the data segment.
        """
        if not kwargs.get('inference'):
            pitches = [note[0] for note in data if type(note) == list]
            random.shuffle(pitches)

            velocities = [note[1] for note in data if type(note) == list]
            random.shuffle(velocities)

            for note in data:
                if type(note) == list:
                    note[0] = pitches.pop(0)
                    note[1] = velocities.pop(0)

        return ['pitch_velocity_permutation'] + meta_data + data, 'pitch_velocity_permutation'

    def fragmentation(self, data: List, meta_data: List, **kwargs) -> Tuple[List[Union[str, List[Union[str, int]]]], str]:
        """
        Fragment the data segment.
        """
        len_segment = len(data)
        # Choose a random percentage between 0.2-0.5 to fragment the data
        fragment_percentage = random.uniform(0.2, 0.5)
        fragment_length = int(len_segment * fragment_percentage)

        fragmented_data = []
        for n, note in enumerate(data):
            if type(note) == list:
                if n < fragment_length:
                    fragmented_data.append(note)
            else:
                fragmented_data.append(note)

        return ['fragmentation'] + meta_data + fragmented_data, 'fragmentation'
    
    def incorrect_transposition(self, data: List, meta_data: List, **kwargs) -> Tuple[List[Union[str, List[Union[str, int]]]], str]:
        """
        Transpose the pitches in the data segment by a random value.
        """
        if not kwargs.get('inference'):
            add_by = 5
            subtract_by = -5
            for n, note in enumerate(data):
                if type(data[n]) == list:
                    if random.choice([True, False]) and (data[n][0] < 127-add_by or data[n][0] > 0+subtract_by):
                        data[n][0] += random.randint(-5, 5)
                else:
                    data[n] = data[n]

        return ['incorrect_transposition'] + meta_data + data, 'incorrect_transposition'

    # Define a function to round a value to the nearest 10
    @staticmethod
    def round_to_nearest_n(input_value, round_to=10):
        rounded_value = round(input_value / round_to) * round_to
        return rounded_value

    def note_modification(self, data: List, meta_data: List, **kwargs) -> Tuple[List[Union[str, List[Union[str, int]]]], str]:
        """
        Modify the notes in the data segment by either omitting them or adding in new notes.
        """
        data_copy = copy.deepcopy(data)
        omitted_data = [i for i in data_copy if type(i) == str]
        data_copy = [i for i in data_copy if type(i) == list]
        skip_idx = []
        for n, note in enumerate(data_copy):
            if type(data_copy[n]) == list and n < (len(data_copy)-1) and n not in skip_idx:
                # Note omission with dynamic probability
                prob = random.uniform(0.1, 0.4)
                if random.uniform(0, 1) < prob:
                    skip_idx.append(n+1)
                    next_note_onset = data_copy[n+1][2]
                    curr_next_note_onset_diff = abs(data_copy[n][2] - next_note_onset)
                    if curr_next_note_onset_diff <= 50:
                        # Keep same onset and duration
                        omitted_data.append(data_copy[n])
                    else:
                        # Increase the duration of the current note by the next note's duration
                        new_duration = min(data_copy[n][3] + data_copy[n+1][3], 5000)
                        data_copy[n][3] = new_duration
                        # Add the current note to the omitted data
                        omitted_data.append(data_copy[n])
                else:
                    omitted_data.append(data_copy[n])

        added_data = []
        for n, note in enumerate(omitted_data):
            if type(omitted_data[n]) == list:
                # Note addition with dynamic probability
                prob = random.uniform(0.1, 0.4)
                if random.uniform(0, 1) < prob and omitted_data[n][3] > 500 and n < (len(omitted_data)-1):
                    # Add current note
                    tmp = copy.deepcopy(omitted_data[n])
                    # Modify the duration of the current note
                    tmp[3] = int(tmp[3] / 2)
                    tmp[3] = self.round_to_nearest_n(tmp[3], 10)
                    added_data.append(tmp)
                    # Add the new note
                    # New pitch between -5 and 5 semitones from the current pitch
                    new_pitch = tmp[0] + random.randint(-5, 5)
                    # New velocity between 45 and 105
                    new_velocity = random.choice([45, 60, 75, 90, 105])
                    diff_curr_next_onset = abs(omitted_data[n+1][2] - tmp[2])
                    # New onset should be between the current onset and the next onset
                    new_onset = tmp[2] + random.randint(0, diff_curr_next_onset)
                    # Round the new onset to the nearest 10
                    new_onset = self.round_to_nearest_n(new_onset)
                    # New duration should be the tmp[3] + or - 10% of the tmp[3]
                    new_duration = min((tmp[3] + random.randint(-int(tmp[3] * 0.1), int(tmp[3] * 0.1))), 5000)
                    # Round the new duration to the nearest 10
                    new_duration = self.round_to_nearest_n(new_duration, 10)
                    added_data.append([new_pitch, new_velocity, new_onset, new_duration])
                else:
                    added_data.append(omitted_data[n])
            else:
                added_data.append(omitted_data[n])
                

        return ['note_modification'] + meta_data + added_data, 'note_modification'

    # Skyline function for separating melody and harmony from the tokenized sequence
    def skyline(self, sequence: list, meta_data=[], diff_threshold=50, static_velocity=True, pitch_threshold=None, **kwargs):
        
        if pitch_threshold is None:
            pitch_threshold = 0
        
        melody = []

        if len(sequence) < 2:
            return ['skyline'] + meta_data + melody, 'skyline'
        
        if type(sequence[0]) == str:
            melody.append(sequence[0])
            sequence = sequence[1:]

        pointer_pitch = sequence[0][0]
        pointer_velocity = sequence[0][1]
        pointer_onset = sequence[0][2]
        pointer_duration = sequence[0][3]

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
                        # Update the pointer
                        pointer_pitch = current_pitch
                        pointer_velocity = current_velocity
                        pointer_onset = current_onset
                        pointer_duration = current_duration
                    else:
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
                                    melody.append([current_pitch, 90, current_onset, current_duration])
                                else:
                                    melody.append([current_pitch, current_velocity, current_onset, current_duration])

            if sequence[i-1] == "<T>":
                melody.append("<T>")
            
            if sequence[i] == "<D>":
                melody.append("<D>")

        return ['skyline'] + meta_data + melody, 'skyline'

    def shorten_list(self, lst, index, segment_indices, context_before, context_after, inference=False):
        # Ensure the list is not empty and the index is within bounds
        if not lst or index < 0 or index >= len(lst):
            raise ValueError("List is empty or index is out of bounds")
        
        # Get index of the segment based on index
        segment_index = segment_indices.index(index)
        
        # Calculate start and end indices for slicing the list
        start_index = max(segment_index - context_before, 0)
        end_index = min(segment_index + context_after, len(segment_indices)-1)
        actual_start_index = segment_indices[start_index]
        if random.uniform(0, 1) < 0.1 and index != 0 and not inference:
            actual_end_index = segment_indices[end_index] # no context after the corrupted segment
        else:
            actual_end_index = segment_indices[end_index] + 1 # context_after + 1 to include the next single item
        
        # Slice the list to get the desired elements
        shortened_list = lst[actual_start_index:actual_end_index]
        
        return shortened_list
    
    def concatenate_list(self, lst):
        """
        Concatenate a list of lists into a single list.
        """

        concatenated_list = []
        for element in lst:
            if type(element) == list:
                concatenated_list += element
            else:
                concatenated_list.append(element)
        return concatenated_list

    def apply_random_corruption(self, data: List, 
                                context_before: int = 5, context_after: int = 1, 
                                meta_data: List = [], t_segment_ind: int = None,
                                inference: bool = False, corruption_type: str = None,
                                run_corruption: bool = True) -> Dict:
        """
        Apply a random corruption function to a segment of the data.
        """

        if corruption_type is not None and corruption_type != 'random':
            corruption_function = self.corruption_functions[corruption_type]
        else:
            corruption_function = random.choice(list(self.corruption_functions.values()))

        data_copy = copy.deepcopy(data)
        separated_sequence = self.seperateitems(data_copy)
        corruption_data = copy.deepcopy(separated_sequence)
        all_segment_indices, index, segment, last_idx_flag = self.get_segment_to_corrupt(corruption_data, t_segment_ind=t_segment_ind, exclude_idx=[])
        segment_copy = copy.deepcopy(segment)

        if run_corruption:
            corrupted_segment, corruption_type = corruption_function(segment_copy, meta_data=meta_data, inference=inference)
            corrupted_segment = ['SEP'] + corrupted_segment + ['SEP']
            corruption_data[index] = corrupted_segment
        else:
            corrupted_segment = segment_copy
            corruption_data[index] = corrupted_segment
            corruption_type = None

        # Modify corrupted_data to shorten context before and after the corrupted segment
        shortened_corrupted_data = self.shorten_list(corruption_data, index, all_segment_indices, context_before=context_before, context_after=context_after, inference=inference)

        corrupted_data_sequence = self.concatenate_list(shortened_corrupted_data)

        output = {
            'original_sequence': data,
            'separated_sequence': separated_sequence,
            't_segment_ind': t_segment_ind,
            'index': index,
            'all_segment_indices': all_segment_indices,
            'corrupted_sequence': corrupted_data_sequence,
            'original_segment': segment,
            'corrupted_segment': corrupted_segment,
            'corruption_type': corruption_type,
            'last_idx_flag': last_idx_flag
        }

        return output


if __name__ == '__main__':
    # Load the data from a file
    data = DataCorruption.read_data_from_file('/homes/kb658/fusion/2_style_transfer/corruptions.txt')

    # Initialize the DataCorruption class with the loaded data
    data_corruption = DataCorruption()

    # Apply a random corruption
    output = data_corruption.apply_random_corruption(data, context_before=5, context_after=1, meta_data=["jazz"], t_segment_ind=None, inference=False, corruption_type=None, run_corruption=True)
    print(output['original_segment'])
    print(output['corrupted_segment'])
