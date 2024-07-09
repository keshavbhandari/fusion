import numpy as np
import ast
import random
import copy
from typing import List, Tuple, Union, Dict

class DataCorruption:
    def __init__(self):
        pass

    @staticmethod
    def read_data_from_file(file_path: str) -> List[Union[str, List[Union[str, int]]]]:
        """
        Read the data from a file and parse it into a list.
        """
        with open(file_path, 'r') as file:
            corruptions_string = file.read()
        data = ast.literal_eval(corruptions_string)
        return data

    def seperateitems(self, data: List[Union[str, List[Union[str, int]]]]) -> List[Union[str, List[Union[str, int]]]]:
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

    def get_segment_to_corrupt(self, data: List[Union[str, List[Union[str, int]]]], t_segment_ind = None) -> Tuple[List[int], List[Union[str, int]], bool]:
        """
        Get a random segment to corrupt from the data.
        """
        indices_to_corrupt = [n for n, i in enumerate(data) if type(i) == list]
        if t_segment_ind is not None:
            assert t_segment_ind < len(indices_to_corrupt), "t_segment_ind should be less than the number of segments in the data"
            random_index = indices_to_corrupt[t_segment_ind]            
        else:
            random_index = random.choice(indices_to_corrupt)
        last_idx_flag = False if random_index < len(data) - 1 else True

        return indices_to_corrupt, random_index, data[random_index], last_idx_flag

    def pitch_velocity_mask(self, data: List[Union[str, List[Union[str, int]]]], meta_data: List) -> Tuple[List[Union[str, List[Union[str, int]]]], str]:
        """
        Apply pitch and velocity mask to the data segment.
        """
        for n, note in enumerate(data):
            if type(data[n]) == list:
                data[n] = ['P', 'V', data[n][2], data[n][3]]

        return ['pitch_velocity_mask'] + meta_data + data, 'pitch_velocity_mask'

    def onset_duration_mask(self, data: List[Union[str, List[Union[str, int]]]], meta_data: List) -> Tuple[List[Union[str, List[Union[str, int]]]], str]:
        """
        Apply onset and duration mask to the data segment.
        """
        for n, note in enumerate(data):
            if type(data[n]) == list:
                data[n] = [data[n][0], data[n][1], 'O', 'D']

        return ['onset_duration_mask'] + meta_data + data, 'onset_duration_mask'

    def general_mask(self, data: List[Union[str, List[Union[str, int]]]], meta_data: List) -> Tuple[List[Union[str, List[Union[str, int]]]], str]:
        """
        Apply a general mask to the data segment.
        """
        str_elements = [i for i in data if type(i) == str]
        output = ['mask'] + str_elements

        return ['whole_mask'] + meta_data + output, 'whole_mask'

    def permute_pitches(self, data: List[Union[str, List[Union[str, int]]]], meta_data: List) -> Tuple[List[Union[str, List[Union[str, int]]]], str]:
        """
        Permute the pitches in the data segment.
        """
        pitches = [note[0] for note in data if type(note) == list]
        random.shuffle(pitches)

        for note in data:
            if type(note) == list:
                note[0] = pitches.pop(0)

        return ['pitch_permutation'] + meta_data + data, 'pitch_permutation'

    def permute_pitch_velocity(self, data: List[Union[str, List[Union[str, int]]]], meta_data: List) -> Tuple[List[Union[str, List[Union[str, int]]]], str]:
        """
        Permute the pitches and velocities in the data segment.
        """
        pitches = [note[0] for note in data if type(note) == list]
        random.shuffle(pitches)

        velocities = [note[1] for note in data if type(note) == list]
        random.shuffle(velocities)

        for note in data:
            if type(note) == list:
                note[0] = pitches.pop(0)
                note[1] = velocities.pop(0)

        return ['pitch_velocity_permutation'] + meta_data + data, 'pitch_velocity_permutation'

    def fragmentation(self, data: List[Union[str, List[Union[str, int]]]], meta_data: List) -> Tuple[List[Union[str, List[Union[str, int]]]], str]:
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
    
    def incorrect_transposition(self, data: List[Union[str, List[Union[str, int]]]], meta_data: List) -> Tuple[List[Union[str, List[Union[str, int]]]], str]:
        """
        Transpose the pitches in the data segment by a random value.
        """
        add_by = 5
        subtract_by = -5
        for n, note in enumerate(data):
            if type(data[n]) == list:
                if random.choice([True, False]) and (data[n][0] < 127-add_by or data[n][0] > 0+subtract_by):
                    data[n][0] += random.randint(-5, 5)
            else:
                data[n] = data[n]

        return ['incorrect_transposition'] + meta_data + data, 'incorrect_transposition'

    def shorten_list(self, lst, index, segment_indices, context_before, context_after):
        # Ensure the list is not empty and the index is within bounds
        if not lst or index < 0 or index >= len(lst):
            raise ValueError("List is empty or index is out of bounds")
        
        # Get index of the segment based on index
        segment_index = segment_indices.index(index)
        
        # Calculate start and end indices for slicing the list
        start_index = max(segment_index - context_before, 0)
        end_index = min(segment_index + context_after, len(segment_indices)-1)
        actual_start_index = segment_indices[start_index]
        if random.uniform(0, 1) < 0.5 and index != 0:
            actual_end_index = segment_indices[end_index] # no context after the corrupted segment
        else:
            actual_end_index = segment_indices[end_index] + 1 # context_after + 1 to include the next single item
        
        # Slice the list to get the desired elements
        shortened_list = lst[actual_start_index:actual_end_index]
        
        return shortened_list

    def apply_random_corruption(self, data: List[Union[str, List[Union[str, int]]]], context_before = 5, context_after = 1, meta_data = [], t_segment_ind=None) -> Dict[str, Union[List[Union[str, List[Union[str, int]]]], str, bool]]:
        """
        Apply a random corruption function to a segment of the data.
        """

        corruption_functions = [
            self.pitch_velocity_mask,
            self.onset_duration_mask,
            self.general_mask,
            self.permute_pitches,
            self.permute_pitch_velocity,
            self.fragmentation,
            self.incorrect_transposition
        ]
        corruption_function = random.choice(corruption_functions)

        corrupted_data = copy.deepcopy(data)
        corrupted_data = self.seperateitems(corrupted_data)
        all_segment_indices, index, segment, last_idx_flag = self.get_segment_to_corrupt(corrupted_data, t_segment_ind=t_segment_ind)
        segment_copy = copy.deepcopy(segment)

        corrupted_segment, corruption_type = corruption_function(segment_copy, meta_data)
        corrupted_segment = ['SEP'] + corrupted_segment + ['SEP']
        corrupted_data[index] = corrupted_segment

        # Modify corrupted_data to shorten context before and after the corrupted segment
        corrupted_data = self.shorten_list(corrupted_data, index, all_segment_indices, context_before=context_before, context_after=context_after)

        # Concatenate back the corrupted data
        corrupted_data_sequence = []
        for element in corrupted_data:
            if type(element) == list:
                corrupted_data_sequence += element
            else:
                corrupted_data_sequence.append(element)

        output = {
            'original_sequence': data,
            'corrupted_sequence': corrupted_data_sequence,
            'original_segment': segment,
            'corrupted_segment': corrupted_segment,
            'corruption_type': corruption_type,
            'last_idx_flag': last_idx_flag
        }

        return output


# Usage example
# Load the data from a file
data = DataCorruption.read_data_from_file('/homes/kb658/fusion/2_style_transfer/corruptions.txt')

# Initialize the DataCorruption class with the loaded data
data_corruption = DataCorruption()

# Apply a random corruption
output = data_corruption.apply_random_corruption(data)
output
