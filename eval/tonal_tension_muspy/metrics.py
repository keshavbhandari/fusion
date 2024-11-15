from typing import Optional, Dict, Any
import os
import shutil
from os import path
import muspy
from miditoolkit import MidiFile
from chorder import Dechorder

from tonal_tension_muspy.midi_miner.tension_calculation import calculate_tonal_tension


def compute_tonal_tension(midi_file: str, key: Optional[str] = "C major", print_info: bool = False, win_sz: int = -1) -> Dict[str, Any]:
    """
    Computes tonal tension metrics from a given MIDI file.

    Parameters:
        midi_file (str): Path to the MIDI file.
        key (Optional[str]): The key signature of the song (default is "C major").
        print_info (bool): Flag to print information about the song (default is True).
        win_sz (int): The window size for calculating the tonal tension (default is -1, which uses the default window size).

    Returns:
        Dict[str, Any]: A dictionary containing the following computed metrics:
            - "times": A list or array of time points related to the tonal tension calculation.
            - "diameter": A list or array representing the diameter of the tonal tension over time.
            - "tensile": A list or array capturing the tensile values calculated for the MIDI file.
            - "centroid_diff": A list or array showing the differences in centroids across the song.
            - "info": A nested dictionary with additional song information:
                - "song_name": The cleaned-up name of the song derived from the MIDI file name.
                - "song_tag": The original file name without the `.mid` extension, useful for identifying file variations.
                - "key": The key signature of the song being analyzed.
                - "n_bars": The total number of bars present in the song, derived from the diameter metric.
    """

    if key is None:
        key = ''

    # Process song name and create a temporary directory for output files
    song_tag = path.basename(midi_file).replace('.mid', '')
    song_name = song_tag.replace('original_', '').replace('generated_', '').replace('Monophonic_', '')
    temp_output_dir = path.join(".temp", song_name)
    os.makedirs(temp_output_dir, exist_ok=True)

    # Run tension calculation command
    times, diameter, tensile, centroid_diff, files_result = calculate_tonal_tension(file_name=midi_file,
                                                                                    output_folder=temp_output_dir,
                                                                                    key_name=key,
                                                                                    key_changed=False,
                                                                                    win_sz=win_sz)

    # Load and print key change information from the JSON result file
    if print_info:
        # with open(path.join(temp_output_dir, 'files_result.json'), 'r') as fp:
        #     files_result = json.load(fp)
        for k in files_result.keys():
            print(f'song name: {k}')
            print(f'song key: {files_result[k][0]}')
            print(f'key change time: {files_result[k][1]}')
            print(f'key change bar: {files_result[k][2]}')
            print(f'key change name: {files_result[k][3]}')
        print(f'The file has {len(diameter)} bars.')

    # Clean up temporary directory
    # shutil.rmtree(temp_output_dir) # file occupied error?

    return {
        "times": times,
        "diameter": diameter,
        "tensile": tensile,
        "centroid_diff": centroid_diff,
        "info": {
            "song_name": song_name,
            "song_tag": song_tag,  # this distinguishes between original and generated from the file name 
            "key": key,
            "n_bars": len(diameter),
        }
    }


def compute_muspy_metrics(midi_file: str,
                          key: Optional[str] = "C major",
                          print_info: bool = False,
                          ignore_chord_inv: bool = True) -> Dict[str, Any]:
    """
    Computes musical metrics from a given MIDI file using MusPy.

    Parameters:
        midi_file (str): Path to the MIDI file.
        key (Optional[str]): The key signature of the song (default is "C major").
        print_info (bool): Flag to print information about the song (default is False).

    Returns:
        Dict[str, Any]: A dictionary containing two main categories of metrics, along with additional information:
            - "pitch_related": A dictionary with pitch-related metrics including:
                - "pitch_range": The range of pitches used in the MIDI file.
                - "n_pitches": The number of unique pitches used.
                - "n_pitch_classes": The number of unique pitch classes used.
                - "polyphony": A measure of polyphony, indicating the extent of simultaneous notes.
                - "polyphony_rate": The rate of polyphony across the song.
                - "pitch_in_scale_rate": The rate of pitches that fall within the specified key signature.
                - "n_unique_chords": The number of unique chords detected, considering root, quality, and bass pitch.
            - "rhythm_related": A dictionary with rhythm-related metrics including:
                - "empty_beat_rate": The proportion of beats without notes.
                - "groove_consistency": A metric for consistency of rhythm patterns, based on the resolution of measures.
            - "info": A nested dictionary with metadata about the song:
                - "song_name": The name of the song, cleaned up from any prefixes in the file name.
                - "song_tag": The original file name without the `.mid` extension, used for identifying different versions.
                - "key": The key signature used for the song analysis.
    """
    song_tag = path.basename(midi_file).replace('.mid', '')
    song_name = path.basename(midi_file).replace('.mid', '').replace('original_', \
                                '').replace('generated_',    '').replace('Monophonic_', '')

    # Load the original and generated MIDI files
    music = muspy.read(midi_file)

    # Calculate the number of unique chords using another chorder library
    # FYI, we use these chord types, counted by beats:
    #     'M': major_map,
    #     'm': minor_map,
    #     'o': diminished_map,
    #     '+': augmented_map,
    #     '7': dominant_map,
    #     'M7': major_seventh_map,
    #     'm7': minor_seventh_map,
    #     'o7': diminished_seventh_map,
    #     '/o7': half_diminished_seventh_map,
    #     'sus2': sus_2_map,
    #     'sus4': sus_4_map
    midi_obj = MidiFile(midi_file)
    dechorder = Dechorder()
    chords = dechorder.dechord(midi_obj)
    if ignore_chord_inv:
        # ignore chord inversions
        _unique_chords = set((chord.root_pc, chord.quality)
                             for chord in chords
                             if chord.root_pc is not None and chord.quality is not None and chord.bass_pc is not None)
        # for diminished chords ('o'), any roots having the same %3 value are considered the same
        unique_chords = {(root % 3, ctype) if ctype == 'o' else (root, ctype) for (root, ctype) in _unique_chords}
    else:
        unique_chords = set((chord.root_pc, chord.quality, chord.bass_pc)
                            for chord in chords
                            if chord.root_pc is not None and chord.quality is not None and chord.bass_pc is not None)
    n_unique_chords = len(unique_chords)

    return {
        "pitch_related": {
            "pitch_range": muspy.pitch_range(music),
            "n_pitches": muspy.n_pitches_used(music),
            "n_pitch_classes": muspy.n_pitch_classes_used(music),
            "polyphony": muspy.polyphony(music),
            "polyphony_rate": muspy.polyphony_rate(music),
            "pitch_in_scale_rate": muspy.pitch_in_scale_rate(music, root=0, mode='major'),
            "n_unique_chords": n_unique_chords,
        },
        "rhythm_related": {
            "empty_beat_rate": muspy.empty_beat_rate(music),
            "groove_consistency": muspy.groove_consistency(music, measure_resolution=480),
        },
        "info": {
            "song_name": song_name,
            "song_tag": song_tag,
            "key": key,
        }
    }


# def compute_tonal_tension_and_muspy_metrics(midi_file: str,
#                                             key: Optional[str] = "C major",
#                                             print_info: bool = False) -> Dict[str, Any]:
#     tonal_tension = compute_tonal_tension(midi_file, key, print_info)
#     muspy_metrics = compute_muspy_metrics(midi_file, key, print_info)

#     output_dict = {}
#     output_dict.update(muspy_metrics)
#     output_dict.update(tonal_tension)
#     return output_dict


def test():
    from eval.tonal_tension_muspy.metrics import compute_tonal_tension, compute_muspy_metrics
    midi_file = 'eval/tonal_tension_muspy/example_files/generated_Abel Baer, Cliff Friend - June Night.mid'
    key = "C major"

    tonal_tension = compute_tonal_tension(midi_file, key)
    muspy_metrics = compute_muspy_metrics(midi_file, key)
    """
    tonal_tension is a dictionary with the following structure:
    {
        "times": (np.array),
        "diameter": (np.array),
        "tensile": (np.array),
        "centroid_diff": (np.array),
        "info": {
            "song_name": (str),
            "song_tag": (str),  # '_original' or '_generated' guessed from the file name 
            "key": (str),
            "n_bars": (int),
        },
    }
    
    muspy_metrics is a dictionary with the following structure:
    {
        "pitch_related": {
            "pitch_range": (int),
            "n_pitches": (int),
            "n_pitch_classes": (int),
            "polyphony": (float),
            "polyphony_rate": (float),
            "pitch_in_scale_rate": (float),
            "n_unique_chords": (int),
        }, 
        "rhythm_related": {
            "empty_beat_rate": (float),
            "groove_consistency": (float),
        },
        "info": {
            "song_name": (str),
            "song_tag": (str),
            "key": (str),
        }
    }
    """
