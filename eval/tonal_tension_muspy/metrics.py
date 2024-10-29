from typing import Optional, Dict, Any
import os
import shutil
from os import path
import muspy

from tonal_tension_muspy.midi_miner.tension_calculation import calculate_tonal_tension


def compute_tonal_tension(midi_file: str, key: Optional[str] = "C major", print_info: bool = False) -> Dict[str, Any]:
    """
    Computes tonal tension metrics from a given MIDI file.

    Parameters:
        midi_file (str): Path to the MIDI file.
        key (Optional[str]): The key signature of the song (default is "C major").
        print_info (bool): Flag to print information about the song (default is True).

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
    times, diameter, tensile, centroid_diff, files_result = calculate_tonal_tension(
        file_name=midi_file,
        output_folder=temp_output_dir,
        key_name=key,
        key_changed=False
    )

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


def compute_muspy_metrics(midi_file: str, key: Optional[str] = "C major", print_info: bool = False) -> Dict[str, Any]:
    song_tag = path.basename(midi_file).replace('.mid', '')
    song_name = path.basename(midi_file).replace('.mid', '').replace('original_', \
                                '').replace('generated_',    '').replace('Monophonic_', '')

    # Load the original and generated MIDI files
    music = muspy.read(midi_file)

    return {
        "pitch_related": {
            "pitch_range": muspy.pitch_range(music),
            "n_pitches": muspy.n_pitches_used(music),
            "n_pitch_classes": muspy.n_pitch_classes_used(music),
            "polyphony": muspy.polyphony(music),
            "polyphony_rate": muspy.polyphony_rate(music),
            "pitch_in_scale_rate": muspy.pitch_in_scale_rate(music, root=0, mode='major'),
        },  # need to re-check root!
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
