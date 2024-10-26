from typing import Optional
import os
from os import path
import pickle
import json
import numpy as np
from matplotlib import pyplot as plt
import muspy


def plot_tension_comparison(data1, data2, title: Optional[str] = None, save_img_path: Optional[str] = None):
    fig, axs = plt.subplots(3, 1, figsize=(20, 15), facecolor='white')
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['grid.color'] = '#DDDDDD'
    plt.rcParams['lines.color'] = '#6C757D'

    metrics = ['diameter', 'tensile', 'centroid_diff']
    titles = ['Diameter', 'Tensile', 'Centroid Diff']

    for i, metric in enumerate(metrics):
        axs[i].plot(data1['times'],
                    data1[metric],
                    marker='o',
                    label=data1['info']["song_name_gen_org"],
                    color='#FF6F61')
        axs[i].plot(data2['times'],
                    data2[metric],
                    marker='s',
                    label=data2['info']["song_name_gen_org"],
                    color='#42A5F5')
        axs[i].set_title(titles[i], color='black')
        axs[i].legend(fontsize=14, facecolor='white', edgecolor='black')
        axs[i].grid(True)

    if title:
        plt.suptitle(title, fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_img_path:
        plt.savefig(save_img_path, bbox_inches='tight')
    plt.show()


def compute_tonal_tension(midi_file: str, task: str, key: Optional[str] = "C major"):
    if key is None:
        key = "Unknown"

    song_name_gen_org = path.basename(midi_file).replace('.mid', '')
    song_name = path.basename(midi_file).replace('.mid', '').replace('original_',
                                                                     '').replace('generated_',
                                                                                 '').replace('Monophonic_', '')
    output_dir = path.join("example/output", task, song_name)

    time_file = path.join(output_dir, song_name_gen_org + '.time')
    diameter_file = path.join(output_dir, song_name_gen_org + '.diameter')
    tensile_file = path.join(output_dir, song_name_gen_org + '.tensile')
    centroid_diff_file = path.join(output_dir, song_name_gen_org + '.centroid_diff')

    # Processing the output files
    if key == "Unknown":
        command = f'python tension_calculation.py -f "{midi_file}" -o "{output_dir}" -k False -w 1'
    else:
        command = f'python tension_calculation.py -f "{midi_file}" -o "{output_dir}" -k False -w 1 -n "{key}"'
    os.system(command)

    # Load the output files
    times = pickle.load(open(time_file, 'rb'))
    diameter = pickle.load(open(diameter_file, 'rb'))
    tensile = pickle.load(open(tensile_file, 'rb'))
    centroid_diff = pickle.load(open(centroid_diff_file, 'rb'))

    # Print information
    with open(os.path.join(output_dir, 'files_result.json'), 'r') as fp:
        keys = json.load(fp)

    for k in keys.keys():
        print(f'song name is {k}')
        print(f'song key is {keys[k][0]}')
        print(f'song key change time {keys[k][1]}')
        print(f'song key change bar {keys[k][2]}')
        print(f'song key change name {keys[k][3]}')

    print(f'the file has {len(diameter)} bars')

    return {
        "times": times,
        "diameter": diameter,
        "tensile": tensile,
        "centroid_diff": centroid_diff,
        "info": {
            "output_dir": output_dir,
            "song_name": song_name,
            "song_name_gen_org": song_name_gen_org,
            "key": key,
            "bars": len(diameter),
        }
    }


def plot_muspy_comparison(data1, data2, title: Optional[str] = None, save_img_path: Optional[str] = None):
    # Set global font size for better consistency
    plt.rcParams.update({
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'legend.fontsize': 14,
        'figure.titlesize': 20,
        'axes.facecolor': 'white',
        'grid.color': '#DDDDDD',
        'lines.color': '#6C757D'
    })

    categories = [
        'pitch range', 'n pitches', 'n pitch classes', 'polyphony', 'polyphony rate', 'pitch in scale rate',
        'empty beat rate', 'groove consistency'
    ]

    values1 = [data1['pitch_related'][k] for k in data1['pitch_related']] + \
              [data1['rhythm_related'][k] for k in data1['rhythm_related']]
    values2 = [data2['pitch_related'][k] for k in data2['pitch_related']] + \
              [data2['rhythm_related'][k] for k in data2['rhythm_related']]
    x = np.arange(len(categories))

    plt.figure(figsize=(10, 6))
    width = 0.35  # the width of the bars

    # Plotting bars
    plt.bar(x - width / 2, values1, width, label=data1['info']["song_name_gen_org"].split('_')[0], color='#FF6F61')
    plt.bar(x + width / 2, values2, width, label=data2['info']["song_name_gen_org"].split('_')[0], color='#42A5F5')
    plt.ylabel('Metric Value')
    plt.xticks(x, categories, rotation=45, ha='right')
    plt.legend(fontsize=14)

    if title:
        plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    # Save the plot if save_img_path is provided
    if save_img_path:
        plt.savefig(save_img_path, bbox_inches='tight')
    plt.show()


data_all = {
    0: {
        'input_org_file':
            'example/data/Harmonization/Sample_1/target_classical/original_Abel Baer, Cliff Friend - June Night.mid',
        'input_gen_file':
            'example/data/Harmonization/Sample_1/target_classical/generated_Abel Baer, Cliff Friend - June Night.mid',
        'key':
            'C major',
        'task':
            'Harmonization(target_classical)'
    },
    1: {
        'input_org_file':
            'example/data/Harmonization/Sample_2/target_classical/original_Johnny Marks - Rudolph The Red-Nosed Reindeer.mid',
        'input_gen_file':
            'example/data/Harmonization/Sample_2/target_classical/generated_Monophonic_Johnny Marks - Rudolph The Red-Nosed Reindeer.mid',
        'key':
            'C major',
        'task':
            'Harmonization(target_classical)'
    },
    2: {
        'input_org_file':
            'example/data/Harmonization/Sample_2/target_jazz/original_Johnny Marks - Rudolph The Red-Nosed Reindeer.mid',
        'input_gen_file':
            'example/data/Harmonization/Sample_2/target_jazz/generated_Monophonic_Johnny Marks - Rudolph The Red-Nosed Reindeer.mid',
        'key':
            'C major',
        'task':
            'Harmonization(target_jazz)'
    },
    3: {
        'input_org_file': 'example/data/Reharmonization/Sample_1/target_jazz/original_Oh_Johnny,_oh_Johnny,_Oh!.mid',
        'input_gen_file': 'example/data/Reharmonization/Sample_1/target_jazz/generated_Oh_Johnny,_oh_Johnny,_Oh!.mid',
        'key': 'C major',
        'task': 'Reharmonization(target_jazz)'
    }
}


def compute_muspy_metrics(midi_file: str, task: str, key: Optional[str] = "C major"):
    song_name_gen_org = path.basename(midi_file).replace('.mid', '')
    song_name = path.basename(midi_file).replace('.mid', '').replace('original_', \
                                '').replace('generated_',    '').replace('Monophonic_', '')
    output_dir = path.join("example/output", task, song_name)

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
            "output_dir": output_dir,
            "song_name": song_name,
            "song_name_gen_org": song_name_gen_org,
            "key": key,
        }
    }


# Tonal Tension
for id, content in data_all.items():
    input_org_file = content['input_org_file']
    input_gen_file = content['input_gen_file']
    key = content['key']
    task = content['task']

    output_org_tension = compute_tonal_tension(input_org_file, task, key)
    output_gen_tension = compute_tonal_tension(input_gen_file, task, key)
    plot_tension_comparison(output_org_tension,
                            output_gen_tension,
                            title=f"Tonal Tension: {task} of  {output_org_tension['info']['song_name']}",
                            save_img_path=output_org_tension['info']['output_dir'] + '/tonal_tension.png')

# MusPy Metrics
for id, content in data_all.items():
    input_org_file = content['input_org_file']
    input_gen_file = content['input_gen_file']
    key = content['key']
    task = content['task']

    # Convert key to root and mode
    # root, mode = key2muspy_dict[key]
    output_org_muspy = compute_muspy_metrics(input_org_file, task, key)
    output_gen_muspy = compute_muspy_metrics(input_gen_file, task, key)

    plot_muspy_comparison(output_org_muspy,
                          output_gen_muspy,
                          title=f"MusPy Metrics: {task} of  {output_org_muspy['info']['song_name']}")
