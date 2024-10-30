import os
import json
import argparse
import numpy as np
from midi_obj_eval.core import PITCH_CLASSES
from midi_obj_eval.core import extract_pretty_midi_features
from midi_obj_eval.core import get_num_notes, get_used_pitch, get_pitch_class_histogram
from midi_obj_eval.core import get_pitch_class_transition_matrix
from midi_obj_eval.core import get_avg_ioi
import matplotlib.pyplot as plt

def evaluate_single_midi(midi_filepath, return_numpy = False):
    pretty_midi_features = extract_pretty_midi_features(midi_filepath)
    num_notes = get_num_notes(pretty_midi_features)
    used_pitch = get_used_pitch(pretty_midi_features)
    pitch_class_histogram = get_pitch_class_histogram(pretty_midi_features)
    pitch_class_transition_matrix = get_pitch_class_transition_matrix(pretty_midi_features, normalize=2)
    avg_ioi = get_avg_ioi(pretty_midi_features)
    metrics = {
        'num_notes': num_notes,
        'used_pitch': used_pitch,
        'pitch_class_histogram': pitch_class_histogram,
        'pitch_class_transition_matrix': pitch_class_transition_matrix,
        'avg_ioi': avg_ioi,
    }
    if return_numpy:
        return metrics
    for key in metrics.keys():
        if isinstance(metrics[key], (np.ndarray, np.generic)):
            metrics[key] = metrics[key].tolist()
    return metrics

def plot_pitch_class_histogram(pitch_class_histogram, save_path):
    fig, ax = plt.subplots(1)
    ax.bar(PITCH_CLASSES, height=pitch_class_histogram)
    fig.savefig(save_path)
    plt.close(fig)

def plot_pitch_class_transition_matrix(pitch_class_transition_matrix, save_path):
    fig, ax = plt.subplots(1)
    ax.set_xticks(np.arange(len(PITCH_CLASSES)), labels=PITCH_CLASSES)
    ax.set_yticks(np.arange(len(PITCH_CLASSES)), labels=PITCH_CLASSES)
    ax.imshow(pitch_class_transition_matrix)
    fig.savefig(save_path)
    plt.close(fig)

def kl_div_discrete(dist1: np.ndarray, dist2: np.ndarray, epsilon=1e-5):
    dist1 = dist1 + epsilon
    dist2 = dist2 + epsilon
    return np.sum(dist1 * np.log(dist1 / dist2))

def compare_single_midi_metrics(metrics1, metrics2):
    metric_pairs = {}
    for key in metrics1.keys():
        assert key in metrics2, f"{key} should also be in the other metric dict"
        metric_pairs[key] = (metrics1[key], metrics2[key])
        if key == 'pitch_class_histogram':
            metric_pairs["pitch_class_kl"] = kl_div_discrete(
                np.array(metrics1[key]),
                np.array(metrics2[key]),
            )
    return metric_pairs

def plot_pitch_class_histogram_pair(pitch_class_histogram_pair, save_path, names = (None, None)):
    name1 = "series1" if names[0] is None else names[0]
    name2 = "series2" if names[1] is None else names[1]
    fig, ax = plt.subplots(1)
    x_axis = np.arange(len(PITCH_CLASSES))
    bias, width = 0.2, 0.4
    ax.bar(x_axis - bias, height=pitch_class_histogram_pair[0], width=width, label=name1)
    ax.bar(x_axis + bias, height=pitch_class_histogram_pair[1], width=width, label=name2)
    ax.set_xticks(x_axis, PITCH_CLASSES)
    ax.set_xlabel("Pitch Classes")
    ax.set_ylabel("Frequency")
    ax.set_title("Pitch Classes Histogram")
    ax.legend()
    fig.savefig(save_path)
    plt.close(fig)

def plot_pitch_class_transition_matrix_pair(pitch_class_transition_matrix_pair, save_path, names = (None, None)):
    name1 = "series1" if names[0] is None else names[0]
    name2 = "series2" if names[1] is None else names[1]
    fig, axs = plt.subplots(1, 2)
    axs[0].set_xticks(np.arange(len(PITCH_CLASSES)), labels=PITCH_CLASSES)
    axs[0].set_yticks(np.arange(len(PITCH_CLASSES)), labels=PITCH_CLASSES)
    axs[0].imshow(pitch_class_transition_matrix_pair[0])
    axs[0].set_title(name1)
    axs[1].set_xticks(np.arange(len(PITCH_CLASSES)), labels=PITCH_CLASSES)
    axs[1].set_yticks(np.arange(len(PITCH_CLASSES)), labels=PITCH_CLASSES)
    axs[1].imshow(pitch_class_transition_matrix_pair[1])
    axs[1].set_title(name2)
    fig.savefig(save_path)
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args in single midi evaluation.')
    parser.add_argument(
        '-midi-path', type=str,
        help='The midi file to evaluate'
    )
    parser.add_argument(
        '-out-dir', type=str, default="./results",
        help='The output directory to save metrics'
    )
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    _, midi_name = os.path.split(args.midi_path)
    midi_name = os.path.splitext(midi_name)[0]

    metrics = evaluate_single_midi(args.midi_path, return_numpy=False)

    out_json_filename = midi_name + '_metrics.json'
    out_pctm_filename = midi_name + '_pctm.pdf'
    out_pitch_hist_filename = midi_name + '_pitch_hist.pdf'

    out_json_filepath = os.path.join(args.out_dir, out_json_filename)
    out_pctm_filepath = os.path.join(args.out_dir, out_pctm_filename)
    out_pitch_hist_filepath = os.path.join(args.out_dir, out_pitch_hist_filename)

    with open(out_json_filepath, "w") as outfile:
        json.dump(metrics, outfile)

    plot_pitch_class_transition_matrix(
        metrics["pitch_class_transition_matrix"],
        out_pctm_filepath
    )
    plot_pitch_class_histogram(
        metrics["pitch_class_histogram"],
        out_pitch_hist_filepath
    )
    print("Saved metrics to {}".format(args.out_dir))