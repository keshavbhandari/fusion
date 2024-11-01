import os
import json
import argparse
from midi_obj_eval.single_midi_eval import evaluate_single_midi, compare_single_midi_metrics
from midi_obj_eval.single_midi_eval import plot_pitch_class_histogram_pair, plot_pitch_class_transition_matrix_pair
from tonal_tension_muspy.metrics import compute_tonal_tension, compute_muspy_metrics
from tonal_tension_muspy.plot import plot_tonal_tension_comparison, plot_muspy_comparison

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args in multiple midi evaluation.')
    parser.add_argument(
        '-midi-path1', type=str,
        help='A directory of the midi files'
    )
    parser.add_argument(
        '-midi-path2', type=str,
        help='Another directory of the midi files'
    )
    parser.add_argument(
        '-out-dir', type=str, default="./results",
        help='The output directory to save metrics'
    )
    parser.add_argument(
        '--key', type=str, default="",
        help='A two-word string like "C major" (MUST DIVIDE BY SPACE) indicating the key of input midi files; default to empty string'
    )
    parser.add_argument(
        '--task', type=str, default="",
        help='A string like "Harmonization(target_classical)" indicating the task type; default to empty string'
    )

    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    _, midi_name1 = os.path.split(args.midi_path1)
    midi_name1 = os.path.splitext(midi_name1)[0]
    metrics1 = evaluate_single_midi(args.midi_path1, return_numpy=False)

    _, midi_name2 = os.path.split(args.midi_path2)
    midi_name2 = os.path.splitext(midi_name2)[0]
    metrics2 = evaluate_single_midi(args.midi_path2, return_numpy=False)

    # misc objective metrics
    metric_pairs = compare_single_midi_metrics(metrics1, metrics2)

    comparison_prefix = midi_name1 + '_vs_' + midi_name2
    out_json_filename = comparison_prefix + '_metrics.json'
    out_pctm_filename = comparison_prefix + '_pctm.pdf'
    out_pitch_hist_filename = comparison_prefix + '_pitch_hist.pdf'

    out_json_filepath = os.path.join(args.out_dir, out_json_filename)
    out_pctm_filepath = os.path.join(args.out_dir, out_pctm_filename)
    out_pitch_hist_filepath = os.path.join(args.out_dir, out_pitch_hist_filename)

    with open(out_json_filepath, "w") as outfile:
        json.dump(metric_pairs, outfile)

    plot_pitch_class_transition_matrix_pair(
        metric_pairs["pitch_class_transition_matrix"],
        out_pctm_filepath,
        names=(midi_name1, midi_name2)
    )
    plot_pitch_class_histogram_pair(
        metric_pairs["pitch_class_histogram"],
        out_pitch_hist_filepath,
        names=(midi_name1, midi_name2)
    )

    # tonal tension metrics
    output_org_tension = compute_tonal_tension(args.midi_path1, args.key)
    output_gen_tension = compute_tonal_tension(args.midi_path2, args.key)
    out_tonal_tension_comparison_filename = comparison_prefix + '_tonal_tension.pdf'
    out_tonal_tension_comparison_filepath = os.path.join(args.out_dir, out_tonal_tension_comparison_filename)
    # Convert key to root and mode
    # root, mode = key2muspy_dict[key]
    output_org_muspy = compute_muspy_metrics(args.midi_path1, args.key)
    output_gen_muspy = compute_muspy_metrics(args.midi_path2, args.key)
    print(output_org_muspy)
    print(output_gen_muspy)
    out_muspy_comparison_filename = comparison_prefix + '_muspy_metrics.pdf'
    out_muspy_comparison_filepath = os.path.join(args.out_dir, out_muspy_comparison_filename)
    plot_tonal_tension_comparison(
        output_org_tension,
        output_gen_tension,
        title=f"Tonal Tension: {args.task} of  {output_org_tension['info']['song_name']}",
        save_img_path=out_tonal_tension_comparison_filepath,
    )
    plot_muspy_comparison(
        output_org_muspy,
        output_gen_muspy,
        title=f"MusPy Metrics: {args.task} of  {output_org_muspy['info']['song_name']}",
        save_img_path=out_muspy_comparison_filepath,
    )
    print("Saved objective metrics to {}".format(args.out_dir))