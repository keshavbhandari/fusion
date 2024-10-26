from eval.tonal_tension_muspy.metrics import compute_tonal_tension, compute_muspy_metrics
from eval.tonal_tension_muspy.plot import plot_tonal_tension_comparison, plot_muspy_comparison

file_list = {
    0: {
        'input_org_file': 'eval/tonal_tension_muspy/example_files/original_Abel Baer, Cliff Friend - June Night.mid',
        'input_gen_file': 'eval/tonal_tension_muspy/example_files/generated_Abel Baer, Cliff Friend - June Night.mid',
        'key': 'C major',
        'task': 'Harmonization(target_classical)'
    },
}

# Tonal Tension
for id, content in file_list.items():
    input_org_file = content['input_org_file']
    input_gen_file = content['input_gen_file']
    key = content['key']
    task = content['task']

    output_org_tension = compute_tonal_tension(input_org_file, key)
    output_gen_tension = compute_tonal_tension(input_gen_file, key)
    plot_tonal_tension_comparison(output_org_tension,
                                  output_gen_tension,
                                  title=f"Tonal Tension: {task} of  {output_org_tension['info']['song_name']}",
                                  save_img_path=None)

# MusPy Metrics
for id, content in file_list.items():
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
