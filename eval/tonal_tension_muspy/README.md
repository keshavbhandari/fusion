# Tonal tension and Muspy metrics
Tonal tension implementation is adopted from [midi-miner](https://github.com/ruiguo-bio/midi-miner.git). 


### Quick Start

Calculate `tonal tension` and `MusPy` metrics for a MIDI file:

```python
from eval.tonal_tension_muspy.metrics import compute_tonal_tension, compute_muspy_metrics

# Specify MIDI file and key
midi_file = 'eval/tonal_tension_muspy/example_files/generated_Abel Baer, Cliff Friend - June Night.mid'
key = "C major" # None if unknown

# Compute tonal tension and muspy metrics
tonal_tension = compute_tonal_tension(midi_file, key)
muspy_metrics = compute_muspy_metrics(midi_file, key)
```

#### `tonal_tension` Output

- **times**: Array of time points.
- **diameter** / **tensile** / **centroid_diff**: Arrays capturing tonal tension metrics.
- **info**: Dictionary with song details (`song_name`, `song_tag`, `key`, `n_bars`).

#### `muspy_metrics` Output

- **pitch_related**: Pitch metrics (e.g., `pitch_range`, `n_pitches`, `polyphony_rate`).
- **rhythm_related**: Rhythm metrics (e.g., `empty_beat_rate`, `groove_consistency`).
- **info**: Additional song details (`song_name`, `song_tag`, `key`).


## Reference

For tonal tension, please cite the following work: 

Guo R, Simpson I, Magnusson T, Kiefer C., Herremans D..  2020.  A variational autoencoder for music generation controlled by tonal tension. Joint Conference on AI Music Creativity (CSMC + MuMe). 

```
@inproceedings{guo2020variational,
  title={A variational autoencoder for music generation controlled by tonal tension},
  author={Guo, Rui and Simpson, Ivor and Magnusson, Thor and Kiefer, Chris and Herremans, Dorien},
  booktitle={Joint Conference on AI Music Creativity (CSMC + MuMe)},
  year={2020}
}
```
