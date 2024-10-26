from typing import Optional, Dict, Any
from os import path
import numpy as np
from matplotlib import pyplot as plt


def plot_tonal_tension_comparison(data1, data2, title: Optional[str] = None, save_img_path: Optional[str] = None):
    fig, axs = plt.subplots(3, 1, figsize=(20, 15), facecolor='white')
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['grid.color'] = '#DDDDDD'
    plt.rcParams['lines.color'] = '#6C757D'

    metrics = ['diameter', 'tensile', 'centroid_diff']
    titles = ['Diameter', 'Tensile', 'Centroid Diff']

    for i, metric in enumerate(metrics):
        axs[i].plot(data1['times'], data1[metric], marker='o', label=data1['info']["song_tag"], color='#FF6F61')
        axs[i].plot(data2['times'], data2[metric], marker='s', label=data2['info']["song_tag"], color='#42A5F5')
        axs[i].set_title(titles[i], color='black')
        axs[i].legend(fontsize=14, facecolor='white', edgecolor='black')
        axs[i].grid(True)

    if title:
        plt.suptitle(title, fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_img_path:
        plt.savefig(save_img_path, bbox_inches='tight')
    plt.show()


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
