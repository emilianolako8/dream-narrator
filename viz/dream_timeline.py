"""
dream_timeline.py
-----------------
PURPOSE: Visualize the decoded dream sequence as a timeline.

WHY THIS EXISTS:
Numbers and text are hard to interpret across time.
A visual timeline instantly shows:
- When REM periods occurred
- How decoded concepts changed across epochs
- Which themes dominated which parts of the dream

This is the standard visualization in sleep research papers.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os


# ─────────────────────────────────────────
# COLOR SCHEMES
# Colors chosen to match standard sleep
# research visualization conventions
# ─────────────────────────────────────────
SLEEP_STAGE_COLORS = {
    'Wake' : '#FF6B6B',   # red
    'REM'  : '#4ECDC4',   # teal
    'N1'   : '#95E1D3',   # light green
    'N2'   : '#3D5A80',   # blue
    'N3'   : '#1A1A2E',   # dark navy
}

CONCEPT_COLORS = {
    # Social
    'person'  : '#FF6B6B', 'face'    : '#FF8E53',
    'crowd'   : '#FFA07A', 'stranger': '#FF4500',
    'family'  : '#DC143C',
    # Places
    'house'   : '#4682B4', 'room'    : '#5F9EA0',
    'forest'  : '#228B22', 'water'   : '#00CED1',
    'city'    : '#708090', 'darkness': '#2F4F4F',
    # Actions
    'running' : '#FFD700', 'flying'  : '#FFA500',
    'falling' : '#FF69B4', 'chasing' : '#FF1493',
    'hiding'  : '#9400D3',
    # Emotions
    'fear'    : '#8B0000', 'joy'     : '#FFD700',
    'confusion': '#DDA0DD','peace'   : '#98FB98',
    'anxiety' : '#CD5C5C',
    # Objects
    'door'    : '#8B4513', 'light'   : '#FFFFE0',
    'animal'  : '#556B2F', 'vehicle' : '#708090',
    'stairs'  : '#BC8F8F',
    # Abstract
    'transformation': '#9370DB', 'memory'  : '#20B2AA',
    'unknown'       : '#696969', 'danger'  : '#B22222',
    'safety'        : '#3CB371',
}


def plot_concept_timeline(decoded_sequence, dominant_themes,
                           save_path=None):
    """
    Plot decoded concepts across dream epochs as a timeline.

    WHY THIS VISUALIZATION:
    Each epoch is a column. Each concept is a colored bar.
    The height of the bar = confidence score.
    This lets you see at a glance how the dream evolved.

    Parameters:
        decoded_sequence : list of epoch dicts with concepts
        dominant_themes  : list of (concept, data) tuples
        save_path        : path to save the figure
    """
    n_epochs = len(decoded_sequence)
    fig = plt.figure(figsize=(14, 12))
    fig.patch.set_facecolor('#0D1117')  # dark background

    gs = GridSpec(2, 1, figure=fig, hspace=0.8)

    # ── TOP PLOT: Concept scores per epoch ──
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#161B22')

    # Plot each concept as a colored bar
    bar_width = 0.25
    epochs    = [e['epoch'] for e in decoded_sequence]

    for epoch_data in decoded_sequence:
        epoch    = epoch_data['epoch']
        concepts = epoch_data['concepts']

        for i, (concept, score) in enumerate(concepts):
            color = CONCEPT_COLORS.get(concept, '#FFFFFF')
            x     = epoch + (i - 1) * bar_width
            ax1.bar(x, score, width=bar_width,
                   color=color, alpha=0.8, edgecolor='none')
            ax1.text(x, score + 0.01, concept[:4],
                    ha='center', va='bottom',
                    fontsize=6, color='white', rotation=45)

    ax1.set_xlabel('Dream Epoch (30s each)',
                   color='white', fontsize=10)
    ax1.set_ylabel('Concept Confidence',
                   color='white', fontsize=10)
    ax1.set_title('Decoded Dream Concepts Across Time',
                  color='white', fontsize=12, pad=10)
    ax1.set_xticks(epochs)
    ax1.set_xticklabels(
        [f'Epoch {e}' for e in epochs],
        rotation=45, color='white', fontsize=8
    )
    ax1.tick_params(colors='white')
    ax1.spines['bottom'].set_color('#30363D')
    ax1.spines['left'].set_color('#30363D')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ── BOTTOM PLOT: Dominant theme frequency ──
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('#161B22')

    top_themes  = dominant_themes[:8]
    theme_names = [t[0] for t in top_themes]
    theme_counts = [t[1]['count'] for t in top_themes]
    theme_colors = [CONCEPT_COLORS.get(t, '#4ECDC4')
                   for t in theme_names]

    bars = ax2.barh(theme_names, theme_counts,
                   color=theme_colors, alpha=0.8, edgecolor='none')

    # Add count labels on bars
    for bar, count in zip(bars, theme_counts):
        ax2.text(
            bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
            str(count), va='center', color='white', fontsize=9
        )

    ax2.set_xlabel('Appearances', color='white', fontsize=10)
    ax2.set_title('Dominant Dream Themes',
                  color='white', fontsize=12, pad=10)
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('#30363D')
    ax2.spines['left'].set_color('#30363D')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.suptitle('DREAM NARRATOR -- Neural Decoding Visualization',
                color='#4ECDC4', fontsize=14, fontweight='bold', y=1.02)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                   facecolor='#0D1117')
        print(f"Timeline saved to {save_path}")

    plt.show()
    return fig


def plot_brain_wave_sample(raw, save_path=None):
    """
    Plot a sample of the raw EEG brain waves.

    WHY:
    Showing the actual brain waves alongside the decoded concepts
    makes the project tangible. People can see:
    "these squiggly lines produced that dream story"

    This is the most visually striking proof of concept.
    """
    fig, axes = plt.subplots(
        len(raw.ch_names), 1,
        figsize=(14, 2 * len(raw.ch_names))
    )
    fig.patch.set_facecolor('#0D1117')

    if len(raw.ch_names) == 1:
        axes = [axes]

    # Plot first 30 seconds of each channel
    sfreq      = raw.info['sfreq']
    n_samples  = int(sfreq * 30)
    data       = raw.get_data()
    times      = raw.times[:n_samples]

    channel_colors = ['#4ECDC4', '#FF6B6B', '#FFD700',
                      '#95E1D3', '#FF8E53']

    for i, (ax, ch_name) in enumerate(zip(axes, raw.ch_names)):
        color      = channel_colors[i % len(channel_colors)]
        channel_data = data[i, :n_samples] * 1e6  # convert to microvolts

        ax.plot(times, channel_data, color=color,
               linewidth=0.5, alpha=0.9)
        ax.set_facecolor('#161B22')
        ax.set_ylabel(f'{ch_name}\n(μV)',
                     color='white', fontsize=8)
        ax.tick_params(colors='white', labelsize=7)
        ax.spines['bottom'].set_color('#30363D')
        ax.spines['left'].set_color('#30363D')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if i == len(raw.ch_names) - 1:
            ax.set_xlabel('Time (seconds)',
                         color='white', fontsize=10)

    plt.suptitle('Raw EEG Brain Waves (First 30 seconds)',
                color='#4ECDC4', fontsize=13,
                fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                   facecolor='#0D1117')
        print(f"Brain wave plot saved to {save_path}")

    plt.show()
    return fig


if __name__ == "__main__":

    import sys
    import os
    sys.path.insert(0, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    ))

    from pipeline.data_loader import load_sleep_recording
    from pipeline.preprocessor import bandpass_filter, remove_powerline_noise
    from pipeline.feature_extractor import extract_features_from_epoch
    from pipeline.neural_encoder import DreamEncoder
    from pipeline.semantic_decoder import SemanticDecoder
    from configs.config_loader import config
    import torch

    # ── Load data ──
    data_dir       = config['data']['raw_dir']
    psg_path       = os.path.join(data_dir, config['data']['psg_file'])
    hypnogram_path = os.path.join(data_dir, config['data']['hypnogram_file'])

    raw, annotations = load_sleep_recording(psg_path, hypnogram_path)
    raw = remove_powerline_noise(raw)
    raw = bandpass_filter(raw)

    sfreq = raw.info['sfreq']
    data  = raw.get_data()

    # ── Extract features ──
    features_list  = []
    epoch_duration = int(sfreq * 30)
    n_epochs       = min(10, data.shape[1] // epoch_duration)

    for i in range(n_epochs):
        start      = i * epoch_duration
        end        = start + epoch_duration
        epoch_data = data[:, start:end]
        features, _ = extract_features_from_epoch(epoch_data, sfreq)
        features_list.append(features)

    features_array = np.array(features_list)
    mean           = features_array.mean(axis=0)
    std            = features_array.std(axis=0) + 1e-10
    features_list  = [(f - mean) / std for f in features_list]

    # ── Load encoder ──
    model = DreamEncoder(
        input_dim=len(features_list[0]),
        embedding_dim=config['encoder']['embedding_dim']
    )
    model.load_state_dict(
        torch.load(config['encoder']['model_path'])
    )
    model.eval()

    # ── Generate embeddings ──
    embeddings = []
    with torch.no_grad():
        for features in features_list:
            x            = torch.FloatTensor(features).unsqueeze(0)
            embedding, _ = model(x)
            embeddings.append(embedding.numpy().flatten())

    # ── Decode concepts ──
    decoder          = SemanticDecoder(
        embedding_dim=config['encoder']['embedding_dim']
    )
    decoded_sequence = decoder.decode_sequence(embeddings, top_n=3)
    dominant_themes  = decoder.summarize_sequence(decoded_sequence)

    # ── Plot timeline ──
    plot_concept_timeline(
        decoded_sequence,
        dominant_themes,
        save_path="data/processed/dream_timeline.png"
    )

    # ── Plot brain waves ──
    plot_brain_wave_sample(
        raw,
        save_path="data/processed/brain_waves.png"
    )