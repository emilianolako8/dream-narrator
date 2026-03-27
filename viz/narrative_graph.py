"""
narrative_graph.py
------------------
PURPOSE: Build a knowledge graph of dream concept relationships.

WHY THIS EXISTS:
The timeline shows concepts over time.
The graph shows how concepts RELATE to each other.
Concepts that appear together in the same epoch are connected.
The more they co-occur, the stronger the connection.

This reveals the hidden associative structure of the dream --
which concepts cluster together, which are central hubs,
which are isolated peripheral elements.

IN NEUROSCIENCE TERMS:
This maps the associative memory network active during REM.
When the hippocampus replays memories, it activates related
concepts together. The graph shows those associations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viz.dream_timeline import CONCEPT_COLORS


class NarrativeGraph:
    """
    Builds and visualizes a knowledge graph of dream concepts.

    WHY A CLASS:
    The graph has state -- nodes, edges, weights.
    A class keeps all that organized and makes it easy
    to build the graph once and visualize it multiple ways.
    """

    def __init__(self):
        self.nodes = {}   # concept -> total score
        self.edges = defaultdict(float)  # (concept1, concept2) -> weight

    def build_from_sequence(self, decoded_sequence):
        """
        Build graph from decoded concept sequence.

        HOW IT WORKS:
        For each epoch, every pair of concepts that appear together
        gets an edge. The edge weight increases each time they
        co-occur. This is called co-occurrence analysis --
        a fundamental technique in NLP and network science.

        Parameters:
            decoded_sequence : list of epoch dicts with concepts
        """
        for epoch_data in decoded_sequence:
            concepts = epoch_data['concepts']

            # Add nodes
            for concept, score in concepts:
                if concept not in self.nodes:
                    self.nodes[concept] = 0
                self.nodes[concept] += score

            # Add edges between all pairs in this epoch
            # This is called a "fully connected subgraph" per epoch
            for i in range(len(concepts)):
                for j in range(i + 1, len(concepts)):
                    c1 = concepts[i][0]
                    c2 = concepts[j][0]
                    # Sort to ensure consistent edge key
                    edge_key = tuple(sorted([c1, c2]))
                    self.edges[edge_key] += 1.0

        print(f"Graph built: {len(self.nodes)} nodes, "
              f"{len(self.edges)} edges")

    def get_node_positions(self):
        """
        Calculate positions for nodes using force-directed layout.

        WHY FORCE DIRECTED:
        Nodes with strong connections are pulled together.
        Nodes with no connections are pushed apart.
        This naturally clusters related concepts visually.

        We implement a simple version of the Fruchterman-Reingold
        algorithm -- the most common graph layout algorithm.
        Used by every graph visualization tool in the world.
        """
        np.random.seed(42)
        n_nodes  = len(self.nodes)
        concepts = list(self.nodes.keys())

        # Start with random positions in a circle
        angles    = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        positions = {
            concept: np.array([np.cos(a), np.sin(a)])
            for concept, a in zip(concepts, angles)
        }

        # Run force directed iterations
        # Attraction: connected nodes pull together
        # Repulsion: all nodes push apart
        learning_rate = 0.1

        for iteration in range(50):
            forces = {c: np.zeros(2) for c in concepts}

            # Repulsion between all pairs
            for i, c1 in enumerate(concepts):
                for j, c2 in enumerate(concepts):
                    if i >= j:
                        continue
                    diff = positions[c1] - positions[c2]
                    dist = max(np.linalg.norm(diff), 0.01)
                    # Repulsion force: inversely proportional to distance
                    repulsion = diff / (dist ** 2) * 0.1
                    forces[c1] += repulsion
                    forces[c2] -= repulsion

            # Attraction along edges
            for (c1, c2), weight in self.edges.items():
                if c1 not in positions or c2 not in positions:
                    continue
                diff       = positions[c2] - positions[c1]
                dist       = max(np.linalg.norm(diff), 0.01)
                # Attraction force: proportional to weight and distance
                attraction = diff * dist * weight * 0.01
                forces[c1] += attraction
                forces[c2] -= attraction

            # Update positions
            for concept in concepts:
                positions[concept] += forces[concept] * learning_rate

            # Decay learning rate
            learning_rate *= 0.95

        return positions

    def visualize(self, save_path=None):
        """
        Visualize the dream concept graph interactively using Plotly.
        Opens in browser. You can zoom, pan, and hover over nodes.
        """
        import plotly.graph_objects as go

        if not self.nodes:
            print("No nodes to visualize. Build graph first.")
            return

        positions = self.get_node_positions()

        # ── Build edge traces ──
        edge_traces = []
        max_weight  = max(self.edges.values()) if self.edges else 1

        for (c1, c2), weight in self.edges.items():
            if c1 not in positions or c2 not in positions:
                continue
            x1, y1 = positions[c1]
            x2, y2 = positions[c2]
            width   = (weight / max_weight) * 6 + 1

            edge_traces.append(go.Scatter(
                x=[x1, x2, None],
                y=[y1, y2, None],
                mode='lines',
                line=dict(width=width, color='rgba(78,205,196,0.5)'),
                hoverinfo='none',
                showlegend=False
            ))

        # ── Build node trace ──
        max_score   = max(self.nodes.values()) if self.nodes else 1
        node_x      = []
        node_y      = []
        node_colors = []
        node_sizes  = []
        node_labels = []
        hover_texts = []

        for concept, score in self.nodes.items():
            if concept not in positions:
                continue
            x, y = positions[concept]
            node_x.append(x)
            node_y.append(y)
            node_colors.append(CONCEPT_COLORS.get(concept, '#4ECDC4'))
            node_sizes.append((score / max_score) * 40 + 15)
            node_labels.append(concept)

            # Build hover text
            connected = [
                f"{c2}({w:.0f})" if c1 == concept else f"{c1}({w:.0f})"
                for (c1, c2), w in self.edges.items()
                if concept in (c1, c2)
            ]
            hover_texts.append(
                f"<b>{concept}</b><br>"
                f"Score: {score:.2f}<br>"
                f"Connected to: {', '.join(connected)}"
            )

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_labels,
            textposition='middle center',
            textfont=dict(size=11, color='white', family='Arial Black'),
            hovertext=hover_texts,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=1, color='white')
            ),
            showlegend=False
        )

        # ── Build figure ──
        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title=dict(
                    text='Dream Concept Network<br>'
                        '<sub>Node size = importance | '
                        'Edge thickness = co-occurrence | '
                        'Hover for details</sub>',
                    font=dict(color='#4ECDC4', size=16),
                    x=0.5
                ),
                paper_bgcolor='#0D1117',
                plot_bgcolor='#0D1117',
                xaxis=dict(showgrid=False, zeroline=False,
                        showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False,
                        showticklabels=False),
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=80),
            )
        )

        # Save as interactive HTML
        if save_path:
            html_path = save_path.replace('.png', '.html')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_html(html_path)
            print(f"Interactive graph saved to {html_path}")

        fig.show()
        return fig
    

if __name__ == "__main__":

    from pipeline.data_loader import load_sleep_recording
    from pipeline.preprocessor import bandpass_filter, remove_powerline_noise
    from pipeline.feature_extractor import extract_features_from_epoch
    from pipeline.neural_encoder import DreamEncoder
    from pipeline.semantic_decoder import SemanticDecoder
    from configs.config_loader import config
    import torch

    # ── Load and preprocess ──
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

    # ── Build and visualize graph ──
    graph = NarrativeGraph()
    graph.build_from_sequence(decoded_sequence)
    graph.visualize(
        save_path="data/processed/narrative_graph.png"
    )