"""
dashboard.py
------------
PURPOSE: Full web dashboard for the dream-narrator pipeline.

WHY STREAMLIT:
Streamlit turns Python scripts into web apps with zero HTML/CSS/JS.
It's the standard tool for ML demo apps in research and industry.
Anyone can use your pipeline without touching code.

HOW TO RUN:
    streamlit run viz/dashboard.py

WHAT IT DOES:
- Shows project introduction
- Lets user select narrative style
- Runs the full pipeline on the downloaded EEG data
- Displays brain wave visualization
- Displays decoded concept timeline
- Displays interactive concept graph
- Displays full dream narrative
"""

import streamlit as st
import numpy as np
import torch
import os
import sys
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config_loader import config
from pipeline.data_loader import load_sleep_recording
from pipeline.preprocessor import bandpass_filter, remove_powerline_noise
from pipeline.feature_extractor import extract_features_from_epoch
from pipeline.neural_encoder import DreamEncoder, DreamEncoderTrainer
from pipeline.semantic_decoder import SemanticDecoder
from pipeline.narrative_builder import NarrativeBuilder
from llm.story_chain import StoryChain
from llm.prompt_templates import list_styles, NARRATIVE_STYLES
from viz.dream_timeline import CONCEPT_COLORS
from viz.narrative_graph import NarrativeGraph


# ─────────────────────────────────────────
# PAGE CONFIG
# Must be the first Streamlit command
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Dream Narrator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─────────────────────────────────────────
# CUSTOM CSS
# Dark theme matching our visualization style
# ─────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0D1117; }
    .stApp { background-color: #0D1117; }
    h1, h2, h3, h4 { color: #4ECDC4; }
    p, li { color: #C9D1D9; }
    .stButton>button {
        background-color: #4ECDC4;
        color: #0D1117;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        padding: 0.5rem 2rem;
    }
    .stButton>button:hover {
        background-color: #3BA39A;
    }
    .metric-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .narrative-box {
        background-color: #161B22;
        border-left: 3px solid #4ECDC4;
        border-radius: 4px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #C9D1D9;
        font-size: 1.05rem;
        line-height: 1.8;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# CACHED FUNCTIONS
# @st.cache_data tells Streamlit to cache
# the result so it doesn't rerun on every
# user interaction. Essential for performance.
# ─────────────────────────────────────────

@st.cache_data
def load_and_preprocess():
    """Load and preprocess EEG data. Cached so it only runs once."""
    data_dir       = config['data']['raw_dir']
    psg_path       = os.path.join(data_dir, config['data']['psg_file'])
    hypnogram_path = os.path.join(data_dir, config['data']['hypnogram_file'])

    raw, annotations = load_sleep_recording(psg_path, hypnogram_path)
    raw = remove_powerline_noise(raw)
    raw = bandpass_filter(raw)

    return raw, annotations


@st.cache_data
def extract_and_encode(_raw):
    """Extract features and encode into embeddings. Cached."""
    sfreq          = _raw.info['sfreq']
    data           = _raw.get_data()
    epoch_duration = int(sfreq * config['preprocessing']['epoch_duration'])
    n_epochs       = min(
        config['preprocessing']['n_epochs'],
        data.shape[1] // epoch_duration
    )

    features_list = []
    for i in range(n_epochs):
        start      = i * epoch_duration
        end        = start + epoch_duration
        epoch_data = data[:, start:end]
        features, names = extract_features_from_epoch(epoch_data, sfreq)
        features_list.append(features)

    # Normalize
    features_array = np.array(features_list)
    mean           = features_array.mean(axis=0)
    std            = features_array.std(axis=0) + 1e-10
    features_list  = [(f - mean) / std for f in features_list]

    # Load or train encoder
    input_dim  = len(features_list[0])
    model      = DreamEncoder(
        input_dim=input_dim,
        embedding_dim=config['encoder']['embedding_dim']
    )
    model_path = config['encoder']['model_path']

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        trainer = DreamEncoderTrainer(model)
        for _ in range(config['encoder']['n_training_epochs']):
            trainer.train_epoch(features_list)
        trainer.save_model(model_path)

    model.eval()
    embeddings = []
    with torch.no_grad():
        for features in features_list:
            x            = torch.FloatTensor(features).unsqueeze(0)
            embedding, _ = model(x)
            embeddings.append(embedding.numpy().flatten())

    return embeddings, features_list


def decode_concepts(embeddings):
    """Decode embeddings into concepts."""
    decoder          = SemanticDecoder(
        embedding_dim=config['encoder']['embedding_dim']
    )
    decoded_sequence = decoder.decode_sequence(
        embeddings,
        top_n=config['decoder']['top_n_concepts']
    )
    dominant_themes  = decoder.summarize_sequence(decoded_sequence)
    return decoded_sequence, dominant_themes


def build_concept_timeline_plotly(decoded_sequence, dominant_themes):
    """Build interactive concept timeline using Plotly."""
    fig = go.Figure()

    for epoch_data in decoded_sequence:
        epoch    = epoch_data['epoch']
        concepts = epoch_data['concepts']
        for i, (concept, score) in enumerate(concepts):
            color = CONCEPT_COLORS.get(concept, '#4ECDC4')
            fig.add_trace(go.Bar(
                x=[epoch],
                y=[score],
                name=concept,
                marker_color=color,
                text=concept,
                textposition='outside',
                hovertemplate=f"<b>{concept}</b><br>"
                              f"Epoch: {epoch}<br>"
                              f"Score: {score:.2f}<extra></extra>",
                showlegend=True
            ))

    fig.update_layout(
        barmode='group',
        paper_bgcolor='#0D1117',
        plot_bgcolor='#161B22',
        font=dict(color='white'),
        title=dict(
            text='Decoded Dream Concepts Per Epoch',
            font=dict(color='#4ECDC4', size=16)
        ),
        xaxis=dict(
            title='Epoch',
            gridcolor='#30363D',
            color='white'
        ),
        yaxis=dict(
            title='Confidence Score',
            gridcolor='#30363D',
            color='white'
        ),
        legend=dict(
            bgcolor='#161B22',
            bordercolor='#30363D',
            font=dict(color='white')
        ),
        height=400
    )
    return fig


def build_brain_wave_plotly(raw):
    """Build interactive brain wave plot."""
    sfreq     = raw.info['sfreq']
    n_samples = int(sfreq * 30)
    data      = raw.get_data()
    times     = raw.times[:n_samples]

    colors = ['#4ECDC4', '#FF6B6B', '#FFD700',
              '#95E1D3', '#FF8E53', '#9370DB', '#3CB371']

    fig = go.Figure()
    for i, ch_name in enumerate(raw.ch_names):
        channel_data = data[i, :n_samples] * 1e6
        color        = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=times,
            y=channel_data,
            name=ch_name,
            line=dict(color=color, width=1),
            hovertemplate=f"<b>{ch_name}</b><br>"
                         f"Time: %{{x:.2f}}s<br>"
                         f"Amplitude: %{{y:.2f}}μV<extra></extra>"
        ))

    fig.update_layout(
        paper_bgcolor='#0D1117',
        plot_bgcolor='#161B22',
        font=dict(color='white'),
        title=dict(
            text='Raw EEG Brain Waves (First 30 seconds)',
            font=dict(color='#4ECDC4', size=16)
        ),
        xaxis=dict(
            title='Time (seconds)',
            gridcolor='#30363D',
            color='white'
        ),
        yaxis=dict(
            title='Amplitude (μV)',
            gridcolor='#30363D',
            color='white'
        ),
        legend=dict(
            bgcolor='#161B22',
            bordercolor='#30363D',
            font=dict(color='white')
        ),
        height=400
    )
    return fig


def build_graph_plotly(decoded_sequence):
    """Build interactive concept network graph."""
    graph = NarrativeGraph()
    graph.build_from_sequence(decoded_sequence)

    if not graph.nodes:
        return None

    positions   = graph.get_node_positions()
    edge_traces = []
    max_weight  = max(graph.edges.values()) if graph.edges else 1

    for (c1, c2), weight in graph.edges.items():
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

    max_score   = max(graph.nodes.values()) if graph.nodes else 1
    node_x      = []
    node_y      = []
    node_colors = []
    node_sizes  = []
    node_labels = []
    hover_texts = []

    for concept, score in graph.nodes.items():
        if concept not in positions:
            continue
        x, y = positions[concept]
        node_x.append(x)
        node_y.append(y)
        node_colors.append(CONCEPT_COLORS.get(concept, '#4ECDC4'))
        node_sizes.append((score / max_score) * 40 + 15)
        node_labels.append(concept)
        connected = [
            f"{c2}({w:.0f})" if c1 == concept else f"{c1}({w:.0f})"
            for (c1, c2), w in graph.edges.items()
            if concept in (c1, c2)
        ]
        hover_texts.append(
            f"<b>{concept}</b><br>"
            f"Score: {score:.2f}<br>"
            f"Connected: {', '.join(connected)}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_labels,
        textposition='middle center',
        textfont=dict(size=11, color='white',
                     family='Arial Black'),
        hovertext=hover_texts,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1, color='white')
        ),
        showlegend=False
    )

    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            paper_bgcolor='#0D1117',
            plot_bgcolor='#0D1117',
            font=dict(color='white'),
            title=dict(
                text='Dream Concept Network',
                font=dict(color='#4ECDC4', size=16)
            ),
            xaxis=dict(showgrid=False, zeroline=False,
                      showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False,
                      showticklabels=False),
            hovermode='closest',
            height=500
        )
    )
    return fig


# ─────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────

def main():

    # ── HEADER ──
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='font-size: 3rem; color: #4ECDC4;'>🧠 DREAM NARRATOR</h1>
        <p style='font-size: 1.2rem; color: #8B949E;'>
            Neural EEG Decoding Pipeline &nbsp;|&nbsp; 
            Brain Waves → Dream Stories
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        style = st.selectbox(
            "Narrative Style",
            options=list(NARRATIVE_STYLES.keys()),
            index=0,
            help="Choose how the dream narrative is written"
        )

        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        **Dream Narrator** decodes EEG brain signals 
        from REM sleep into dream narratives using:
        - Signal processing (MNE)
        - Deep learning (PyTorch)
        - LLM narrative generation (Groq/Llama 3)
        """)

        st.markdown("---")
        st.markdown("### Pipeline")
        st.markdown("""
        1. Load EEG (.edf)
        2. Filter & clean signal
        3. Extract frequency features
        4. Encode → dream fingerprint
        5. Decode → concepts
        6. Generate → narrative
        """)

        run_button = st.button("🚀 Run Pipeline", use_container_width=True)

    # ── MAIN CONTENT ──
    if not run_button:
        # Show intro when pipeline hasn't run yet
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class='metric-card'>
                <h3>📡 EEG Decoding</h3>
                <p>Processes real polysomnography recordings 
                from the PhysioNet Sleep-EDF dataset.</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class='metric-card'>
                <h3>🤖 Neural Encoding</h3>
                <p>Autoencoder compresses brain features 
                into 16-dimensional dream fingerprints.</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class='metric-card'>
                <h3>✨ LLM Narration</h3>
                <p>Llama 3 reconstructs dream narratives 
                from decoded concept sequences.</p>
            </div>
            """, unsafe_allow_html=True)

        st.info("👈 Click **Run Pipeline** in the sidebar to start!")
        return

    # ── RUN PIPELINE ──
    with st.spinner("Loading and preprocessing EEG data..."):
        raw, annotations = load_and_preprocess()

    with st.spinner("Extracting features and encoding..."):
        embeddings, features_list = extract_and_encode(raw)

    with st.spinner("Decoding brain states into concepts..."):
        decoded_sequence, dominant_themes = decode_concepts(embeddings)

    # ── METRICS ──
    st.markdown("## 📊 Pipeline Results")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recording Duration",
                f"{raw.times[-1]/60:.1f} min")
    col2.metric("Epochs Processed",
                len(decoded_sequence))
    col3.metric("Unique Concepts",
                len(set([
                    c for e in decoded_sequence
                    for c, s in e['concepts']
                ])))
    col4.metric("Dominant Theme",
                dominant_themes[0][0] if dominant_themes else "N/A")

    st.divider()

    # ── BRAIN WAVES ──
    st.markdown("## 🌊 Raw Brain Waves")
    st.markdown("*Real EEG voltage measurements from a sleeping human*")
    brain_fig = build_brain_wave_plotly(raw)
    st.plotly_chart(brain_fig, use_container_width=True)

    st.divider()

    # ── CONCEPT TIMELINE ──
    st.markdown("## 📡 Decoded Dream Concepts")
    st.markdown("*Concepts decoded from brain fingerprints across time*")
    timeline_fig = build_concept_timeline_plotly(
        decoded_sequence, dominant_themes
    )
    st.plotly_chart(timeline_fig, use_container_width=True)

    st.divider()

    # ── CONCEPT GRAPH ──
    st.markdown("## 🕸️ Dream Concept Network")
    st.markdown("*How concepts relate to each other in this dream*")
    graph_fig = build_graph_plotly(decoded_sequence)
    if graph_fig:
        st.plotly_chart(graph_fig, use_container_width=True)

    st.divider()

    # ── NARRATIVE ──
    st.markdown("## ✨ Dream Narrative")
    st.markdown(f"*Style: **{style}***")

    with st.spinner(f"Generating {style} dream narrative with Llama 3..."):
        chain  = StoryChain(style=style)
        result = chain.run(decoded_sequence, dominant_themes)

    # Show titles
    st.markdown("### 🎬 Suggested Dream Titles")
    st.markdown(f"*{result['titles']}*")

    # Show setting
    st.markdown("### 🏛️ Dream Setting")
    st.markdown(f"*{result['setting']}*")

    # Show emotional arc
    st.markdown("### 💫 Emotional Arc")
    st.markdown(f"*{result['emotional_arc']}*")

    # Show characters
    if result['characters']:
        st.markdown("### 🎭 Dream Characters")
        for char in result['characters']:
            st.markdown(
                f"**{char['name'].upper()}** "
                f"(appeared {char['appearances']}x) — "
                f"{char['description']}"
            )

    # Show full narrative
    st.markdown("### 📖 Full Narrative")
    st.markdown(
        f"<div class='narrative-box'>{result['narrative']}</div>",
        unsafe_allow_html=True
    )

    # Download button
    full_report = f"""DREAM NARRATOR REPORT
{'='*50}

TITLES:
{result['titles']}

SETTING:
{result['setting']}

EMOTIONAL ARC:
{result['emotional_arc']}

NARRATIVE:
{result['narrative']}
"""
    st.download_button(
        label="📥 Download Dream Report",
        data=full_report,
        file_name="dream_report.txt",
        mime="text/plain"
    )


if __name__ == "__main__":
    main()