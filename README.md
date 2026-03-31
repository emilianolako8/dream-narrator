# Dream Narrator
### Neural EEG Decoding Pipeline — Brain Waves → Dream Stories

> *The first open-source pipeline that decodes REM sleep EEG recordings into full dream narratives using signal processing, deep learning, and LLM-based story reconstruction.*

![Python](https://img.shields.io/badge/Python-3.14-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?style=flat-square&logo=pytorch)
![MNE](https://img.shields.io/badge/MNE-1.6-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
![Tests](https://img.shields.io/badge/Tests-26%2F26%20passing-brightgreen?style=flat-square)

---

## What is Dream Narrator?

Dream Narrator is a complete neural decoding research pipeline that transforms raw EEG brain signals recorded during REM sleep into coherent dream narratives.

It combines:
- **Neuroscience** — sleep stage detection, frequency band analysis
- **Signal Processing** — artifact removal, bandpass filtering, PSD estimation
- **Deep Learning** — autoencoder-based dream fingerprint generation
- **Semantic Decoding** — concept library alignment via cosine similarity
- **LLM Narration** — multi-step story chain using Llama 3 via Groq

---

## Pipeline Overview
```
Raw EEG (.edf)
      ↓
Preprocessing     → bandpass filter, artifact removal
      ↓
Feature Extraction → frequency band powers, theta/alpha ratio
      ↓
Neural Encoder    → 16-dimensional dream fingerprint
      ↓
Semantic Decoder  → decoded concepts per epoch
      ↓
Story Chain       → setting + arc + characters + narrative
      ↓
Dream Report      → full narrative + visualizations
```

---

## Results

Given a real PhysioNet Sleep-EDF recording, the pipeline produces:

**Decoded Neural Sequence:**
```
Epoch  0: animal, door, joy
Epoch  1: falling, running, family
Epoch  5: darkness, room, house
Epoch  6: hiding, chasing, flying
Epoch  9: chasing, flying, hiding
```

**Dominant Themes:** chasing (3x), hiding (3x), flying (3x), family (3x)

**Generated Narrative (vivid style):**
> *You lifted off the ground, your body defying gravity as you took to the skies, flying above the treetops, the wind rushing past your face. But the exhilaration was short-lived, as the darkness below began to take shape, and you realized that you were being chased — and the nightmare was far from over.*

---

## Project Structure
```
dream-narrator/
│
├── pipeline/
│   ├── data_loader.py          # Load EDF sleep recordings
│   ├── preprocessor.py         # Signal cleaning and filtering
│   ├── feature_extractor.py    # Frequency band power extraction
│   ├── neural_encoder.py       # Autoencoder dream fingerprints
│   ├── semantic_decoder.py     # Concept library decoding
│   └── narrative_builder.py    # LLM narrative generation
│
├── llm/
│   ├── prompt_templates.py     # 5 narrative styles
│   ├── character_extractor.py  # Recurring dream character detection
│   └── story_chain.py          # Multi-step LLM pipeline
│
├── viz/
│   ├── dream_timeline.py       # Concept timeline visualization
│   ├── narrative_graph.py      # Interactive concept network
│   └── dashboard.py            # Full Streamlit web app
│
├── notebooks/
│   ├── 01_explore_data.ipynb       # EEG data exploration
│   ├── 02_train_encoder.ipynb      # Encoder training
│   ├── 03_decode_and_narrate.ipynb # Full pipeline walkthrough
│   └── 04_evaluate_results.ipynb   # Evaluation metrics
│
├── tests/
│   ├── test_preprocessor.py    # 7 preprocessing tests
│   ├── test_encoder.py         # 9 encoder tests
│   └── test_decoder.py         # 10 decoder tests
│
├── configs/
│   ├── config.yaml             # All project settings
│   └── config_loader.py        # Config management
│
├── models/
│   └── dream_encoder.pt        # Trained encoder weights
│
└── main.py                     # Single command pipeline runner
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/emilianolako8/dream-narrator.git
cd dream-narrator
```

### 2. Create virtual environment
```bash
py -3.14 -m venv venv
source venv/bin/activate  # Windows: source venv/Scripts/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API keys
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free Groq API key at [console.groq.com](https://console.groq.com)

### 5. Download EEG data
```bash
python3 -c "import mne; mne.datasets.sleep_physionet.age.fetch_data(subjects=[0], recording=[1])"
```

---

## Usage

### Run the full pipeline
```bash
python main.py
```

### Launch the web dashboard
```bash
streamlit run viz/dashboard.py
```

### Run tests
```bash
pytest tests/ -v
```

### Explore notebooks
```bash
jupyter notebook
```

---

## Narrative Styles

The pipeline supports 5 different narrative styles:

| Style | Description |
|---|---|
| `vivid` | Immersive, cinematic dream narrative in second person |
| `psychological` | Jungian analysis of dream symbols and meaning |
| `poetic` | Lyrical, metaphorical dream interpretation |
| `cinematic` | Scene description like a film director's notes |
| `neuroscience` | Explains dream content through brain processes |

Change the style in `configs/config.yaml`:
```yaml
llm:
  style: "psychological"
```

---

## Neuroscience Background

### Why EEG for Dream Decoding?
During REM sleep the brain produces characteristic **theta waves (4-8 Hz)** — the frequency of hippocampal memory replay. By analyzing the power and distribution of these waves across electrodes, we can infer what cognitive processes are active.

### Frequency Bands
| Band | Frequency | Dream Relevance |
|---|---|---|
| Delta | 0.5-4 Hz | Deep sleep marker |
| **Theta** | **4-8 Hz** | **Memory replay, dreaming** |
| Alpha | 8-13 Hz | Relaxed wakefulness |
| Beta | 13-30 Hz | Active cognition |

### The Continuity Hypothesis
Research shows dream content reflects waking experiences and emotional states. High anxiety → more threat-related dream content. This pipeline is designed to eventually incorporate emotional state markers as conditioning variables.

---

## Evaluation

**26/26 tests passing**

| Component | Metric | Score |
|---|---|---|
| Preprocessing | SNR improvement | +measured dB |
| Feature extraction | Features per epoch | 63 |
| Neural encoder | Compression ratio | 3.9x |
| Semantic decoder | Decoding stability | 100% |
| Full pipeline | End-to-end | ✓ Operational |

---

## Roadmap

### Short term
- [ ] Integrate THINGS-EEG for real EEG-concept grounding
- [ ] Add file uploader to dashboard for custom recordings

### Medium term
- [ ] DreamBank cross-modal supervision
- [ ] Anxiety/stress marker integration
- [ ] Multi-subject evaluation

### Long term
- [ ] Text-to-EEG generative model
- [ ] Real-time REM detection
- [ ] Clinical validation study

---

## Tech Stack

| Layer | Technology |
|---|---|
| EEG Processing | MNE-Python |
| Deep Learning | PyTorch |
| Signal Processing | SciPy, NumPy |
| LLM | Llama 3 via Groq |
| Visualization | Plotly, Matplotlib |
| Web App | Streamlit |
| Data | PhysioNet Sleep-EDF |
| Testing | pytest |

---

## Citation

If you use this project in your research:
```
Lako, E. (2026). Dream Narrator: Neural EEG Decoding Pipeline
for Dream Narrative Reconstruction. GitHub.
https://github.com/emilianolako8/dream-narrator
```

---

## Author

**Emiliano Lako**
Electronics Engineering Student | Neural Interface Engineer

Building at the intersection of neuroscience, signal processing, and AI.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*"The stuff that dreams are made of." — Shakespeare*