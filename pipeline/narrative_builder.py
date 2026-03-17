"""
narrative_builder.py
--------------------
PURPOSE: Transform decoded dream concepts into a coherent narrative.
This is the final creative step of the pipeline.

HOW IT WORKS:
1. Takes the decoded concept sequence from semantic_decoder.py
2. Takes the dominant dream themes
3. Sends them to Llama 3 via Groq API
4. LLM reconstructs a dream narrative from the concepts

WHY AN LLM FOR THIS:
The concepts we decoded (chasing, hiding, flying, darkness) are
just fragments. A language model has read millions of dream journals,
stories, and psychological texts. It knows how these concepts
connect into coherent narratives.

We're not asking it to invent a dream — we're asking it to
reconstruct the most likely narrative that would produce
these specific brain state patterns in this specific order.
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class NarrativeBuilder:
    """
    Builds dream narratives from decoded EEG concept sequences.
    """

    def __init__(self):
        self.client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.model  = 'llama-3.1-8b-instant'

    def build_narrative(self, decoded_sequence, dominant_themes):
        """
        Generate a dream narrative from decoded concepts.

        Parameters:
            decoded_sequence : list of epoch dicts with concepts
            dominant_themes  : list of (concept, data) tuples

        Returns:
            narrative : string containing the dream story
        """

        # ── Format the concept sequence for the prompt ──
        sequence_text = ""
        for epoch in decoded_sequence:
            epoch_num = epoch['epoch']
            concepts  = [c for c, s in epoch['concepts']]
            sequence_text += f"  Moment {epoch_num}: {', '.join(concepts)}\n"

        # ── Format dominant themes ──
        themes_text = ", ".join([t[0] for t in dominant_themes[:5]])

        # ── Build the prompt ──
        # This is called "prompt engineering" — carefully crafting
        # the instructions to get the best output from the LLM
        prompt = f"""You are a dream interpreter with deep knowledge of neuroscience and psychology.

I have decoded the following sequence of mental concepts from a person's EEG brain signals during REM sleep. These concepts were extracted from their brain waves in chronological order — each "moment" represents 30 seconds of dream activity.

DECODED BRAIN SIGNAL SEQUENCE:
{sequence_text}

DOMINANT RECURRING THEMES: {themes_text}

Based on this sequence of decoded neural patterns, reconstruct the most likely dream narrative this person experienced. 

Important guidelines:
- Follow the chronological order of the concepts
- Make it feel like a real dream — vivid, slightly surreal, emotionally resonant
- Keep it to 2-3 paragraphs
- Write in second person ("You were...")
- Don't mention EEG or brain signals — just tell the dream story

Dream narrative:"""

        # ── Call the LLM ──
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    'role'   : 'system',
                    'content': 'You are a vivid dream narrator who reconstructs dreams from neural decoded concept sequences. Your narratives are immersive, psychologically rich, and follow the exact sequence of concepts provided.'
                },
                {
                    'role'   : 'user',
                    'content': prompt
                }
            ],
            temperature=0.8,  # higher = more creative
            max_tokens=500
        )

        narrative = response.choices[0].message.content
        return narrative

    def build_dream_report(self, decoded_sequence, dominant_themes, narrative):
        """
        Build a complete dream report combining all pipeline outputs.

        This is the final deliverable of the entire pipeline —
        a structured report that shows the raw decoding AND
        the reconstructed narrative side by side.
        """

        report = []
        report.append("=" * 60)
        report.append("         DREAM NARRATOR — NEURAL DECODING REPORT")
        report.append("=" * 60)

        report.append("\n📡 DECODED NEURAL SEQUENCE:")
        report.append("-" * 40)
        for epoch in decoded_sequence:
            epoch_num = epoch['epoch']
            concepts  = [(c, f"{s:.2f}") for c, s in epoch['concepts']]
            concept_str = ", ".join([f"{c} ({s})" for c, s in concepts])
            report.append(f"  Epoch {epoch_num:2d} | {concept_str}")

        report.append("\n🧠 DOMINANT DREAM THEMES:")
        report.append("-" * 40)
        for concept, data in dominant_themes[:5]:
            bar = "█" * data['count']
            report.append(f"  {concept:15} {bar} (x{data['count']})")

        report.append("\n✨ RECONSTRUCTED DREAM NARRATIVE:")
        report.append("-" * 40)
        report.append(narrative)
        report.append("=" * 60)

        return "\n".join(report)


if __name__ == "__main__":

    import numpy as np
    import torch
    import os

    from pipeline.data_loader      import load_sleep_recording
    from pipeline.preprocessor     import bandpass_filter, remove_powerline_noise
    from pipeline.feature_extractor import extract_features_from_epoch
    from pipeline.neural_encoder   import DreamEncoder
    from pipeline.semantic_decoder import SemanticDecoder

    # ── Load and preprocess ──
    data_dir       = os.path.join(os.path.expanduser("~"), "mne_data", "physionet-sleep-data")
    psg_path       = os.path.join(data_dir, "SC4001E0-PSG.edf")
    hypnogram_path = os.path.join(data_dir, "SC4001EC-Hypnogram.edf")

    raw, annotations = load_sleep_recording(psg_path, hypnogram_path)
    raw = remove_powerline_noise(raw)
    raw = bandpass_filter(raw)

    sfreq = raw.info['sfreq']
    data  = raw.get_data()

    # ── Extract features ──
    print("Extracting features...")
    features_list  = []
    epoch_duration = int(sfreq * 30)
    n_epochs       = min(10, data.shape[1] // epoch_duration)

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

    # ── Load encoder ──
    input_dim = len(features_list[0])
    model     = DreamEncoder(input_dim=input_dim, embedding_dim=16)
    model.load_state_dict(torch.load("models/dream_encoder.pt"))
    model.eval()

    # ── Generate embeddings ──
    print("Generating dream fingerprints...")
    embeddings = []
    with torch.no_grad():
        for features in features_list:
            x            = torch.FloatTensor(features).unsqueeze(0)
            embedding, _ = model(x)
            embeddings.append(embedding.numpy().flatten())

    # ── Decode concepts ──
    print("Decoding concepts...")
    decoder          = SemanticDecoder(embedding_dim=16)
    decoded_sequence = decoder.decode_sequence(embeddings, top_n=3)
    dominant_themes  = decoder.summarize_sequence(decoded_sequence)

    # ── Build narrative ──
    print("Building dream narrative with Llama 3...\n")
    builder   = NarrativeBuilder()
    narrative = builder.build_narrative(decoded_sequence, dominant_themes)

    # ── Print full report ──
    report = builder.build_dream_report(decoded_sequence, dominant_themes, narrative)
    print(report)

    # ── Save report ──
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/dream_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("\nReport saved to data/processed/dream_report.txt")