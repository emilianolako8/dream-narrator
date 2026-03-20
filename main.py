"""
main.py
-------
PURPOSE: Single entrypoint for the entire dream-narrator pipeline.
Run this file to process EEG data and generate a dream narrative.

USAGE:
    python main.py

HOW IT WORKS:
    1. Loads config from configs/config.yaml
    2. Loads and preprocesses EEG data
    3. Extracts features from epochs
    4. Trains neural encoder (or loads existing model)
    5. Decodes concepts from brain fingerprints
    6. Builds dream narrative using LLM
    7. Saves and displays the full report

WHY main.py EXISTS:
    Professional ML projects always have a single entrypoint.
    It ties all pipeline components together cleanly.
    Anyone who clones your repo can run the entire system
    with just: python main.py
"""

import os
import sys
import numpy as np
import torch

# ── Add project root to Python path ──
# This ensures all pipeline imports work correctly
# no matter where Python is called from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config_loader import config
from pipeline.data_loader import load_sleep_recording, print_recording_info
from pipeline.preprocessor import bandpass_filter, remove_powerline_noise
from pipeline.feature_extractor import extract_features_from_epoch
from pipeline.neural_encoder import DreamEncoder, DreamEncoderTrainer
from pipeline.semantic_decoder import SemanticDecoder
from pipeline.narrative_builder import NarrativeBuilder


def print_header():
    """Print a nice header when the pipeline starts."""
    print("=" * 60)
    print("         DREAM NARRATOR — NEURAL DECODING PIPELINE")
    print("         EEG → Features → Embeddings → Dream Story")
    print("=" * 60)
    print()


def step1_load_data():
    """
    STEP 1: Load raw EEG data.
    
    Reads paths and settings from config.yaml.
    Returns the raw MNE object and annotations.
    """
    print("STEP 1: Loading EEG data...")
    print("-" * 40)

    # Read paths from config — no hardcoded values
    data_dir       = config['data']['raw_dir']
    psg_path       = os.path.join(data_dir, config['data']['psg_file'])
    hypnogram_path = os.path.join(data_dir, config['data']['hypnogram_file'])

    raw, annotations = load_sleep_recording(psg_path, hypnogram_path)
    print_recording_info(raw)

    return raw, annotations


def step2_preprocess(raw):
    """
    STEP 2: Clean the EEG signal.
    
    Applies bandpass filter using settings from config.
    Returns cleaned raw object.
    """
    print("STEP 2: Preprocessing EEG signal...")
    print("-" * 40)

    low  = config['preprocessing']['bandpass_low']
    high = config['preprocessing']['bandpass_high']

    raw = remove_powerline_noise(raw)
    raw = bandpass_filter(raw, low_freq=low, high_freq=high)

    print(f"Signal cleaned. Frequency range: {low}-{high} Hz")
    print()

    return raw


def step3_extract_features(raw):
    """
    STEP 3: Extract features from EEG epochs.
    
    Slices the recording into 30-second epochs.
    Extracts frequency band powers from each.
    Returns normalized feature list.
    """
    print("STEP 3: Extracting features from epochs...")
    print("-" * 40)

    sfreq          = raw.info['sfreq']
    data           = raw.get_data()
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

    # Normalize features
    features_array = np.array(features_list)
    mean           = features_array.mean(axis=0)
    std            = features_array.std(axis=0) + 1e-10
    features_list  = [(f - mean) / std for f in features_list]

    print(f"Extracted features from {n_epochs} epochs")
    print(f"Feature vector size: {len(features_list[0])}")
    print()

    return features_list, sfreq


def step4_encode(features_list):
    """
    STEP 4: Encode features into dream fingerprints.
    
    Loads existing model if available.
    Otherwise trains a new encoder from scratch.
    Returns list of 16-number embeddings.
    """
    print("STEP 4: Encoding brain states into dream fingerprints...")
    print("-" * 40)

    input_dim     = len(features_list[0])
    embedding_dim = config['encoder']['embedding_dim']
    model_path    = config['encoder']['model_path']
    model         = DreamEncoder(
        input_dim=input_dim,
        embedding_dim=embedding_dim
    )

    # Load existing model or train a new one
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print("No existing model found. Training new encoder...")
        trainer  = DreamEncoderTrainer(
            model,
            learning_rate=config['encoder']['learning_rate']
        )
        n_epochs = config['encoder']['n_training_epochs']
        for epoch in range(n_epochs):
            avg_loss = trainer.train_epoch(features_list)
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:3d}/{n_epochs} | Loss: {avg_loss:.6f}")
        trainer.save_model(model_path)

    # Generate embeddings
    model.eval()
    embeddings = []
    with torch.no_grad():
        for features in features_list:
            x            = torch.FloatTensor(features).unsqueeze(0)
            embedding, _ = model(x)
            embeddings.append(embedding.numpy().flatten())

    print(f"Generated {len(embeddings)} dream fingerprints")
    print(f"Fingerprint size: {len(embeddings[0])} dimensions")
    print()

    # Save embeddings
    embeddings_path = config['output']['embeddings_path']
    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
    np.save(embeddings_path, np.array(embeddings))
    print(f"Embeddings saved to {embeddings_path}")
    print()

    return embeddings


def step5_decode_concepts(embeddings):
    """
    STEP 5: Decode embeddings into human concepts.
    
    Compares each fingerprint to concept library.
    Returns decoded sequence and dominant themes.
    """
    print("STEP 5: Decoding brain states into concepts...")
    print("-" * 40)

    embedding_dim    = config['encoder']['embedding_dim']
    top_n            = config['decoder']['top_n_concepts']
    decoder          = SemanticDecoder(embedding_dim=embedding_dim)
    decoded_sequence = decoder.decode_sequence(embeddings, top_n=top_n)
    dominant_themes  = decoder.summarize_sequence(decoded_sequence)

    print("Decoded concept sequence:")
    for epoch_data in decoded_sequence:
        concepts    = [c for c, s in epoch_data['concepts']]
        print(f"  Epoch {epoch_data['epoch']:2d}: {', '.join(concepts)}")

    print()
    print("Dominant themes:")
    for concept, data in dominant_themes[:config['decoder']['top_n_themes']]:
        bar = "█" * data['count']
        print(f"  {concept:15} {bar}")

    print()
    return decoded_sequence, dominant_themes


def step6_build_narrative(decoded_sequence, dominant_themes):
    """
    STEP 6: Build dream narrative using LLM.
    
    Sends decoded concepts to Llama 3 via Groq.
    Returns full narrative text.
    """
    print("STEP 6: Building dream narrative with Llama 3...")
    print("-" * 40)

    builder   = NarrativeBuilder()
    narrative = builder.build_narrative(decoded_sequence, dominant_themes)
    report    = builder.build_dream_report(
        decoded_sequence,
        dominant_themes,
        narrative
    )

    # Save report
    if config['output']['save_report']:
        report_path = config['output']['report_path']
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to {report_path}")

    print()
    return report


def main():
    """
    Master function — runs the entire pipeline.
    
    WHY a main() function:
    Wrapping everything in main() is standard Python practice.
    It keeps the code organized and makes it importable
    without running automatically.

    The if __name__ == "__main__" check at the bottom
    ensures main() only runs when you execute this file
    directly — not when another file imports from it.
    """
    print_header()

    # Run all pipeline steps in order
    raw, annotations          = step1_load_data()
    raw                       = step2_preprocess(raw)
    features_list, sfreq      = step3_extract_features(raw)
    embeddings                = step4_encode(features_list)
    decoded_sequence, themes  = step5_decode_concepts(embeddings)
    report                    = step6_build_narrative(
                                    decoded_sequence,
                                    themes
                                )

    # Print final report
    print(report)
    print("\nPipeline complete!")


if __name__ == "__main__":
    main()