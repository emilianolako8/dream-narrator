"""
semantic_decoder.py
-------------------
PURPOSE: Map EEG dream fingerprints to human-readable concepts.
This is the bridge between brain signals and language.

HOW IT WORKS:
1. We define a library of concept embeddings
2. For each dream fingerprint, we find the closest concepts
3. We return the top-N most likely concepts for that brain state

WHY THIS APPROACH:
Real semantic decoding requires thousands of labeled EEG-concept pairs.
We don't have that yet. So we use a "concept library" approach:
- Define concepts as vectors in the same space as our embeddings
- Use cosine similarity to find which concepts match each brain state
- This gives us meaningful output even with limited training data
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
import os


# ─────────────────────────────────────────────────────────
# CONCEPT LIBRARY
# These are the semantic concepts we can decode.
# In a full system this would come from a dataset like
# THINGS-EEG which has 22,000 concepts paired with EEG.
# For now we define a curated set of dream-relevant concepts.
# ─────────────────────────────────────────────────────────

DREAM_CONCEPTS = [
    # People and social
    "person", "face", "crowd", "stranger", "family",
    # Places
    "house", "room", "forest", "water", "city", "darkness",
    # Actions
    "running", "flying", "falling", "chasing", "hiding",
    # Emotions
    "fear", "joy", "confusion", "peace", "anxiety",
    # Objects
    "door", "light", "animal", "vehicle", "stairs",
    # Abstract
    "transformation", "memory", "unknown", "danger", "safety"
]


class ConceptLibrary:
    """
    A library of concept vectors in embedding space.

    WHY:
    We need a way to compare brain fingerprints to known concepts.
    Each concept gets a vector in the same 16-dimensional space
    as our dream fingerprints.

    In a real system these vectors would come from:
    - Training on EEG-image pairs (THINGS-EEG dataset)
    - Aligning with CLIP text embeddings
    - Few-shot learning from labeled examples

    For now we initialize them smartly and let the
    alignment model learn to map brain states to them.
    """

    def __init__(self, embedding_dim=16, concepts=None):
        self.embedding_dim = embedding_dim
        self.concepts = concepts or DREAM_CONCEPTS
        self.n_concepts = len(self.concepts)

        # Initialize concept vectors
        # We use a structured initialization based on
        # known neuroscience of dream content categories
        self.concept_vectors = self._initialize_concept_vectors()

    def _initialize_concept_vectors(self):
        """
        Initialize concept vectors with structure.

        WHY NOT RANDOM:
        Random initialization means all concepts are equally
        similar to each brain state — useless for decoding.

        Instead we group concepts by category and initialize
        them with similar vectors within each category.
        This gives the decoder a meaningful starting point.
        """
        np.random.seed(42)  # reproducibility

        vectors = {}

        # Define category centers in embedding space
        # These positions are chosen to be well-separated
        category_centers = {
             'social'  : np.array([0.8, 0.2, -0.3] + [0.1] * (self.embedding_dim - 3)),
             'places'  : np.array([-0.3, 0.8, 0.2] + [0.1] * (self.embedding_dim - 3)),
             'actions' : np.array([0.2, -0.3, 0.8] + [0.1] * (self.embedding_dim - 3)),
             'emotions': np.array([-0.8, -0.2, 0.3] + [0.1] * (self.embedding_dim - 3)),
             'objects' : np.array([0.3, -0.8, -0.2] + [0.1] * (self.embedding_dim - 3)),
             'abstract': np.array([-0.2, 0.3, -0.8] + [0.1] * (self.embedding_dim - 3)),
        }

        # Map concepts to categories
        concept_categories = {
            'person': 'social', 'face': 'social', 'crowd': 'social',
            'stranger': 'social', 'family': 'social',
            'house': 'places', 'room': 'places', 'forest': 'places',
            'water': 'places', 'city': 'places', 'darkness': 'places',
            'running': 'actions', 'flying': 'actions', 'falling': 'actions',
            'chasing': 'actions', 'hiding': 'actions',
            'fear': 'emotions', 'joy': 'emotions', 'confusion': 'emotions',
            'peace': 'emotions', 'anxiety': 'emotions',
            'door': 'objects', 'light': 'objects', 'animal': 'objects',
            'vehicle': 'objects', 'stairs': 'objects',
            'transformation': 'abstract', 'memory': 'abstract',
            'unknown': 'abstract', 'danger': 'abstract', 'safety': 'abstract',
        }

        for concept in self.concepts:
            category = concept_categories.get(concept, 'abstract')
            center = category_centers[category]
            # Add small random noise around category center
            noise = np.random.normal(0, 0.1, self.embedding_dim)
            vector = center + noise
            # Normalize to unit length
            vector = vector / (np.linalg.norm(vector) + 1e-10)
            vectors[concept] = vector

        return vectors

    def get_vector(self, concept):
        """Get the vector for a specific concept."""
        return self.concept_vectors.get(concept)

    def get_all_vectors(self):
        """Get all concept vectors as a matrix."""
        return np.array([self.concept_vectors[c] for c in self.concepts])

    def get_all_concepts(self):
        """Get list of all concept names."""
        return self.concepts


class SemanticDecoder:
    """
    Decodes EEG dream fingerprints into human-readable concepts.

    HOW IT WORKS:
    1. Take a dream fingerprint (16 numbers)
    2. Compare it to all concept vectors using cosine similarity
    3. Return the top-N most similar concepts

    WHY COSINE SIMILARITY:
    Cosine similarity measures the ANGLE between two vectors,
    not their magnitude. This means it compares the "direction"
    of brain states, not their strength.

    This is important because the same dream concept might
    produce embeddings of different magnitudes across people
    but similar directions.
    """

    def __init__(self, embedding_dim=16):
        self.embedding_dim = embedding_dim
        self.concept_library = ConceptLibrary(embedding_dim)

    def decode(self, embedding, top_n=5):
        """
        Decode a dream fingerprint into top-N concepts.

        Parameters:
            embedding : numpy array of shape (embedding_dim,)
            top_n     : number of top concepts to return

        Returns:
            List of (concept, similarity_score) tuples
            sorted by similarity (highest first)
        """
        # Ensure embedding is the right shape
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().numpy()
        embedding = embedding.flatten()

        # Get all concept vectors
        concept_vectors = self.concept_library.get_all_vectors()
        concepts = self.concept_library.get_all_concepts()

        # Compute cosine similarity between embedding and all concepts
        # Shape: (1, n_concepts)
        similarities = cosine_similarity(
            embedding.reshape(1, -1),
            concept_vectors
        )[0]

        # Sort by similarity (highest first)
        sorted_indices = np.argsort(similarities)[::-1]

        # Return top-N concepts with scores
        top_concepts = [
            (concepts[i], float(similarities[i]))
            for i in sorted_indices[:top_n]
        ]

        return top_concepts

    def decode_sequence(self, embeddings, top_n=3):
        """
        Decode a sequence of dream fingerprints.
        This is used for a full REM period — multiple epochs.

        Parameters:
            embeddings : list of embedding arrays
            top_n      : concepts per epoch

        Returns:
            List of decoded concept lists (one per epoch)
        """
        decoded_sequence = []

        for i, embedding in enumerate(embeddings):
            concepts = self.decode(embedding, top_n=top_n)
            decoded_sequence.append({
                'epoch'   : i,
                'concepts': concepts
            })

        return decoded_sequence

    def summarize_sequence(self, decoded_sequence):
        """
        Summarize a decoded sequence into a concept frequency map.

        WHY:
        Across a full REM period, some concepts appear repeatedly.
        These recurring concepts are likely the "themes" of the dream.
        We count frequency to find the dominant themes.
        """
        concept_counts = {}

        for epoch_data in decoded_sequence:
            for concept, score in epoch_data['concepts']:
                if concept not in concept_counts:
                    concept_counts[concept] = {'count': 0, 'total_score': 0}
                concept_counts[concept]['count'] += 1
                concept_counts[concept]['total_score'] += score

        # Sort by count then by total score
        sorted_concepts = sorted(
            concept_counts.items(),
            key=lambda x: (x[1]['count'], x[1]['total_score']),
            reverse=True
        )

        return sorted_concepts


if __name__ == "__main__":

    from pipeline.data_loader import load_sleep_recording
    from pipeline.preprocessor import bandpass_filter, remove_powerline_noise
    from pipeline.feature_extractor import extract_features_from_epoch
    from pipeline.neural_encoder import DreamEncoder

    import os

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

    # ── Load trained encoder ──
    input_dim = len(features_list[0])
    model     = DreamEncoder(input_dim=input_dim, embedding_dim=16)
    model.load_state_dict(torch.load("models/dream_encoder.pt"))
    model.eval()

    # ── Generate embeddings ──
    print("Generating dream fingerprints...")
    embeddings = []
    with torch.no_grad():
        for features in features_list:
            x         = torch.FloatTensor(features).unsqueeze(0)
            embedding, _ = model(x)
            embeddings.append(embedding.numpy().flatten())

    # ── Decode concepts ──
    print("\nDecoding brain states into concepts...")
    decoder          = SemanticDecoder(embedding_dim=16)
    decoded_sequence = decoder.decode_sequence(embeddings, top_n=3)

    print("\n--- DECODED DREAM SEQUENCE ---")
    for epoch_data in decoded_sequence:
        epoch   = epoch_data['epoch']
        concepts = epoch_data['concepts']
        concept_str = ", ".join([f"{c} ({s:.2f})" for c, s in concepts])
        print(f"Epoch {epoch:2d}: {concept_str}")

    # ── Summarize themes ──
    print("\n--- DOMINANT DREAM THEMES ---")
    themes = decoder.summarize_sequence(decoded_sequence)
    for concept, data in themes[:5]:
        print(f"  {concept:15}: appeared {data['count']}x | score {data['total_score']:.2f}")

    print("\nSemantic decoding complete!")