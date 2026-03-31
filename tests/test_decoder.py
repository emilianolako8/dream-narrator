"""
test_decoder.py
---------------
PURPOSE: Unit tests for the semantic decoder.

WHAT WE TEST:
- Decoder returns correct number of concepts
- Concepts are from the known concept library
- Similarity scores are valid (between -1 and 1)
- Sequence decoding works correctly
- Theme summarization works correctly
- Decoder is stable across multiple runs
"""

import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.semantic_decoder import SemanticDecoder, DREAM_CONCEPTS


# ─────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────

@pytest.fixture
def decoder():
    """Create a SemanticDecoder instance."""
    return SemanticDecoder(embedding_dim=16)


@pytest.fixture
def sample_embedding():
    """Create a random normalized embedding."""
    np.random.seed(42)
    embedding = np.random.randn(16).astype(np.float32)
    embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
    return embedding


@pytest.fixture
def sample_sequence():
    """Create a sample decoded sequence."""
    return [
        {'epoch': 0, 'concepts': [('animal', 0.34),
                                   ('door', 0.24),
                                   ('joy', 0.21)]},
        {'epoch': 1, 'concepts': [('falling', 0.17),
                                   ('family', 0.10),
                                   ('stranger', 0.04)]},
        {'epoch': 2, 'concepts': [('chasing', 0.51),
                                   ('hiding', 0.48),
                                   ('flying', 0.48)]},
        {'epoch': 3, 'concepts': [('family', 0.42),
                                   ('stranger', 0.39),
                                   ('chasing', 0.35)]},
    ]


# ─────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────

def test_decoder_instantiation():
    """Test decoder can be created with various dimensions."""
    for dim in [8, 16, 32]:
        decoder = SemanticDecoder(embedding_dim=dim)
        assert decoder is not None, \
            f"Failed to create decoder with dim={dim}"


def test_decode_returns_correct_count(decoder, sample_embedding):
    """
    Test decode returns exactly top_n concepts.
    
    WHY: If wrong number returned, UI and narrative
    builder receive unexpected input and break.
    """
    for top_n in [1, 3, 5, 10]:
        concepts = decoder.decode(sample_embedding, top_n=top_n)
        assert len(concepts) == top_n, \
            f"Expected {top_n} concepts, got {len(concepts)}"


def test_decode_returns_known_concepts(decoder, sample_embedding):
    """
    Test all decoded concepts are from the concept library.
    
    WHY: Unknown concepts would break the narrative builder
    and concept graph. All outputs must be from known set.
    """
    concepts = decoder.decode(sample_embedding, top_n=5)
    for concept, score in concepts:
        assert concept in DREAM_CONCEPTS, \
            f"Unknown concept returned: {concept}"


def test_decode_scores_are_valid(decoder, sample_embedding):
    """
    Test similarity scores are valid cosine similarities.
    
    WHY: Cosine similarity must be in [-1, 1].
    Values outside this range indicate a math error.
    """
    concepts = decoder.decode(sample_embedding, top_n=5)
    for concept, score in concepts:
        assert -1.0 <= score <= 1.0, \
            f"Invalid score for {concept}: {score}"


def test_decode_sorted_by_score(decoder, sample_embedding):
    """
    Test concepts are returned in descending score order.
    
    WHY: We want top concepts first.
    If unsorted, the "top" concept might not be the best.
    """
    concepts = decoder.decode(sample_embedding, top_n=5)
    scores   = [s for _, s in concepts]

    assert scores == sorted(scores, reverse=True), \
        "Concepts not sorted by score (descending)"


def test_decode_sequence_length(decoder, sample_embedding):
    """Test decode_sequence returns one result per embedding."""
    embeddings = [sample_embedding] * 5
    decoded    = decoder.decode_sequence(embeddings, top_n=3)

    assert len(decoded) == 5, \
        f"Expected 5 decoded epochs, got {len(decoded)}"


def test_decode_sequence_epoch_numbers(decoder, sample_embedding):
    """Test epoch numbers are correct in sequence output."""
    n          = 4
    embeddings = [sample_embedding] * n
    decoded    = decoder.decode_sequence(embeddings, top_n=3)

    for i, epoch_data in enumerate(decoded):
        assert epoch_data['epoch'] == i, \
            f"Wrong epoch number: expected {i}, got {epoch_data['epoch']}"


def test_summarize_sequence(decoder, sample_sequence):
    """
    Test theme summarization counts correctly.
    
    WHY: Theme counts drive the narrative structure.
    Wrong counts = wrong dominant themes = wrong narrative.
    """
    themes = decoder.summarize_sequence(sample_sequence)

    # Convert to dict for easy lookup
    theme_dict = {t[0]: t[1] for t in themes}

    # 'chasing' appears in epochs 2 and 3 = 2 times
    assert 'chasing' in theme_dict, \
        "chasing should be in themes"
    assert theme_dict['chasing']['count'] == 2, \
        f"chasing should appear 2x, got {theme_dict['chasing']['count']}"

    # 'family' appears in epochs 1 and 3 = 2 times
    assert 'family' in theme_dict, \
        "family should be in themes"
    assert theme_dict['family']['count'] == 2, \
        f"family should appear 2x, got {theme_dict['family']['count']}"


def test_decoder_stability(decoder, sample_embedding):
    """
    Test decoder produces same results on repeated calls.
    
    WHY: Our decoder is deterministic (no randomness).
    Same input should always produce same output.
    Non-determinism would make results unreproducible.
    """
    concepts1 = decoder.decode(sample_embedding, top_n=3)
    concepts2 = decoder.decode(sample_embedding, top_n=3)

    for (c1, s1), (c2, s2) in zip(concepts1, concepts2):
        assert c1 == c2, \
            f"Unstable concept: {c1} vs {c2}"
        assert abs(s1 - s2) < 1e-6, \
            f"Unstable score: {s1} vs {s2}"


def test_different_embeddings_different_concepts(decoder):
    """
    Test that very different embeddings produce different top concepts.
    
    WHY: If all embeddings decode to the same concept,
    the decoder is not discriminating between brain states.
    """
    np.random.seed(0)
    e1 = np.array([1.0] + [0.0] * 15)
    e2 = np.array([0.0] * 15 + [1.0])

    top1 = decoder.decode(e1, top_n=1)[0][0]
    top2 = decoder.decode(e2, top_n=1)[0][0]

    assert top1 != top2, \
        "Completely different embeddings decoded to same concept"