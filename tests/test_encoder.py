"""
test_encoder.py
---------------
PURPOSE: Unit tests for the neural encoder.

WHAT WE TEST:
- Model can be instantiated with different dimensions
- Forward pass produces correct output shapes
- Encoder compresses correctly
- Decoder reconstructs correctly
- Model can be saved and loaded
- Training step reduces loss
"""

import pytest
import numpy as np
import torch
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.neural_encoder import DreamEncoder, DreamEncoderTrainer


# ─────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────

@pytest.fixture
def sample_features():
    """Create synthetic normalized feature vectors."""
    np.random.seed(42)
    n_epochs   = 10
    n_features = 63
    features   = np.random.randn(n_epochs, n_features).astype(np.float32)
    return [features[i] for i in range(n_epochs)]


@pytest.fixture
def model():
    """Create a DreamEncoder with standard dimensions."""
    return DreamEncoder(input_dim=63, embedding_dim=16)


@pytest.fixture
def trainer(model):
    """Create a DreamEncoderTrainer."""
    return DreamEncoderTrainer(model, learning_rate=0.001)


# ─────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────

def test_model_instantiation():
    """Test model can be created with various dimensions."""
    for input_dim in [32, 63, 128]:
        for embedding_dim in [8, 16, 32]:
            model = DreamEncoder(
                input_dim=input_dim,
                embedding_dim=embedding_dim
            )
            assert model is not None, \
                f"Failed to create model with {input_dim}→{embedding_dim}"


def test_encoder_output_shape(model):
    """
    Test encoder produces correct embedding shape.
    
    WHY: Wrong shape = everything downstream breaks.
    This is the most critical shape test.
    """
    batch_size = 1
    input_dim  = 63
    x          = torch.randn(batch_size, input_dim)

    embedding = model.encode(x)

    assert embedding.shape == (batch_size, 16), \
        f"Wrong embedding shape: {embedding.shape}"


def test_decoder_output_shape(model):
    """Test decoder reconstructs to original input shape."""
    batch_size    = 1
    embedding_dim = 16
    z             = torch.randn(batch_size, embedding_dim)

    reconstruction = model.decode(z)

    assert reconstruction.shape == (batch_size, 63), \
        f"Wrong reconstruction shape: {reconstruction.shape}"


def test_forward_pass(model):
    """Test full forward pass returns correct shapes."""
    x              = torch.randn(1, 63)
    embedding, recon = model(x)

    assert embedding.shape == (1, 16), \
        f"Wrong embedding shape: {embedding.shape}"
    assert recon.shape == (1, 63), \
        f"Wrong reconstruction shape: {recon.shape}"


def test_embedding_values_bounded(model):
    """
    Test embedding values are bounded between -1 and 1.
    
    WHY: We use Tanh activation at the end of encoder.
    Tanh output is always in [-1, 1].
    Values outside this range indicate a bug.
    """
    model.eval()
    with torch.no_grad():
        x         = torch.randn(5, 63)
        embedding = model.encode(x)

    assert embedding.min() >= -1.0, \
        f"Embedding below -1: {embedding.min()}"
    assert embedding.max() <= 1.0, \
        f"Embedding above 1: {embedding.max()}"


def test_no_nan_in_output(model):
    """Test forward pass produces no NaN or infinite values."""
    x                = torch.randn(1, 63)
    embedding, recon = model(x)

    assert not torch.any(torch.isnan(embedding)), \
        "NaN in embedding"
    assert not torch.any(torch.isinf(embedding)), \
        "Inf in embedding"
    assert not torch.any(torch.isnan(recon)), \
        "NaN in reconstruction"


def test_training_reduces_loss(trainer, sample_features):
    """
    Test that training actually reduces loss.
    
    WHY: This verifies the training loop is working correctly.
    If loss doesn't decrease, something is wrong with
    gradients, optimizer, or loss function.
    """
    # Get initial loss
    initial_loss = trainer.train_epoch(sample_features)

    # Train for several epochs
    for _ in range(20):
        final_loss = trainer.train_epoch(sample_features)

    assert final_loss < initial_loss, \
        f"Training didn't reduce loss: {initial_loss:.4f} → {final_loss:.4f}"


def test_model_save_load(model, tmp_path):
    """
    Test model can be saved and loaded correctly.
    
    WHY: If save/load is broken, trained models are lost.
    We use tmp_path (pytest built-in) for temporary files.
    """
    save_path = str(tmp_path / "test_model.pt")

    # Get embedding before saving
    model.eval()
    x = torch.randn(1, 63)
    with torch.no_grad():
        embedding_before = model.encode(x)

    # Save
    torch.save(model.state_dict(), save_path)
    assert os.path.exists(save_path), "Model file not created"

    # Load into new model
    new_model = DreamEncoder(input_dim=63, embedding_dim=16)
    new_model.load_state_dict(torch.load(save_path))
    new_model.eval()

    # Compare embeddings
    with torch.no_grad():
        embedding_after = new_model.encode(x)

    assert torch.allclose(embedding_before, embedding_after), \
        "Loaded model produces different embeddings"


def test_different_inputs_different_embeddings(model):
    """
    Test that different inputs produce different embeddings.
    
    WHY: If all inputs produce the same embedding,
    the encoder learned a trivial solution (mode collapse).
    This is a common failure mode in autoencoders.
    """
    model.eval()
    with torch.no_grad():
        x1 = torch.randn(1, 63)
        x2 = torch.randn(1, 63)
        e1 = model.encode(x1)
        e2 = model.encode(x2)

    assert not torch.allclose(e1, e2), \
        "Different inputs produced identical embeddings (mode collapse)"