"""
neural_encoder.py
-----------------
PURPOSE: Compress EEG features into a dense "dream fingerprint" vector.
This is the core deep learning model of the dream-narrator pipeline.

WHY THIS MATTERS:
Raw features are high-dimensional and noisy.
The encoder learns to compress them into a small vector
that captures the essential brain state — the dream fingerprint.

ARCHITECTURE:
Input (n_features) → 64 → 32 → 16 (embedding)
Each layer compresses and abstracts the information.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os


class DreamEncoder(nn.Module):
    """
    A deep neural network that encodes EEG features
    into a compact dream embedding vector.

    WHY nn.Module:
    In PyTorch, every neural network inherits from nn.Module.
    This gives us automatic parameter tracking, GPU support,
    and all the training machinery for free.
    """

    def __init__(self, input_dim, embedding_dim=16):
        """
        Define the architecture of the network.

        Parameters:
            input_dim    : number of input features (from feature extractor)
            embedding_dim: size of the final dream fingerprint vector
        
        WHY __init__:
        This is where we DEFINE the layers.
        Think of it as drawing the blueprint of the network
        before any data flows through it.
        """
        super(DreamEncoder, self).__init__()

        self.input_dim     = input_dim
        self.embedding_dim = embedding_dim

        # ─────────────────────────────────────────
        # THE ENCODER LAYERS
        # Each Linear layer is a set of neurons
        # nn.Linear(in, out) means:
        #   - take 'in' numbers
        #   - multiply by learned weights
        #   - output 'out' numbers
        # ─────────────────────────────────────────

        self.encoder = nn.Sequential(

            # Layer 1: input → 64
            # Learns basic combinations of EEG features
            nn.Linear(input_dim, 64),
            nn.ReLU(),              # activation: kill negatives
            nn.LayerNorm(64),     # normalize: keeps training stable
            nn.Dropout(0.3),        # dropout: prevents memorization

            # Layer 2: 64 → 32
            # Learns more abstract brain state patterns
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Dropout(0.2),

            # Layer 3: 32 → embedding_dim (16)
            # Final compression: the dream fingerprint
            nn.Linear(32, embedding_dim),
            nn.Tanh()   # tanh keeps values between -1 and 1
                        # good for embeddings — bounded space
        )

        # ─────────────────────────────────────────
        # THE DECODER LAYERS
        # Mirror image of encoder
        # Used during training to verify the embedding
        # captures enough information to reconstruct input
        # ─────────────────────────────────────────

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.LayerNorm(32),

            nn.Linear(32, 64),
            nn.ReLU(),
            nn.LayerNorm(64),

            nn.Linear(64, input_dim)
            # No activation at end — we want raw reconstructed values
        )

    def encode(self, x):
        """
        Forward pass through encoder only.
        Returns the dream fingerprint.

        This is what we use at inference time —
        we only need the embedding, not the reconstruction.
        """
        return self.encoder(x)

    def decode(self, z):
        """
        Forward pass through decoder only.
        Reconstructs features from embedding.

        Used during training to measure how much
        information the embedding preserves.
        """
        return self.decoder(z)

    def forward(self, x):
        """
        Full forward pass: encode then decode.

        WHY forward():
        PyTorch calls this automatically when you do model(x).
        It defines how data flows through the network.

        During training: encode → decode → compute loss
        During inference: encode only → get embedding
        """
        embedding = self.encode(x)
        reconstruction = self.decode(embedding)
        return embedding, reconstruction


class DreamEncoderTrainer:
    """
    Handles training the DreamEncoder.

    WHY a separate trainer class:
    Keeps the model architecture (DreamEncoder) separate
    from the training logic. Clean separation of concerns.
    This is standard practice in ML engineering.
    """

    def __init__(self, model, learning_rate=0.001):
        """
        Parameters:
            model        : DreamEncoder instance
            learning_rate: how fast the model learns
                          too high = unstable training
                          too low  = very slow training
                          0.001 is a safe default
        """
        self.model = model

        # Adam optimizer — the most popular optimizer in deep learning
        # It automatically adjusts the learning rate during training
        # Much smarter than basic gradient descent
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate
        )

        # MSE Loss — Mean Squared Error
        # Measures how different reconstruction is from original
        # Lower loss = encoder is preserving more information
        self.criterion = nn.MSELoss()

        # Use GPU if available, otherwise CPU
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"Training on: {self.device}")
        self.model.to(self.device)

    def train_step(self, batch):
        """
        One training step on a batch of EEG features.

        The training loop in 4 steps — this is universal
        across ALL PyTorch training, memorize this pattern:

        1. Forward pass  → run data through model
        2. Compute loss  → measure how wrong we are
        3. Backward pass → compute gradients (blame assignment)
        4. Update weights → adjust weights to be less wrong
        """
        self.model.train()  # set model to training mode

        # Move data to device (GPU or CPU)
        batch = batch.to(self.device)

        # Step 1: Forward pass
        embedding, reconstruction = self.model(batch)

        # Step 2: Compute loss
        # How different is reconstruction from original?
        loss = self.criterion(reconstruction, batch)

        # Step 3: Backward pass
        # PyTorch automatically computes gradients
        self.optimizer.zero_grad()  # clear old gradients first
        loss.backward()             # compute new gradients

        # Step 4: Update weights
        self.optimizer.step()

        return loss.item(), embedding

    def train_epoch(self, features_list):
        """
        Train on all EEG epochs for one full pass.

        Parameters:
            features_list: list of feature vectors (one per epoch)
        """
        total_loss = 0

        for features in features_list:
            # Convert numpy array to PyTorch tensor
            # A tensor is just a PyTorch-compatible numpy array
            batch = torch.FloatTensor(features).unsqueeze(0)

            loss, embedding = self.train_step(batch)
            total_loss += loss

        avg_loss = total_loss / len(features_list)
        return avg_loss

    def save_model(self, path):
        """Save trained model to disk."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to: {path}")

    def load_model(self, path):
        """Load trained model from disk."""
        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from: {path}")


if __name__ == "__main__":

    from pipeline.data_loader import load_sleep_recording
    from pipeline.preprocessor import bandpass_filter, remove_powerline_noise
    from pipeline.feature_extractor import extract_features_from_epoch

    # ── Load and preprocess data ──
    data_dir       = os.path.join(os.path.expanduser("~"), "mne_data", "physionet-sleep-data")
    psg_path       = os.path.join(data_dir, "SC4001E0-PSG.edf")
    hypnogram_path = os.path.join(data_dir, "SC4001EC-Hypnogram.edf")

    raw, annotations = load_sleep_recording(psg_path, hypnogram_path)
    raw = remove_powerline_noise(raw)
    raw = bandpass_filter(raw)

    sfreq = raw.info['sfreq']
    data  = raw.get_data()

    # ── Extract features from multiple epochs ──
    print("Extracting features from epochs...")
    features_list = []
    epoch_duration = int(sfreq * 30)  # 30 seconds per epoch
    n_epochs = min(10, data.shape[1] // epoch_duration)  # use first 10 epochs

    for i in range(n_epochs):
        start = i * epoch_duration
        end   = start + epoch_duration
        epoch_data = data[:, start:end]
        features, names = extract_features_from_epoch(epoch_data, sfreq)
        features_list.append(features)
    
    # Normalize features to zero mean and unit variance
    features_array = np.array(features_list)
    mean = features_array.mean(axis=0)
    std  = features_array.std(axis=0) + 1e-10
    features_list = [(f - mean) / std for f in features_list]

    print(f"Extracted features from {len(features_list)} epochs")
    print(f"Feature vector size: {len(features_list[0])}")

    # ── Build and train encoder ──
    input_dim = len(features_list[0])
    model     = DreamEncoder(input_dim=input_dim, embedding_dim=16)
    trainer   = DreamEncoderTrainer(model, learning_rate=0.001)

    print(f"\nModel architecture:")
    print(model)

    # ── Train for a few epochs ──
    print("\nTraining encoder...")
    n_training_epochs = 20

    for epoch in range(n_training_epochs):
        avg_loss = trainer.train_epoch(features_list)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{n_training_epochs} | Loss: {avg_loss:.6f}")

    # ── Test encoding ──
    print("\nTesting encoder on first epoch...")
    model.eval()  # set to evaluation mode
    with torch.no_grad():  # no gradient computation needed
        test_input = torch.FloatTensor(features_list[0]).unsqueeze(0)
        embedding, _ = model(test_input)
        print(f"Input size  : {test_input.shape}")
        print(f"Embedding   : {embedding.shape}")
        print(f"Dream fingerprint: {embedding.numpy().flatten()[:8]}...")

    # ── Save model ──
    os.makedirs("models", exist_ok=True)
    trainer.save_model("models/dream_encoder.pt")

    print("\nNeural encoder complete!")