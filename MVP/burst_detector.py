"""
burst_detector.py — OSKAR v0.4 Multimodal Intelligence
--------------------------------------------------------
Detects Temporal Coordination Attacks — coordinated posting bursts
where multiple accounts suddenly post similar content at nearly the
same time, indicating an orchestrated narrative attack.

Architecture: LSTM Autoencoder
- Input:  A sliding window of (timestamp, user_count) pairs
- Output: Reconstruction error → anomaly_score (0.0 – 1.0)
  - 0.0 = Normal organic posting patterns
  - 1.0 = Extreme burst anomaly (likely coordinated attack)

The model is lightweight (unsupervised) and trains on the historical
posting-rate time series for each user/topic fingerprint. No labels
are required — anomalies are detected as high reconstruction errors.

Usage:
    from burst_detector import BurstDetector
    bd = BurstDetector()

    # Feed a rolling window of posting rates (posts per minute)
    # Each entry: (unix_timestamp, n_unique_users_posting)
    events = [
        {"ts": 1708000000, "user_count": 3},
        {"ts": 1708000060, "user_count": 5},
        {"ts": 1708000120, "user_count": 47},   # ← Suspicious spike
        {"ts": 1708000180, "user_count": 52},
        {"ts": 1708000240, "user_count": 49},
    ]
    result = bd.detect(events)
    # result = {"anomaly_score": 0.87, "is_burst": True, "window_size": 5}
"""

import os
import math
import time
from typing import Optional

import torch
import torch.nn as nn
import numpy as np

SEQ_LEN    = 16   # Sliding window length (time steps considered)
INPUT_DIM  = 2    # Features: [norm_timestamp_delta, norm_user_count]
HIDDEN_DIM = 32   # LSTM hidden state size
LATENT_DIM = 8    # Bottleneck size
THRESHOLD  = 0.25 # Reconstruction MSE above which we flag a burst


# ─── LSTM Autoencoder Architecture ──────────────────────────────────────────

class _LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))


class _LSTMDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, input_dim, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.fc      = nn.Linear(latent_dim, hidden_dim)
        self.lstm    = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.out     = nn.Linear(hidden_dim, input_dim)

    def forward(self, z):
        h0 = self.fc(z).unsqueeze(0)       # (1, batch, hidden)
        # Repeat latent vector across the time dimension
        inp = h0.permute(1, 0, 2).expand(-1, self.seq_len, -1)  # (batch, seq, hidden)
        out, _ = self.lstm(inp, (h0, torch.zeros_like(h0)))
        return self.out(out)


class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                 latent_dim=LATENT_DIM, seq_len=SEQ_LEN):
        super().__init__()
        self.encoder = _LSTMEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = _LSTMDecoder(latent_dim, hidden_dim, input_dim, seq_len)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# ─── BurstDetector Public Interface ─────────────────────────────────────────

class BurstDetector:
    """
    Unsupervised temporal burst anomaly detector using LSTM Autoencoder.

    Initialization:
      - Loads a pre-trained model from 'models/burst_detector.pt' if it exists
      - Otherwise initialises random weights (untrained baseline)
        → untrained model gives low anomaly scores for everything.
        → Feed real training data via burst_detector.py --train to get drift detection.

    The detector is primarily used in OSKAR as a fast anomaly signal fed into
    risk_fusion as an additional multiplier alongside the GNN bot score.
    """

    def __init__(self, model_path: str = "models/burst_detector.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = LSTMAutoEncoder().to(self.device)
        self.model.eval()
        self.trained = False
        self._try_load(model_path)

    def _try_load(self, path: str):
        if os.path.exists(path):
            try:
                state = torch.load(path, map_location=self.device)
                self.model.load_state_dict(state)
                self.trained = True
                print(f"[BurstDetector] Loaded weights from {path}")
            except Exception as e:
                print(f"[BurstDetector] Could not load model: {e}. Using untrained baseline.")
        else:
            print(f"[BurstDetector] No model at '{path}'. Using untrained baseline.")

    def _preprocess(self, events: list[dict]) -> Optional[torch.Tensor]:
        """
        Convert raw event list into normalized (SEQ_LEN, INPUT_DIM) tensor.
        Pads with zeros if fewer than SEQ_LEN events are provided.
        """
        if not events:
            return None

        # Extract features: [timestamp_delta, user_count]
        timestamps   = [e.get("ts",         0.0) for e in events]
        user_counts  = [e.get("user_count",  1)   for e in events]

        # Compute time deltas between consecutive posts (first delta = 0)
        deltas = [0.0] + [
            float(timestamps[i] - timestamps[i-1])
            for i in range(1, len(timestamps))
        ]

        # Normalize
        max_delta = max(deltas) if max(deltas) > 0 else 1.0
        max_count = max(user_counts) if max(user_counts) > 0 else 1.0
        features = [
            [d / max_delta, c / max_count]
            for d, c in zip(deltas, user_counts)
        ]

        # Pad or truncate to SEQ_LEN
        if len(features) < SEQ_LEN:
            pad = [[0.0, 0.0]] * (SEQ_LEN - len(features))
            features = pad + features   # pre-pad with zeros
        else:
            features = features[-SEQ_LEN:]  # use most recent window

        return torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def detect(self, events: list[dict]) -> dict:
        """
        Detect temporal burst anomalies in a posting event stream.

        Args:
            events: List of dicts with keys:
                - "ts":         Unix timestamp (int or float)
                - "user_count": Number of unique users posting in this interval (int)

        Returns:
            {
                "anomaly_score":   float,   # 0.0 (normal) → 1.0 (extreme burst)
                "is_burst":        bool,    # True if anomaly_score > THRESHOLD
                "window_size":     int,     # Number of events used
                "trained_model":   bool     # Whether a trained model was loaded
            }
        """
        if not events:
            return {
                "anomaly_score": 0.0,
                "is_burst":      False,
                "window_size":   0,
                "trained_model": self.trained
            }

        x = self._preprocess(events)
        if x is None:
            return {
                "anomaly_score": 0.0,
                "is_burst":      False,
                "window_size":   0,
                "trained_model": self.trained
            }

        # Forward pass through autoencoder
        recon = self.model(x)
        mse = torch.mean((x - recon) ** 2).item()

        # Sigmoid-normalize reconstruction error to [0, 1]
        # High MSE (burst) → score close to 1.0
        anomaly_score = float(1.0 - math.exp(-mse * 10))
        anomaly_score = round(min(1.0, max(0.0, anomaly_score)), 4)

        return {
            "anomaly_score": anomaly_score,
            "is_burst":      anomaly_score > THRESHOLD,
            "window_size":   min(len(events), SEQ_LEN),
            "trained_model": self.trained
        }

    def quick_score(self, events: list[dict]) -> float:
        """Convenience method — returns just the anomaly_score float."""
        return self.detect(events)["anomaly_score"]
