"""Dual bullish/bearish initial state encoder."""

from __future__ import annotations

import torch
from torch import nn

from config import MODEL_HIDDEN_DIM
from model.bdg_ode.polarity_seed import PolaritySeed


class DualEmotionEncoder(nn.Module):
    """Encode node features into v3 bull/bear initial states."""

    def __init__(self, input_dim: int, hidden_dim: int = MODEL_HIDDEN_DIM):
        super().__init__()
        self.seed = PolaritySeed(input_dim)
        self.bull = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.bear = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        bull, bear, _seed = self.forward_with_seed(x)
        return bull, bear

    def forward_with_seed(self, x):
        """Return bull/bear initial states plus the scalar polarity seed s_i."""
        seed = self.seed(x)
        bull_seed = seed.clamp_min(0.0)
        bear_seed = (-seed).clamp_min(0.0)
        return (
            torch.tanh(self.bull(torch.cat([x, bull_seed], dim=-1))),
            torch.tanh(self.bear(torch.cat([x, bear_seed], dim=-1))),
            seed,
        )



