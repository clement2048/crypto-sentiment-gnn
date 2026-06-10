"""Dual bullish/bearish initial state encoder."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from config import MODEL_HIDDEN_DIM


class DualEmotionEncoder(nn.Module):
    """Encode node features into non-negative bull and bear states."""

    def __init__(self, input_dim: int, hidden_dim: int = MODEL_HIDDEN_DIM):
        super().__init__()
        self.bull = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.bear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return F.softplus(self.bull(x)), F.softplus(self.bear(x))



