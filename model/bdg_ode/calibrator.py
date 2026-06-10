"""Trainable verdict calibrator."""

from __future__ import annotations

import torch
from torch import nn

from config import CALIBRATOR_HIDDEN_DIM


class VerdictCalibrator(nn.Module):
    """Map graph representation to bullish probability."""

    def __init__(self, input_dim: int, hidden_dim: int = CALIBRATOR_HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph_repr: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(graph_repr)).view(1)



