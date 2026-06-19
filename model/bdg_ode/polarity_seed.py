"""Learnable initial polarity seed for v3 Bi-ODE."""

from __future__ import annotations

import torch
from torch import nn


class PolaritySeed(nn.Module):
    """Map node features to s_i in [-1, 1].

    s_i is only an initialization signal for the two ODE channels, not the final
    prediction label.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(x))
