"""Learnable scalar polarity seed for Bi-ODE initialization.

This is the first trainable step after graph tensorization. It reads each node's
feature vector and produces a scalar in [-1, 1]:

- positive values indicate that the node should initially contribute more to
  the bull channel;
- negative values indicate stronger bear-channel initialization;
- values near zero leave both channels weakly initialized.

The seed is not supervised directly by labels and is not exposed to LLM agents.
It is an internal bridge from general node features to dual ODE channels.
"""

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
