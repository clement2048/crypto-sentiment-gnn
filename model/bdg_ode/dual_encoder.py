"""Encode graph node features into initial bull/bear ODE states.

Input from graph tensorization:
`graph_tensor.x` has one row per graph node. Each row contains structural node
features and, optionally, appended text embeddings.

Encoder flow:
1. `PolaritySeed` maps each node feature vector to scalar seed `s_i` in [-1, 1].
2. Positive part `max(s_i, 0)` becomes a bull-channel initialization cue.
3. Negative part `max(-s_i, 0)` becomes a bear-channel initialization cue.
4. The original node features are concatenated with the relevant seed cue.
5. Two independent MLP branches produce initial hidden states `bull0` and
   `bear0` for the Bi-ODE dynamics.

These hidden states are not final predictions. They are just the initial
condition for continuous graph evolution.
"""

from __future__ import annotations

import torch
from torch import nn

from config import MODEL_HIDDEN_DIM
from model.bdg_ode.polarity_seed import PolaritySeed


class DualEmotionEncoder(nn.Module):
    """Encode node features into initial states for the two emotion channels."""

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
        """Return only bull/bear initial states for inference paths."""
        bull, bear, _seed = self.forward_with_seed(x)
        return bull, bear

    def forward_with_seed(self, x):
        """Return bull/bear initial states plus the scalar polarity seed.

        `seed` is kept because auxiliary losses can use it to align initial
        bull/bear strengths with the learned polarity cue.
        """
        seed = self.seed(x)
        bull_seed = seed.clamp_min(0.0)
        bear_seed = (-seed).clamp_min(0.0)
        return (
            torch.tanh(self.bull(torch.cat([x, bull_seed], dim=-1))),
            torch.tanh(self.bear(torch.cat([x, bear_seed], dim=-1))),
            seed,
        )


