"""Graph-level readout for terminal bull/bear ODE states.

Input:
- `bull`: terminal bull-channel node states `[num_nodes, hidden_dim]`.
- `bear`: terminal bear-channel node states `[num_nodes, hidden_dim]`.

Output:
One graph representation vector made by concatenating:
1. average bull state over nodes;
2. average bear state over nodes;
3. max bull activation over nodes;
4. max bear activation over nodes;
5. mean bull-bear contrast.

This converts node-level ODE evolution into one graph-level vector for the
classifier/calibrator.
"""

from __future__ import annotations

import torch
from torch import nn


class DualReadout(nn.Module):
    """Pool bull/bear node states into one graph representation."""

    def forward(self, bull: torch.Tensor, bear: torch.Tensor) -> torch.Tensor:
        # Mean pooling captures global discussion tendency; max pooling keeps
        # the strongest local argument/comment activation; contrast preserves
        # the direction of bull-vs-bear separation.
        bull_mean = bull.mean(dim=0)
        bear_mean = bear.mean(dim=0)
        bull_max = bull.max(dim=0).values
        bear_max = bear.max(dim=0).values
        contrast = bull_mean - bear_mean
        return torch.cat([bull_mean, bear_mean, bull_max, bear_max, contrast], dim=0)


