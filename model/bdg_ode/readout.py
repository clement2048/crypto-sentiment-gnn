"""Graph-level readout for dual emotion states."""

from __future__ import annotations

import torch
from torch import nn


class DualReadout(nn.Module):
    """Pool bull/bear node states into one graph representation."""

    def forward(self, bull: torch.Tensor, bear: torch.Tensor) -> torch.Tensor:
        bull_mean = bull.mean(dim=0)
        bear_mean = bear.mean(dim=0)
        bull_max = bull.max(dim=0).values
        bear_max = bear.max(dim=0).values
        contrast = bull_mean - bear_mean
        return torch.cat([bull_mean, bear_mean, bull_max, bear_max, contrast], dim=0)



