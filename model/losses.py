"""Prototype loss helpers."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def classification_loss(prob: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(prob.view_as(label), label)


def smoothness_loss(states: list[torch.Tensor]) -> torch.Tensor:
    """L1: penalize large second-order changes in an ODE state path."""
    if len(states) < 3:
        return torch.tensor(0.0, device=states[0].device if states else None)
    total = torch.tensor(0.0, device=states[0].device)
    for prev, cur, nxt in zip(states, states[1:], states[2:]):
        total = total + torch.mean((nxt - 2 * cur + prev) ** 2)
    return total / max(len(states) - 2, 1)


def mutual_exclusion_loss(bull: torch.Tensor, bear: torch.Tensor) -> torch.Tensor:
    """L3: discourage one node from carrying strong bull and bear signals together."""
    return torch.mean((bull * bear) ** 2)


def initial_alignment_loss(initial: torch.Tensor, projected_input: torch.Tensor) -> torch.Tensor:
    """L0: keep ODE initial states close to projected input semantics."""
    return F.mse_loss(initial, projected_input)


def regression_strength_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Optional L_reg for continuous sentiment strength labels."""
    return F.mse_loss(predicted.view_as(target), target)



