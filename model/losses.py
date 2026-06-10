"""Prototype loss helpers."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def classification_loss(prob: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(prob.view_as(label), label)



