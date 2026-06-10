"""Temporal train/validation/test split for CommentBlock samples."""

from __future__ import annotations

from dataclasses import dataclass

from config import SPLIT_RATIO_EPSILON, TEST_RATIO, TRAIN_RATIO, VAL_RATIO
from data.schema import CommentBlock


@dataclass
class SplitResult:
    train: list[CommentBlock]
    val: list[CommentBlock]
    test: list[CommentBlock]

    @property
    def all(self) -> list[CommentBlock]:
        return [*self.train, *self.val, *self.test]


def temporal_split_blocks(
    blocks: list[CommentBlock],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> SplitResult:
    """Sort by t0, then split without shuffling."""
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > SPLIT_RATIO_EPSILON:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    sorted_blocks = sorted(blocks, key=lambda item: item.t0)
    n_items = len(sorted_blocks)
    train_end = int(n_items * train_ratio)
    val_end = train_end + int(n_items * val_ratio)
    return SplitResult(
        train=sorted_blocks[:train_end],
        val=sorted_blocks[train_end:val_end],
        test=sorted_blocks[val_end:],
    )



