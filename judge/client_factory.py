"""Factory helpers for judge providers."""

from __future__ import annotations

from agent.openai_compatible import SiliconFlowJudgeClient


def create_judge_client(mode: str = "siliconflow"):
    """Create the judge client used by current paper experiments."""
    if mode == "siliconflow":
        return SiliconFlowJudgeClient()
    raise ValueError(f"Unsupported judge mode: {mode}")
