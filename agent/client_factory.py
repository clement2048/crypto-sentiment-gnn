"""Factory helpers for debate clients."""

from __future__ import annotations

from agent.llm_client import DebateClient
from agent.openai_compatible import SiliconFlowOpenAICompatibleDebateClient


def create_debate_client(mode: str = "siliconflow") -> DebateClient:
    """Create the debate client used by current paper experiments."""
    if mode == "siliconflow":
        return SiliconFlowOpenAICompatibleDebateClient()
    raise ValueError(f"Unsupported debate client mode: {mode}")
