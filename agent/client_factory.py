"""Factory helpers for debate clients."""

from __future__ import annotations

from agent.anthropic_compatible import DeepSeekAnthropicDebateClient
from agent.llm_client import DebateClient
from agent.openai_compatible import BailianOpenAICompatibleDebateClient, SiliconFlowOpenAICompatibleDebateClient


def create_debate_client(mode: str = "siliconflow") -> DebateClient:
    """根据运行模式创建辩论 client。"""
    if mode == "deepseek":
        return DeepSeekAnthropicDebateClient()
    if mode == "bailian":
        return BailianOpenAICompatibleDebateClient()
    if mode == "siliconflow":
        return SiliconFlowOpenAICompatibleDebateClient()
    raise ValueError(f"Unsupported debate client mode: {mode}")
