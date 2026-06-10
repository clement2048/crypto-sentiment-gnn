"""Factory helpers for debate clients."""

from __future__ import annotations

from agent.bailian_client import BailianOpenAICompatibleDebateClient
from agent.deepseek_client import DeepSeekAnthropicDebateClient
from agent.llm_client import DebateClient
from agent.mock_client import MockDebateClient


def create_debate_client(mode: str = "mock") -> DebateClient:
    """根据运行模式创建辩论 client。"""
    if mode == "mock":
        return MockDebateClient()
    if mode == "deepseek":
        return DeepSeekAnthropicDebateClient()
    if mode == "bailian":
        return BailianOpenAICompatibleDebateClient()
    raise ValueError(f"Unsupported debate client mode: {mode}")
