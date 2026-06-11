"""Factory helpers for debate clients."""

from __future__ import annotations

from agent.anthropic_compatible import DeepSeekAnthropicDebateClient, MiniMaxAnthropicDebateClient
from agent.llm_client import DebateClient
from agent.mock_client import MockDebateClient
from agent.openai_compatible import BailianOpenAICompatibleDebateClient


def create_debate_client(mode: str = "mock") -> DebateClient:
    """根据运行模式创建辩论 client。"""
    if mode == "mock":
        return MockDebateClient()
    if mode == "deepseek":
        return DeepSeekAnthropicDebateClient()
    if mode == "bailian":
        return BailianOpenAICompatibleDebateClient()
    if mode == "minimax":
        return MiniMaxAnthropicDebateClient()
    raise ValueError(f"Unsupported debate client mode: {mode}")
