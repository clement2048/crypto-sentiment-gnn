"""Factory helpers for judge providers."""

from __future__ import annotations

from agent.anthropic_compatible import DeepSeekJudgeClient
from agent.openai_compatible import BailianJudgeClient, SiliconFlowJudgeClient


def create_judge_client(mode: str = "siliconflow"):
    """根据运行模式创建法官 provider。"""
    if mode == "deepseek":
        return DeepSeekJudgeClient()
    if mode == "bailian":
        return BailianJudgeClient()
    if mode == "siliconflow":
        return SiliconFlowJudgeClient()
    raise ValueError(f"Unsupported judge mode: {mode}")
