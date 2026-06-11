"""Offline debate-agent prototype for v2."""

from agent.anthropic_compatible import (
    DeepSeekAnthropicDebateClient,
    DeepSeekJudgeClient,
    MiniMaxAnthropicDebateClient,
    MiniMaxJudgeClient,
)
from agent.client_factory import create_debate_client
from agent.debate_orchestrator import DebateOrchestrator
from agent.openai_compatible import BailianJudgeClient, BailianOpenAICompatibleDebateClient
from agent.schema import Argument, DebateTranscript, Evidence

__all__ = [
    "Argument",
    "BailianJudgeClient",
    "BailianOpenAICompatibleDebateClient",
    "DebateOrchestrator",
    "DebateTranscript",
    "DeepSeekAnthropicDebateClient",
    "DeepSeekJudgeClient",
    "Evidence",
    "MiniMaxAnthropicDebateClient",
    "MiniMaxJudgeClient",
    "create_debate_client",
]



