"""Debate-agent providers and orchestration for v2."""

from agent.client_factory import create_debate_client
from agent.debate_orchestrator import DebateOrchestrator
from agent.openai_compatible import (
    SiliconFlowJudgeClient,
    SiliconFlowOpenAICompatibleDebateClient,
)
from agent.schema import Argument, DebateTranscript, Evidence

__all__ = [
    "Argument",
    "DebateOrchestrator",
    "DebateTranscript",
    "Evidence",
    "SiliconFlowJudgeClient",
    "SiliconFlowOpenAICompatibleDebateClient",
    "create_debate_client",
]
