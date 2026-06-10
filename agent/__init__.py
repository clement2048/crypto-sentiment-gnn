"""Offline debate-agent prototype for v2."""

from agent.client_factory import create_debate_client
from agent.bailian_client import BailianOpenAICompatibleDebateClient
from agent.deepseek_client import DeepSeekAnthropicDebateClient
from agent.debate_orchestrator import DebateOrchestrator
from agent.schema import Argument, DebateTranscript, Evidence

__all__ = [
    "Argument",
    "BailianOpenAICompatibleDebateClient",
    "DebateOrchestrator",
    "DebateTranscript",
    "DeepSeekAnthropicDebateClient",
    "Evidence",
    "create_debate_client",
]



