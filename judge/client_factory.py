"""Factory helpers for judge providers."""

from __future__ import annotations

from agent.schema import DebateTranscript
from debate_graph.schema import HeteroGraph
from judge.bailian_judge_client import BailianJudgeClient
from judge.deepseek_judge_client import DeepSeekJudgeClient
from judge.judge_schema import JudgeOutput
from judge.model_aware_judge import ModelAwareMockJudge
from model.model_summary import ModelOutputSummary


class ModelAwareMockJudgeClient:
    """Adapter that shares the same call shape as online judge providers."""

    def __init__(self):
        self.inner = ModelAwareMockJudge()

    def judge(
        self,
        transcript: DebateTranscript,
        model_summary: ModelOutputSummary,
        graph: HeteroGraph,
    ) -> JudgeOutput:
        return self.inner.judge(transcript, model_summary)


def create_judge_client(mode: str = "mock"):
    """根据运行模式创建法官 provider。"""
    if mode == "mock":
        return ModelAwareMockJudgeClient()
    if mode == "deepseek":
        return DeepSeekJudgeClient()
    if mode == "bailian":
        return BailianJudgeClient()
    raise ValueError(f"Unsupported judge mode: {mode}")
