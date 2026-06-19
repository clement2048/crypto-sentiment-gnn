from __future__ import annotations

from agent.prompts import normalize_role
from agent.schema import Argument, Camp, DebateTranscript, Evidence
from data.schema import CommentBlock
from judge.judge_schema import JudgeOutput, JudgeScoreVector
from model.model_summary import ModelOutputSummary
from profiles.user_profile import UserProfile


class FakeDebateClient:
    """Deterministic test double for DebateClient."""

    def generate_argument(
        self,
        block: CommentBlock,
        profiles: dict[str, UserProfile],
        camp: Camp,
        role: str,
        round_index: int,
        seq: int,
        prior_arguments: list[Argument],
        phase: str = "initial_argument",
        available_target_ids: list[str] | None = None,
    ) -> Argument:
        root = block.root_comment
        targets = list(available_target_ids or [])[-2:]
        normalized_role = normalize_role(role)
        direction = "bullish" if camp == "bull" else "bearish"
        return Argument(
            argument_id=f"{block.block_id}:r{round_index}:s{seq}:{camp}",
            agent_id=f"{camp}_{normalized_role}",
            camp=camp,
            role=normalized_role,
            claim=f"[{phase}] {direction} test argument for {block.product or 'asset'}: {root.text[:48]}",
            evidence=[
                Evidence(
                    source=f"comment:{root.comment_id}",
                    quote=root.text[:120],
                    relevance=0.8,
                    source_type="root_comment",
                    source_id=root.comment_id,
                )
            ],
            confidence=0.62 if camp == "bear" else 0.58,
            target_args=targets,
            cited_comment_ids=[],
            round=round_index,
            seq=seq,
            phase=phase,
            t_index=float(round_index - 1) + float(seq - 1) / 2.0,
        )


class FakeJudgeClient:
    """Deterministic test double for online judge clients."""

    def judge(
        self,
        transcript: DebateTranscript,
        model_summary: ModelOutputSummary,
        graph,
    ) -> JudgeOutput:
        verdict = "BULLISH" if model_summary.bullish_probability >= 0.5 else "BEARISH"
        p_bull = 0.7 if verdict == "BULLISH" else 0.3
        p_bear = 1.0 - p_bull
        return JudgeOutput(
            verdict=verdict,
            confidence=0.7,
            report=(
                f"Fake judge decision for {transcript.block_id}: {verdict}. "
                f"ODE bullish_probability={model_summary.bullish_probability:.3f}."
            ),
            score_vector=JudgeScoreVector(
                p_bull=p_bull,
                p_bear=p_bear,
                q_bull=0.5,
                q_bear=0.5,
                e_bull=0.5,
                e_bear=0.5,
                c=0.5,
                d=0.5,
                a=0.5,
                rho=0.7,
            ),
            consistency_flags=[],
        )
