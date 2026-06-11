"""Archived deterministic mock pipeline.

This file keeps the old offline mock debate and judge implementations for
historical reference only. The active project code no longer imports it.
"""

from __future__ import annotations

from statistics import mean

from agent.prompts import normalize_role
from agent.schema import Argument, Camp, DebateTranscript, Evidence
from config import PROBABILITY_MAX, PROBABILITY_MIN
from data.schema import CommentBlock
from judge.consistency import check_judge_consistency
from judge.judge_schema import JudgeOutput, JudgeScoreVector
from model.model_summary import ModelOutputSummary
from profiles.user_profile import UserProfile


MOCK_DEBATE_TARGET_LIMIT = 2
MOCK_CONFIDENCE_BASE = 0.55
MOCK_CONFIDENCE_PRIOR_CAP = 4
MOCK_CONFIDENCE_PRIOR_BONUS = 0.02
MOCK_CONFIDENCE_PROFILE_BONUS = 0.08
MOCK_CONFIDENCE_MAX = 0.92
MOCK_ROOT_COMMENT_RELEVANCE = 0.8
MOCK_POST_RELEVANCE = 0.6
MOCK_PROFILE_RELEVANCE = 0.5

LEGACY_JUDGE_NEUTRAL_MARGIN = 0.05
LEGACY_JUDGE_CONFIDENCE_BASE = 0.5

MODEL_AWARE_JUDGE_MODEL_WEIGHT = 0.60
MODEL_AWARE_JUDGE_DEBATE_WEIGHT = 0.25
MODEL_AWARE_JUDGE_MARGIN_WEIGHT = 0.15
MODEL_AWARE_JUDGE_DEBATE_CENTER = 0.5
MODEL_AWARE_JUDGE_DEBATE_DIFF_DIVISOR = 1.0
MODEL_AWARE_JUDGE_MARGIN_BULL_VALUE = 1.0
MODEL_AWARE_JUDGE_MARGIN_BEAR_VALUE = 0.0
MODEL_AWARE_JUDGE_NEUTRAL_MARGIN = 0.08
MODEL_AWARE_JUDGE_CONFIDENCE_BASE = 0.5
MODEL_AWARE_JUDGE_CONFIDENCE_DIFF_DIVISOR = 2.0
JUDGE_EVIDENCE_QUALITY_SCALE = 3.0
JUDGE_ROLE_COVERAGE_SCALE = 4.0
DIVISION_EPSILON = 1e-6


class ArchivedMockDebateClient:
    """Old deterministic debate client, kept out of the active pipeline."""

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
        profile = profiles.get(root.author)
        profile_bias = profile.stance_bias if profile else 0.0
        product = block.product or "asset"
        confidence = _confidence_from_context(camp, profile_bias, len(prior_arguments))
        if available_target_ids is None:
            available_target_ids = [item.argument_id for item in prior_arguments if item.camp != camp]
        target_ids = available_target_ids[-MOCK_DEBATE_TARGET_LIMIT:] if MOCK_DEBATE_TARGET_LIMIT > 0 else []

        evidence = [
            Evidence("root_comment", root.comment_id, root.text[:120], MOCK_ROOT_COMMENT_RELEVANCE),
            Evidence("post", block.post_id, block.post_content[:120], MOCK_POST_RELEVANCE),
        ]
        if profile is not None:
            evidence.append(
                Evidence(
                    "profile",
                    root.author,
                    f"history_count={profile.history_count}, stance_bias={profile.stance_bias:.2f}",
                    MOCK_PROFILE_RELEVANCE,
                )
            )

        return Argument(
            argument_id=f"{block.block_id}:r{round_index}:s{seq}:{camp}",
            agent_id=f"{camp}_{role}",
            camp=camp,
            role=role,
            claim=_phase_prefix(phase)
            + _claim_for_role(camp, role, product, root.text, profile_bias, bool(target_ids)),
            evidence=evidence,
            confidence=confidence,
            targets=target_ids,
            cited_comment_ids=[root.comment_id],
            round=round_index,
            seq=seq,
            phase=phase,
        )


class ArchivedMockJudgeAgent:
    """Old debate-only rule judge."""

    def judge(self, transcript: DebateTranscript) -> JudgeOutput:
        bull_args = [item for item in transcript.arguments if item.camp == "bull"]
        bear_args = [item for item in transcript.arguments if item.camp == "bear"]
        bull_conf = mean([item.confidence for item in bull_args]) if bull_args else 0.0
        bear_conf = mean([item.confidence for item in bear_args]) if bear_args else 0.0
        total = max(len(transcript.arguments), 1)
        p_bull = _clamp01((len(bull_args) / total + bull_conf) / 2)
        p_bear = _clamp01((len(bear_args) / total + bear_conf) / 2)
        verdict = _verdict_from_scores(p_bull, p_bear, LEGACY_JUDGE_NEUTRAL_MARGIN)
        confidence = _clamp01(LEGACY_JUDGE_CONFIDENCE_BASE + abs(p_bull - p_bear))
        output = JudgeOutput(
            verdict=verdict,
            confidence=confidence,
            report=(
                f"Archived mock judge reviewed {len(transcript.arguments)} arguments for "
                f"{transcript.block_id}. Bull score={p_bull:.2f}, bear score={p_bear:.2f}; "
                f"verdict={verdict}."
            ),
            score_vector=JudgeScoreVector(
                p_bull=p_bull,
                p_bear=p_bear,
                q_bull=_evidence_quality(bull_args),
                q_bear=_evidence_quality(bear_args),
                e_bull=bull_conf,
                e_bear=bear_conf,
                c=_coverage_score(transcript),
                d=_dispute_score(transcript),
                a=_attack_score(transcript),
                rho=confidence,
            ),
            consistency_flags=[],
        )
        output.consistency_flags = check_judge_consistency(output)
        return output


class ArchivedModelAwareMockJudge:
    """Old model-aware rule judge used before online judge providers were default."""

    def judge(self, transcript: DebateTranscript, model_summary: ModelOutputSummary) -> JudgeOutput:
        bull_args = [item for item in transcript.arguments if item.camp == "bull"]
        bear_args = [item for item in transcript.arguments if item.camp == "bear"]
        bull_conf = mean([item.confidence for item in bull_args]) if bull_args else 0.0
        bear_conf = mean([item.confidence for item in bear_args]) if bear_args else 0.0
        debate_bull = _clamp01(
            (MODEL_AWARE_JUDGE_DEBATE_CENTER + bull_conf - bear_conf)
            / MODEL_AWARE_JUDGE_DEBATE_DIFF_DIVISOR
        )
        model_bull = _clamp01(model_summary.bullish_probability)
        margin_bull = (
            MODEL_AWARE_JUDGE_MARGIN_BULL_VALUE
            if model_summary.bull_bear_margin >= 0
            else MODEL_AWARE_JUDGE_MARGIN_BEAR_VALUE
        )
        p_bull = _clamp01(
            MODEL_AWARE_JUDGE_MODEL_WEIGHT * model_bull
            + MODEL_AWARE_JUDGE_DEBATE_WEIGHT * debate_bull
            + MODEL_AWARE_JUDGE_MARGIN_WEIGHT * margin_bull
        )
        p_bear = 1.0 - p_bull
        verdict = _verdict_from_scores(p_bull, p_bear, MODEL_AWARE_JUDGE_NEUTRAL_MARGIN)
        confidence = _clamp01(
            MODEL_AWARE_JUDGE_CONFIDENCE_BASE
            + abs(p_bull - p_bear) / MODEL_AWARE_JUDGE_CONFIDENCE_DIFF_DIVISOR
        )
        score_vector = JudgeScoreVector(
            p_bull=p_bull,
            p_bear=p_bear,
            q_bull=_evidence_quality(bull_args),
            q_bear=_evidence_quality(bear_args),
            e_bull=_clamp01(
                model_summary.bull_mean
                / (model_summary.bull_mean + model_summary.bear_mean + DIVISION_EPSILON)
            ),
            e_bear=_clamp01(
                model_summary.bear_mean
                / (model_summary.bull_mean + model_summary.bear_mean + DIVISION_EPSILON)
            ),
            c=_coverage_score(transcript),
            d=_dispute_score(transcript),
            a=_attack_score(transcript),
            rho=confidence,
        )
        output = JudgeOutput(
            verdict=verdict,
            confidence=confidence,
            report=(
                f"Archived model-aware mock judge for {transcript.block_id}: {verdict}, "
                f"confidence={confidence:.3f}, model bullish_probability="
                f"{model_summary.bullish_probability:.3f}."
            ),
            score_vector=score_vector,
            consistency_flags=[],
        )
        output.consistency_flags = check_judge_consistency(output)
        return output


def _claim_for_role(
    camp: Camp,
    role: str,
    product: str,
    root_text: str,
    profile_bias: float,
    has_targets: bool,
) -> str:
    direction = "bullish" if camp == "bull" else "bearish"
    normalized_role = normalize_role(role)
    if normalized_role == "technical_analysis_agent":
        return f"{product} shows a possible {direction} technical reading from the discussion tone; root comment: {root_text[:48]}"
    if normalized_role == "fundamental_analysis_agent":
        return f"The post context gives {product} a possible {direction} fundamental narrative, pending stronger evidence."
    if normalized_role == "sentiment_contagion_agent":
        bias_text = "bull-biased" if profile_bias > 0 else "bear-biased" if profile_bias < 0 else "neutral-or-cold-start"
        return f"The author profile is {bias_text}; social sentiment may transmit a {direction} reading."
    if normalized_role == "risk_analysis_agent":
        return f"The bear risk view argues that uncertainty around {product} can support a {direction} reading."
    if normalized_role == "onchain_skeptic_agent":
        return f"The available text does not prove healthy on-chain accumulation for {product}, leaving a {direction} risk case."
    if normalized_role == "sentiment_reversal_agent":
        return f"Visible emotion around {product} may be fragile or crowded, supporting a {direction} reversal reading."
    if normalized_role == "reflection_agent" and has_targets:
        return f"The {direction} reflection agent answers opposing critiques while noting evidence limits."
    if normalized_role == "reflection_agent":
        return f"The {direction} reflection agent gives a cautious thesis and names uncertainty before overclaiming."
    if has_targets:
        return f"The {direction} camp responds that opposing arguments do not rule out this direction."
    return f"The {direction} camp adds a risk view while keeping this direction as a viable explanation."


def _phase_prefix(phase: str) -> str:
    labels = {
        "initial_argument": "[initial] ",
        "intra_reflection": "[intra-reflection] ",
        "intra_response": "[intra-response] ",
        "cross_response": "[cross-response] ",
        "counter_reflection": "[counter-reflection] ",
        "counter_rebuttal": "[counter-rebuttal] ",
        "reflection_summary": "[reflection-summary] ",
    }
    return labels.get(phase, "")


def _confidence_from_context(camp: Camp, profile_bias: float, prior_count: int) -> float:
    base = MOCK_CONFIDENCE_BASE + min(prior_count, MOCK_CONFIDENCE_PRIOR_CAP) * MOCK_CONFIDENCE_PRIOR_BONUS
    if camp == "bull" and profile_bias > 0:
        base += MOCK_CONFIDENCE_PROFILE_BONUS
    if camp == "bear" and profile_bias < 0:
        base += MOCK_CONFIDENCE_PROFILE_BONUS
    return _clamp01(min(MOCK_CONFIDENCE_MAX, base))


def _verdict_from_scores(p_bull: float, p_bear: float, neutral_margin: float) -> str:
    if abs(p_bull - p_bear) < neutral_margin:
        return "NEUTRAL"
    if p_bull > p_bear:
        return "BULLISH"
    return "BEARISH"


def _evidence_quality(arguments: list[Argument]) -> float:
    if not arguments:
        return 0.0
    return _clamp01(mean(len(item.evidence) for item in arguments) / JUDGE_EVIDENCE_QUALITY_SCALE)


def _coverage_score(transcript: DebateTranscript) -> float:
    return _clamp01(len({item.role for item in transcript.arguments}) / JUDGE_ROLE_COVERAGE_SCALE)


def _dispute_score(transcript: DebateTranscript) -> float:
    return _clamp01(sum(1 for item in transcript.arguments if item.targets) / max(len(transcript.arguments), 1))


def _attack_score(transcript: DebateTranscript) -> float:
    by_id = {item.argument_id: item for item in transcript.arguments}
    cross_targets = 0
    for argument in transcript.arguments:
        for target_id in argument.targets:
            target = by_id.get(target_id)
            if target is not None and target.camp != argument.camp:
                cross_targets += 1
    return _clamp01(cross_targets / max(len(transcript.arguments), 1))


def _clamp01(value: float) -> float:
    return max(PROBABILITY_MIN, min(PROBABILITY_MAX, float(value)))
