"""Offline judge agent for structured debate transcripts."""

from __future__ import annotations

from statistics import mean

from agent.schema import DebateTranscript
from config import (
    JUDGE_EVIDENCE_QUALITY_SCALE,
    JUDGE_ROLE_COVERAGE_SCALE,
    LEGACY_JUDGE_CONFIDENCE_BASE,
    LEGACY_JUDGE_NEUTRAL_MARGIN,
    PROBABILITY_MAX,
    PROBABILITY_MIN,
)
from judge.consistency import check_judge_consistency
from judge.judge_schema import JudgeOutput, JudgeScoreVector


class MockJudgeAgent:
    """Deterministic judge that summarizes debate statistics."""

    def judge(self, transcript: DebateTranscript) -> JudgeOutput:
        bull_args = [item for item in transcript.arguments if item.camp == "bull"]
        bear_args = [item for item in transcript.arguments if item.camp == "bear"]
        bull_conf = mean([item.confidence for item in bull_args]) if bull_args else 0.0
        bear_conf = mean([item.confidence for item in bear_args]) if bear_args else 0.0
        total = max(len(transcript.arguments), 1)
        p_bull = _clamp01((len(bull_args) / total + bull_conf) / 2)
        p_bear = _clamp01((len(bear_args) / total + bear_conf) / 2)

        if abs(p_bull - p_bear) < LEGACY_JUDGE_NEUTRAL_MARGIN:
            verdict = "NEUTRAL"
        elif p_bull > p_bear:
            verdict = "BULLISH"
        else:
            verdict = "BEARISH"

        confidence = _clamp01(LEGACY_JUDGE_CONFIDENCE_BASE + abs(p_bull - p_bear))
        score_vector = JudgeScoreVector(
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
        )
        report = (
            f"Offline judge reviewed {len(transcript.arguments)} arguments for {transcript.block_id}. "
            f"Bull score={p_bull:.2f}, bear score={p_bear:.2f}; verdict={verdict}."
        )
        output = JudgeOutput(
            verdict=verdict,
            confidence=confidence,
            report=report,
            score_vector=score_vector,
            consistency_flags=[],
        )
        output.consistency_flags = check_judge_consistency(output)
        return output


def _evidence_quality(arguments) -> float:
    if not arguments:
        return 0.0
    counts = [len(item.evidence) for item in arguments]
    return _clamp01(mean(counts) / JUDGE_EVIDENCE_QUALITY_SCALE)


def _coverage_score(transcript: DebateTranscript) -> float:
    roles = {item.role for item in transcript.arguments}
    return _clamp01(len(roles) / JUDGE_ROLE_COVERAGE_SCALE)


def _dispute_score(transcript: DebateTranscript) -> float:
    targeted = sum(1 for item in transcript.arguments if item.targets)
    return _clamp01(targeted / max(len(transcript.arguments), 1))


def _attack_score(transcript: DebateTranscript) -> float:
    cross_targets = 0
    by_id = {item.argument_id: item for item in transcript.arguments}
    for argument in transcript.arguments:
        for target in argument.targets:
            target_arg = by_id.get(target)
            if target_arg is not None and target_arg.camp != argument.camp:
                cross_targets += 1
    return _clamp01(cross_targets / max(len(transcript.arguments), 1))


def _clamp01(value: float) -> float:
    return max(PROBABILITY_MIN, min(PROBABILITY_MAX, float(value)))



