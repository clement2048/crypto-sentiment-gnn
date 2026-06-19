"""Consistency checks for judge outputs."""

from __future__ import annotations

from config import PROBABILITY_MAX, PROBABILITY_MIN
from judge.judge_schema import JudgeOutput


def check_judge_consistency(output: JudgeOutput) -> list[str]:
    flags: list[str] = []
    scores = output.score_vector
    if not PROBABILITY_MIN <= output.confidence <= PROBABILITY_MAX:
        flags.append("confidence_out_of_range")

    for name, value in scores.to_dict().items():
        if not PROBABILITY_MIN <= value <= PROBABILITY_MAX:
            flags.append(f"{name}_out_of_range")

    if output.verdict == "BULLISH" and scores.p_bull < scores.p_bear:
        flags.append("verdict_score_direction_mismatch")
    if output.verdict == "BEARISH" and scores.p_bear < scores.p_bull:
        flags.append("verdict_score_direction_mismatch")
    return flags



