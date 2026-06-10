"""Schemas for LLM judge outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from config import PROBABILITY_MAX, PROBABILITY_MIN

Verdict = Literal["BULLISH", "BEARISH", "NEUTRAL"]


@dataclass
class JudgeScoreVector:
    p_bull: float
    p_bear: float
    q_bull: float
    q_bear: float
    e_bull: float
    e_bear: float
    c: float
    d: float
    a: float
    rho: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JudgeScoreVector":
        return cls(
            p_bull=_clamp01(data.get("p_bull", 0.0)),
            p_bear=_clamp01(data.get("p_bear", 0.0)),
            q_bull=_clamp01(data.get("q_bull", 0.0)),
            q_bear=_clamp01(data.get("q_bear", 0.0)),
            e_bull=_clamp01(data.get("e_bull", 0.0)),
            e_bear=_clamp01(data.get("e_bear", 0.0)),
            c=_clamp01(data.get("c", 0.0)),
            d=_clamp01(data.get("d", 0.0)),
            a=_clamp01(data.get("a", 0.0)),
            rho=_clamp01(data.get("rho", 0.0)),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "p_bull": self.p_bull,
            "p_bear": self.p_bear,
            "q_bull": self.q_bull,
            "q_bear": self.q_bear,
            "e_bull": self.e_bull,
            "e_bear": self.e_bear,
            "c": self.c,
            "d": self.d,
            "a": self.a,
            "rho": self.rho,
        }


@dataclass
class JudgeOutput:
    verdict: Verdict
    confidence: float
    report: str
    score_vector: JudgeScoreVector
    consistency_flags: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JudgeOutput":
        verdict = data.get("verdict")
        if verdict not in ("BULLISH", "BEARISH", "NEUTRAL"):
            raise ValueError(f"Invalid judge verdict: {verdict}")
        score_data = data.get("score_vector")
        if not isinstance(score_data, dict):
            raise ValueError("Judge output missing score_vector")
        return cls(
            verdict=verdict,
            confidence=_clamp01(data.get("confidence", 0.0)),
            report=str(data.get("report") or ""),
            score_vector=JudgeScoreVector.from_dict(score_data),
            consistency_flags=[str(item) for item in data.get("consistency_flags", [])],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "report": self.report,
            "score_vector": self.score_vector.to_dict(),
            "consistency_flags": self.consistency_flags,
        }


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(PROBABILITY_MIN, min(PROBABILITY_MAX, numeric))



