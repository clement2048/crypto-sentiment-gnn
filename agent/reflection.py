"""Reflection signals for the v3 debater-reflection loop.

The reflection signal is derived from judge reports and model confidence only.
It must not include ground-truth labels, p1, or future prices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import (
    REFLECTION_CONFIDENCE_THRESHOLD,
    REFLECTION_HIGH_CONFIDENCE_THRESHOLD,
    REFLECTION_MIN_WEAK_DIMS,
)


@dataclass
class ReflectionSignal:
    verdict: str = "BULLISH"
    confidence: float = 0.0
    weak_dims: list[str] = field(default_factory=list)
    supplement_suggestions: list[str] = field(default_factory=list)
    ode_margin: float | None = None
    mean_argument_confidence: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "weak_dims": self.weak_dims,
            "supplement_suggestions": self.supplement_suggestions,
            "ode_margin": self.ode_margin,
            "mean_argument_confidence": self.mean_argument_confidence,
        }


def should_continue_reflection(signal: ReflectionSignal) -> bool:
    """Return True when the judge report indicates weak or uncertain debate."""
    if signal.confidence < REFLECTION_CONFIDENCE_THRESHOLD:
        return True
    return len(signal.weak_dims) >= REFLECTION_MIN_WEAK_DIMS


def reflection_converged(signal: ReflectionSignal) -> bool:
    """Return True when reflection has no actionable weakness left."""
    return signal.confidence >= REFLECTION_HIGH_CONFIDENCE_THRESHOLD or not signal.weak_dims
