"""Reflection signal objects for the Judge-guided debate loop.

Reflection is the feedback bridge from Judge back to debaters:

1. Judge reads the debate transcript, graph, and model summary.
2. Judge outputs a structured report with confidence, weak dimensions, and
   supplement suggestions.
3. `reflection_signal_from_judge(...)` in `judge/report_parser.py` converts
   that Judge output into `ReflectionSignal`.
4. `should_continue_reflection(...)` decides whether the current debate needs
   more argument supplements.
5. `DebateOrchestrator.add_reflection_rounds(...)` uses the signal to append
   bull/bear supplement arguments.

The signal is deliberately limited. It can describe weak reasoning dimensions,
low confidence, or model/Judge uncertainty, but it must not include
ground-truth labels, p1, or future prices.
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
    """Safe, LLM-facing feedback distilled from Judge output.

    Fields:
    - `verdict`: previous Judge verdict. It is useful for bookkeeping, but
      debaters should not treat it as ground truth.
    - `confidence`: Judge confidence in that verdict.
    - `weak_dims`: dimensions that need stronger evidence or rebuttal.
    - `supplement_suggestions`: natural-language repair hints.
    - `ode_margin`: optional graph-model margin used as an uncertainty signal.
    - `mean_argument_confidence`: optional debate-strength signal.
    """

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
    """Return True when the debate should receive another supplement round."""
    # Low Judge confidence means the current transcript/model evidence is not
    # decisive enough, so the next bull/bear pair may add useful information.
    if signal.confidence < REFLECTION_CONFIDENCE_THRESHOLD:
        return True
    # Even with acceptable confidence, many weak dimensions indicate the report
    # found multiple evidence or reasoning gaps worth repairing.
    return len(signal.weak_dims) >= REFLECTION_MIN_WEAK_DIMS


def reflection_converged(signal: ReflectionSignal) -> bool:
    """Return True when reflection has no actionable weakness left."""
    return signal.confidence >= REFLECTION_HIGH_CONFIDENCE_THRESHOLD or not signal.weak_dims
