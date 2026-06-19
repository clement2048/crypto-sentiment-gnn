"""Parse judge reports into reflection signals.

The parser only reads judge-produced strength/weakness fields. It never reads
the sample label or future market fields.
"""

from __future__ import annotations

from agent.reflection import ReflectionSignal
from judge.judge_schema import JudgeOutput
from model.model_summary import ModelOutputSummary


def reflection_signal_from_judge(
    output: JudgeOutput,
    model_summary: ModelOutputSummary | None = None,
    mean_argument_confidence: float | None = None,
) -> ReflectionSignal:
    return ReflectionSignal(
        verdict=output.verdict,
        confidence=output.confidence,
        weak_dims=list(output.weak_dims or []),
        supplement_suggestions=list(output.supplement_suggestions or []),
        ode_margin=abs(model_summary.bull_bear_margin) if model_summary else None,
        mean_argument_confidence=mean_argument_confidence,
    )
