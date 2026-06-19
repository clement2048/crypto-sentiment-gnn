"""Serializable summaries of model and ODE outputs for the judge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelOutputSummary:
    bullish_probability: float
    bull_mean: float
    bear_mean: float
    bull_max: float
    bear_max: float
    bull_bear_margin: float
    ode_steps: int
    net_score_mean: float = 0.0
    net_score_max: float = 0.0
    relation_count: int = 0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "bullish_probability": self.bullish_probability,
            "bull_mean": self.bull_mean,
            "bear_mean": self.bear_mean,
            "bull_max": self.bull_max,
            "bear_max": self.bear_max,
            "bull_bear_margin": self.bull_bear_margin,
            "ode_steps": self.ode_steps,
            "net_score_mean": self.net_score_mean,
            "net_score_max": self.net_score_max,
            "relation_count": self.relation_count,
        }

    @staticmethod
    def field_descriptions() -> dict[str, str]:
        """Human-readable semantics for judge prompts and exported records."""
        return {
            "bullish_probability": (
                "Calibrated graph-model probability for BULLISH. It is a learned model output, "
                "not a ground-truth label and not a final verdict."
            ),
            "bull_mean": (
                "Mean terminal value of the internal bull-channel node states after ODE evolution. "
                "This is an uncalibrated latent statistic."
            ),
            "bear_mean": (
                "Mean terminal value of the internal bear-channel node states after ODE evolution. "
                "This is an uncalibrated latent statistic."
            ),
            "bull_max": (
                "Maximum terminal value among bull-channel node states. It indicates the strongest "
                "bull-channel activation, not a final direction label."
            ),
            "bear_max": (
                "Maximum terminal value among bear-channel node states. It indicates the strongest "
                "bear-channel activation, not a final direction label."
            ),
            "bull_bear_margin": (
                "bull_mean minus bear_mean. Positive means the internal bull channel is larger on "
                "average; negative means the bear channel is larger. This is not calibrated."
            ),
            "ode_steps": "Number of sampled ODE integration steps used by the model.",
            "net_score_mean": (
                "Mean of sigmoid(bull_state) minus sigmoid(bear_state) over nodes. Positive means "
                "the internal bull channel is above the bear channel on average. It is an "
                "uncalibrated diagnostic and may conflict with bullish_probability."
            ),
            "net_score_max": (
                "Largest absolute per-node bull-vs-bear sigmoid difference. It measures internal "
                "channel separation strength, not final confidence."
            ),
            "relation_count": "Number of graph relation types supplied to the ODE model.",
        }

    @staticmethod
    def interpretation_notes() -> list[str]:
        """Rules that prevent the judge from over-reading internal model diagnostics."""
        return [
            "Do not treat any model_summary field as a ground-truth label or future market information.",
            "Do not convert bullish_probability into a hard direction unless the value is far from 0.5.",
            "If bullish_probability is close to 0.5, treat the model's calibrated direction signal as weak.",
            "bull_mean, bear_mean, bull_bear_margin, net_score_mean, and net_score_max are internal ODE diagnostics, not calibrated probabilities.",
            "When model diagnostics conflict with the debate graph, explain the conflict and rely more on argument logic and evidence quality.",
            "The final verdict must be based on both debate evidence and model diagnostics, not on a single numeric field.",
        ]

    def explained_dict(self) -> dict[str, Any]:
        """Return values with descriptions for LLM-facing judge input."""
        values = self.to_dict()
        descriptions = self.field_descriptions()
        return {
            "values": values,
            "field_descriptions": descriptions,
            "interpretation_notes": self.interpretation_notes(),
        }



