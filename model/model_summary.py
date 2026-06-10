"""Serializable summaries of model and ODE outputs for the judge."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelOutputSummary:
    bullish_probability: float
    predicted_label: int
    bull_mean: float
    bear_mean: float
    bull_max: float
    bear_max: float
    bull_bear_margin: float
    ode_steps: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "bullish_probability": self.bullish_probability,
            "predicted_label": self.predicted_label,
            "bull_mean": self.bull_mean,
            "bear_mean": self.bear_mean,
            "bull_max": self.bull_max,
            "bear_max": self.bear_max,
            "bull_bear_margin": self.bull_bear_margin,
            "ode_steps": self.ode_steps,
        }



