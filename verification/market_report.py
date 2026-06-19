"""Post-hoc market behavior verification.

Volume/activity is reported as intensity only. It is not used as a bullish or
bearish direction label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ActivityLevel = Literal["low", "normal", "high", "unknown"]


@dataclass
class MarketBehaviorResult:
    delta_p: float
    price_direction_label: int
    verdict_matches_price: bool
    delta_v: float | None
    activity_level: ActivityLevel

    def to_dict(self) -> dict[str, float | int | bool | str | None]:
        return {
            "delta_p": self.delta_p,
            "price_direction_label": self.price_direction_label,
            "verdict_matches_price": self.verdict_matches_price,
            "delta_v": self.delta_v,
            "activity_level": self.activity_level,
        }


def verify_market_behavior(
    p0: float,
    p1: float,
    verdict: str,
    volume_before: float | None = None,
    volume_after: float | None = None,
) -> MarketBehaviorResult:
    delta_p = (p1 - p0) / p0 if p0 else 0.0
    price_direction_label = 1 if delta_p >= 0 else -1
    verdict_label = 1 if verdict == "BULLISH" else -1 if verdict == "BEARISH" else 0
    delta_v = _delta_v(volume_before, volume_after)
    return MarketBehaviorResult(
        delta_p=delta_p,
        price_direction_label=price_direction_label,
        verdict_matches_price=verdict_label == price_direction_label,
        delta_v=delta_v,
        activity_level=_activity_level(delta_v),
    )


def _delta_v(before: float | None, after: float | None) -> float | None:
    if before in (None, 0) or after is None:
        return None
    return (after - before) / before


def _activity_level(delta_v: float | None) -> ActivityLevel:
    if delta_v is None:
        return "unknown"
    if delta_v >= 0.5:
        return "high"
    if delta_v <= -0.2:
        return "low"
    return "normal"
