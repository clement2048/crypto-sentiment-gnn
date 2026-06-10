"""User-profile feature computation with strict temporal boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean, pstdev

from config import (
    COLD_START_ACTIVITY,
    COLD_START_CONSISTENCY,
    COLD_START_EMOTION_STABILITY,
    COLD_START_INFLUENCE,
    COLD_START_REACTION_CONSISTENCY,
    COLD_START_STANCE_BIAS,
    PRICE_TIE_COUNTS_AS_BULLISH,
    SINGLE_HISTORY_EMOTION_STABILITY,
)


@dataclass(frozen=True)
class UserHistoryRecord:
    author: str
    text: str
    timestamp: datetime
    label: int | None
    product: str | None
    p0: float | None = None
    p1: float | None = None
    reply_count: int = 0


@dataclass
class UserProfile:
    author: str
    stance_bias: float
    consistency: float
    activity: float
    influence: float
    historical_reaction_consistency: float
    emotion_stability: float
    asset_preference: dict[str, float] = field(default_factory=dict)
    built_until: datetime | None = None
    history_count: int = 0

    @classmethod
    def cold_start(cls, author: str, built_until: datetime) -> "UserProfile":
        return cls(
            author=author,
            stance_bias=COLD_START_STANCE_BIAS,
            consistency=COLD_START_CONSISTENCY,
            activity=COLD_START_ACTIVITY,
            influence=COLD_START_INFLUENCE,
            historical_reaction_consistency=COLD_START_REACTION_CONSISTENCY,
            emotion_stability=COLD_START_EMOTION_STABILITY,
            asset_preference={},
            built_until=built_until,
            history_count=0,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "author": self.author,
            "stance_bias": self.stance_bias,
            "consistency": self.consistency,
            "activity": self.activity,
            "influence": self.influence,
            "historical_reaction_consistency": self.historical_reaction_consistency,
            "emotion_stability": self.emotion_stability,
            "asset_preference": self.asset_preference,
            "built_until": self.built_until.isoformat(sep=" ") if self.built_until else None,
            "history_count": self.history_count,
        }
    


### ==================
###  构建用户画像
### ==================
def build_user_profile(
    author: str,
    history: list[UserHistoryRecord],
    as_of: datetime,
) -> UserProfile:
    """Build one profile from history that must already satisfy timestamp < as_of."""
    safe_history = [item for item in history if item.timestamp < as_of]
    if len(safe_history) != len(history):
        raise ValueError("User profile history contains records at or after t0")
    if not safe_history:
        return UserProfile.cold_start(author, built_until=as_of)

    labels = [item.label for item in safe_history if item.label in (-1, 1)]
    bull = sum(1 for label in labels if label == 1)
    bear = sum(1 for label in labels if label == -1)
    labeled_total = len(labels)
    stance_bias = (bull - bear) / labeled_total if labeled_total else COLD_START_STANCE_BIAS
    consistency = max(bull, bear) / labeled_total if labeled_total else COLD_START_CONSISTENCY

    activity = float(len(safe_history))
    influence = float(mean(item.reply_count for item in safe_history))

    reaction_checks = [
        item
        for item in safe_history
        if item.label in (-1, 1) and item.p0 not in (None, 0) and item.p1 is not None
    ]
    if reaction_checks:
        matches = 0
        for item in reaction_checks:
            assert item.p0 is not None
            assert item.p1 is not None
            if PRICE_TIE_COUNTS_AS_BULLISH:
                direction = 1 if item.p1 >= item.p0 else -1
            else:
                direction = 1 if item.p1 > item.p0 else -1
            matches += int(direction == item.label)
        reaction_consistency = matches / len(reaction_checks)
    else:
        reaction_consistency = COLD_START_REACTION_CONSISTENCY

    emotion_values = [float(label) for label in labels]
    emotion_stability = (
        1.0 / (1.0 + pstdev(emotion_values))
        if len(emotion_values) > 1
        else SINGLE_HISTORY_EMOTION_STABILITY
    )

    product_counts: dict[str, int] = {}
    for item in safe_history:
        if item.product:
            product_counts[item.product] = product_counts.get(item.product, 0) + 1
    product_total = sum(product_counts.values())
    asset_preference = (
        {product: count / product_total for product, count in sorted(product_counts.items())}
        if product_total
        else {}
    )

    built_until = max(item.timestamp for item in safe_history)
    return UserProfile(
        author=author,
        stance_bias=stance_bias,
        consistency=consistency,
        activity=activity,
        influence=influence,
        historical_reaction_consistency=reaction_consistency,
        emotion_stability=emotion_stability,
        asset_preference=asset_preference,
        built_until=built_until,
        history_count=len(safe_history),
    )


