"""Deterministic offline debate client."""

from __future__ import annotations

from agent.prompts import normalize_role
from agent.schema import Argument, Camp, Evidence
from config import (
    MOCK_CONFIDENCE_BASE,
    MOCK_CONFIDENCE_MAX,
    MOCK_CONFIDENCE_PRIOR_BONUS,
    MOCK_CONFIDENCE_PRIOR_CAP,
    MOCK_CONFIDENCE_PROFILE_BONUS,
    MOCK_DEBATE_TARGET_LIMIT,
    MOCK_POST_RELEVANCE,
    MOCK_PROFILE_RELEVANCE,
    MOCK_ROOT_COMMENT_RELEVANCE,
    PROBABILITY_MAX,
    PROBABILITY_MIN,
)
from data.schema import CommentBlock
from profiles.user_profile import UserProfile


class MockDebateClient:
    """Generate reproducible structured arguments without network access."""

    def generate_argument(
        self,
        block: CommentBlock,
        profiles: dict[str, UserProfile],
        camp: Camp,
        role: str,
        round_index: int,
        seq: int,
        prior_arguments: list[Argument],
        phase: str = "initial_argument",
        available_target_ids: list[str] | None = None,
    ) -> Argument:
        root = block.root_comment
        profile = profiles.get(root.author)
        profile_bias = profile.stance_bias if profile else 0.0
        product = block.product or "asset"
        confidence = _confidence_from_context(camp, profile_bias, len(prior_arguments))
        if available_target_ids is None:
            available_target_ids = [item.argument_id for item in prior_arguments if item.camp != camp]
        target_ids = available_target_ids[-MOCK_DEBATE_TARGET_LIMIT:] if MOCK_DEBATE_TARGET_LIMIT > 0 else []

        claim = _claim_for_role(
            camp=camp,
            role=role,
            product=product,
            root_text=root.text,
            profile_bias=profile_bias,
            has_targets=bool(target_ids),
        )
        evidence = [
            Evidence(
                source_type="root_comment",
                source_id=root.comment_id,
                quote=root.text[:120],
                relevance=MOCK_ROOT_COMMENT_RELEVANCE,
            ),
            Evidence(
                source_type="post",
                source_id=block.post_id,
                quote=block.post_content[:120],
                relevance=MOCK_POST_RELEVANCE,
            ),
        ]
        if profile is not None:
            evidence.append(
                Evidence(
                    source_type="profile",
                    source_id=root.author,
                    quote=f"history_count={profile.history_count}, stance_bias={profile.stance_bias:.2f}",
                    relevance=MOCK_PROFILE_RELEVANCE,
                )
            )

        argument_id = f"{block.block_id}:r{round_index}:s{seq}:{camp}"
        return Argument(
            argument_id=argument_id,
            agent_id=f"{camp}_{role}",
            camp=camp,
            role=role,
            claim=_phase_prefix(phase) + claim,
            evidence=evidence,
            confidence=confidence,
            targets=target_ids,
            cited_comment_ids=[root.comment_id],
            round=round_index,
            seq=seq,
            phase=phase,
        )


def _claim_for_role(
    camp: Camp,
    role: str,
    product: str,
    root_text: str,
    profile_bias: float,
    has_targets: bool,
) -> str:
    direction = "bullish" if camp == "bull" else "bearish"
    normalized_role = normalize_role(role)
    if normalized_role == "technical_analysis_agent":
        return f"{product} shows a possible {direction} technical reading from the discussion tone; root comment: {root_text[:48]}"
    if normalized_role == "fundamental_analysis_agent":
        return f"The post context gives {product} a possible {direction} fundamental narrative, pending stronger evidence."
    if normalized_role == "sentiment_contagion_agent":
        bias_text = "bull-biased" if profile_bias > 0 else "bear-biased" if profile_bias < 0 else "neutral-or-cold-start"
        return f"The author profile is {bias_text}; social sentiment may transmit a {direction} reading."
    if normalized_role == "risk_analysis_agent":
        return f"The bear risk view argues that uncertainty around {product} can support a {direction} reading."
    if normalized_role == "onchain_skeptic_agent":
        return f"The available text does not prove healthy on-chain accumulation for {product}, leaving a {direction} risk case."
    if normalized_role == "sentiment_reversal_agent":
        return f"Visible emotion around {product} may be fragile or crowded, supporting a {direction} reversal reading."
    if normalized_role == "reflection_agent" and has_targets:
        return f"The {direction} reflection agent answers opposing critiques while noting evidence limits."
    if normalized_role == "reflection_agent":
        return f"The {direction} reflection agent gives a cautious thesis and names uncertainty before overclaiming."
    if has_targets:
        return f"The {direction} camp responds that opposing arguments do not rule out this direction."
    return f"The {direction} camp adds a risk view while keeping this direction as a viable explanation."


def _phase_prefix(phase: str) -> str:
    labels = {
        "initial_argument": "[initial] ",
        "intra_reflection": "[intra-reflection] ",
        "intra_response": "[intra-response] ",
        "cross_response": "[cross-response] ",
        "counter_reflection": "[counter-reflection] ",
        "counter_rebuttal": "[counter-rebuttal] ",
        "reflection_summary": "[reflection-summary] ",
    }
    return labels.get(phase, "")


def _confidence_from_context(camp: Camp, profile_bias: float, prior_count: int) -> float:
    base = MOCK_CONFIDENCE_BASE + min(prior_count, MOCK_CONFIDENCE_PRIOR_CAP) * MOCK_CONFIDENCE_PRIOR_BONUS
    if camp == "bull" and profile_bias > 0:
        base += MOCK_CONFIDENCE_PROFILE_BONUS
    if camp == "bear" and profile_bias < 0:
        base += MOCK_CONFIDENCE_PROFILE_BONUS
    return max(PROBABILITY_MIN, min(MOCK_CONFIDENCE_MAX, min(PROBABILITY_MAX, base)))



