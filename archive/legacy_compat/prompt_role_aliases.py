"""Legacy role aliases from the old multi-agent debate prototype.

This module is archived reference only. Active prompts no longer normalize old
role names into bull_agent/bear_agent.
"""

BULL_ROLE = "bull_agent"
BEAR_ROLE = "bear_agent"

ROLE_ALIASES = {
    "price_action_agent": BULL_ROLE,
    "news_context_agent": BULL_ROLE,
    "user_profile_agent": BULL_ROLE,
    "technical_analysis_agent": BULL_ROLE,
    "fundamental_analysis_agent": BULL_ROLE,
    "sentiment_contagion_agent": BULL_ROLE,
    "risk_rebuttal_agent": BEAR_ROLE,
    "risk_analysis_agent": BEAR_ROLE,
    "onchain_skeptic_agent": BEAR_ROLE,
    "sentiment_reversal_agent": BEAR_ROLE,
    "reflection_agent": BULL_ROLE,
}

