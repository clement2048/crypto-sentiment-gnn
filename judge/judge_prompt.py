"""Prompt template for the future online judge provider."""

from __future__ import annotations

from agent.prompts import JUDGE_SYSTEM_PROMPT


JUDGE_JSON_INSTRUCTION = """Output pure JSON matching JudgeOutput schema. Do not use markdown."""


JUDGE_OUTPUT_SCHEMA_PROMPT = f"""{JUDGE_SYSTEM_PROMPT}

{JUDGE_JSON_INSTRUCTION}

Required JSON object:
{{
  "verdict": "BULLISH | BEARISH | NEUTRAL",
  "confidence": 0.0,
  "report": "structured natural-language analysis",
  "score_vector": {{
    "p_bull": 0.0,
    "p_bear": 0.0,
    "q_bull": 0.0,
    "q_bear": 0.0,
    "e_bull": 0.0,
    "e_bear": 0.0,
    "c": 0.0,
    "d": 0.0,
    "a": 0.0,
    "rho": 0.0
  }},
  "consistency_flags": []
}}

Score meanings:
- p_bull / p_bear: final persuasive strength of bull/bear camp.
- q_bull / q_bear: evidence quality of bull/bear camp.
- e_bull / e_bear: ODE/model evolution support for bull/bear direction.
- c: role coverage and information coverage.
- d: debate depth, including rebuttal-chain sufficiency.
- a: agreement between debate-graph logic and ODE/model trend.
- rho: final self-rated confidence.
"""
