"""Prompt template for the future online judge provider."""

from __future__ import annotations

from agent.prompts import read_agent_spec


JUDGE_JSON_INSTRUCTION = """Output pure JSON matching JudgeOutput schema. Do not use markdown."""


JUDGE_OUTPUT_SCHEMA_PROMPT = f"""{read_agent_spec('judge.md')}

{JUDGE_JSON_INSTRUCTION}

Required JSON object:
{{
  "verdict": "BULLISH | BEARISH",
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
  "weak_dims": ["dimension names for reflection, no ground-truth labels"],
  "supplement_suggestions": ["safe suggestions for bull/bear debaters"],
  "consistency_flags": []
}}

Score meanings:
- p_bull / p_bear: final persuasive strength of bull/bear camp.
- q_bull / q_bear: logic quality of bull/bear camp.
- e_bull / e_bear: evidence quality of bull/bear camp.
- c: consensus degree.
- d: debate depth, including rebuttal-chain sufficiency.
- a: agreement between debate-graph logic and ODE/model trend.
- rho: final self-rated confidence.

Report requirements:
- report must be a five-section structured text: verdict, argument strength, weak dimensions, supplement suggestions, reasoning.
- verdict must choose one direction. Do not output any third option, unclear, abstain, or tie.
- weak_dims and supplement_suggestions must not reveal ground-truth labels, p1, or future prices.
- model_summary is provided as values plus field_descriptions and interpretation_notes. Read those explanations before using any numeric model field.
- Do not treat internal ODE diagnostics such as bull_mean, bear_mean, bull_bear_margin, or net_score_mean as calibrated probabilities or final labels.
"""
