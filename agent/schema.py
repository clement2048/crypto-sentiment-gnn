"""Schemas for structured bull/bear debate outputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from config import PROBABILITY_MAX, PROBABILITY_MIN
from data.schema import datetime_to_str, parse_datetime

Camp = Literal["bull", "bear"]
EvidenceSource = Literal["root_comment", "reply", "profile", "post", "argument", "prior_argument"]


@dataclass
class Evidence:
    source: str
    quote: str
    relevance: float
    source_type: EvidenceSource
    source_id: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Evidence":
        source = str(data.get("source") or "").strip()
        source_type = _normalize_evidence_source_type(data.get("source_type"))
        source_id = str(data.get("source_id") or "")
        if source and ":" in source:
            prefix, _, suffix = source.partition(":")
            source_type = _normalize_evidence_source_type(prefix)
            source_id = suffix
        elif not source:
            source = _compose_source(source_type, source_id)
        return cls(
            source=source,
            quote=str(data.get("quote") or ""),
            relevance=_clamp01(data.get("relevance", 0.0)),
            source_type=source_type,
            source_id=source_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "quote": self.quote,
            "relevance": self.relevance,
        }


@dataclass
class Argument:
    argument_id: str
    agent_id: str
    camp: Camp
    role: str
    claim: str
    evidence: list[Evidence]
    confidence: float
    target_args: list[str]
    cited_comment_ids: list[str]
    round: int
    seq: int
    phase: str = ""
    t_index: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Argument":
        role = str(data.get("role") or "")
        camp = data.get("camp") or _camp_from_role(role)
        if camp not in ("bull", "bear"):
            raise ValueError(f"Invalid argument camp: {camp}")
        argument_id = str(data.get("argument_id") or "")
        claim = str(data.get("claim") or "")
        if not argument_id:
            raise ValueError("Argument missing argument_id")
        if not claim:
            raise ValueError("Argument missing claim")
        return cls(
            argument_id=argument_id,
            agent_id=str(data.get("agent_id") or ""),
            camp=camp,
            role=role,
            claim=claim,
            evidence=[Evidence.from_dict(item) for item in data.get("evidence", [])],
            confidence=_clamp01(data.get("confidence", 0.0)),
            target_args=[str(item) for item in data.get("target_args", [])],
            cited_comment_ids=[str(item) for item in data.get("cited_comment_ids", [])],
            round=int(data.get("round") or 0),
            seq=int(data.get("seq") or 0),
            phase=str(data.get("phase") or ""),
            t_index=float(data.get("t_index") or 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "argument_id": self.argument_id,
            "agent_id": self.agent_id,
            "camp": self.camp,
            "role": self.role,
            "claim": self.claim,
            "evidence": [item.to_dict() for item in self.evidence],
            "confidence": self.confidence,
            "target_args": self.target_args,
            "cited_comment_ids": self.cited_comment_ids,
            "round": self.round,
            "seq": self.seq,
            "phase": self.phase,
            "t_index": self.t_index,
        }


@dataclass
class DebateTranscript:
    block_id: str
    t0: datetime
    rounds: int
    arguments: list[Argument]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DebateTranscript":
        t0 = parse_datetime(data.get("t0"))
        if t0 is None:
            raise ValueError("DebateTranscript missing valid t0")
        return cls(
            block_id=str(data.get("block_id") or ""),
            t0=t0,
            rounds=int(data.get("rounds") or 0),
            arguments=[Argument.from_dict(item) for item in data.get("arguments", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "t0": datetime_to_str(self.t0),
            "rounds": self.rounds,
            "arguments": [item.to_dict() for item in self.arguments],
        }


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(PROBABILITY_MIN, min(PROBABILITY_MAX, numeric))


def _camp_from_role(role: str) -> Camp | None:
    if role == "bull_agent":
        return "bull"
    if role == "bear_agent":
        return "bear"
    return None


def _normalize_evidence_source_type(value: Any) -> EvidenceSource:
    text = str(value or "").strip()
    aliases = {
        "root": "root_comment",
        "comment": "root_comment",
        "comment_block": "root_comment",
        "conversation": "root_comment",
        "user_profile": "profile",
        "author_profile": "profile",
        "prior": "prior_argument",
        "previous_argument": "prior_argument",
    }
    normalized = aliases.get(text, text)
    if normalized in ("root_comment", "reply", "profile", "post", "argument", "prior_argument"):
        return normalized  # type: ignore[return-value]
    return "post"


def _compose_source(source_type: EvidenceSource, source_id: str) -> str:
    if source_type == "root_comment":
        return f"comment:{source_id}" if source_id else "comment"
    if source_type == "reply":
        return f"comment:{source_id}" if source_id else "comment"
    if source_type == "profile":
        return f"profile:{source_id}" if source_id else "profile"
    if source_type in ("argument", "prior_argument"):
        return f"argument:{source_id}" if source_id else "argument"
    return "post"



