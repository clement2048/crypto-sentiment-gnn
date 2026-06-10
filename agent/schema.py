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
    source_type: EvidenceSource
    source_id: str
    quote: str
    relevance: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Evidence":
        source_type = _normalize_evidence_source_type(data.get("source_type"))
        return cls(
            source_type=source_type,
            source_id=str(data.get("source_id") or ""),
            quote=str(data.get("quote") or ""),
            relevance=_clamp01(data.get("relevance", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
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
    targets: list[str]
    cited_comment_ids: list[str]
    round: int
    seq: int
    phase: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Argument":
        camp = data.get("camp")
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
            role=str(data.get("role") or ""),
            claim=claim,
            evidence=[Evidence.from_dict(item) for item in data.get("evidence", [])],
            confidence=_clamp01(data.get("confidence", 0.0)),
            targets=[str(item) for item in data.get("targets", [])],
            cited_comment_ids=[str(item) for item in data.get("cited_comment_ids", [])],
            round=int(data.get("round") or 0),
            seq=int(data.get("seq") or 0),
            phase=str(data.get("phase") or ""),
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
            "targets": self.targets,
            "cited_comment_ids": self.cited_comment_ids,
            "round": self.round,
            "seq": self.seq,
            "phase": self.phase,
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



