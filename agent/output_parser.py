"""Parse and validate structured Agent JSON outputs."""
from __future__ import annotations

import json
import re

from agent.schema import Argument


def parse_argument_json(text: str) -> Argument:
    """Parse one Agent response as an Argument."""
    data = _parse_json_object(text)
    return Argument.from_dict(data)


def _parse_json_object(text: str) -> dict:
    raw = text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:])
        raw = raw.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        repaired = _repair_json_object(raw)
        if repaired is not None:
            parsed = repaired
        else:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                raise
            repaired = _repair_json_object(match.group(0))
            if repaired is not None:
                parsed = repaired
            else:
                parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Agent output must be a JSON object")
    return parsed


def _repair_json_object(raw: str) -> dict | None:
    """Best-effort local repair for common LLM JSON mistakes."""
    try:
        from json_repair import repair_json
    except ImportError:
        return None
    try:
        repaired = repair_json(raw, return_objects=True)
    except Exception:
        return None
    return repaired if isinstance(repaired, dict) else None



