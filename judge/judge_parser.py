"""Parse and validate structured judge JSON outputs."""

from __future__ import annotations

import json
import re

from agent.output_parser import _parse_json_object
from judge.judge_schema import JudgeOutput


def parse_judge_json(text: str) -> JudgeOutput:
    try:
        parsed = _parse_json_object(text)
    except json.JSONDecodeError:
        raw = text.strip()
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Judge output must be a JSON object")
    return JudgeOutput.from_dict(parsed)



