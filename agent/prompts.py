"""Load markdown context files for bull/bear debate agents."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

SPEC_DIR = Path(__file__).resolve().parents[1] / "agent_specs"


@lru_cache(maxsize=None)
def read_agent_spec(name: str) -> str:
    return (SPEC_DIR / name).read_text(encoding="utf-8").strip()


def get_agent_system_prompt(role: str) -> str:
    role_file = "bull_agent.md" if role == "bull_agent" else "bear_agent.md"
    return f"{read_agent_spec('shared_rules.md')}\n\n{read_agent_spec(role_file)}"
