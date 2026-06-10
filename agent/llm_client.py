"""LLM client protocol for future online debate providers."""

from __future__ import annotations

from typing import Protocol

from agent.schema import Argument, Camp
from data.schema import CommentBlock
from profiles.user_profile import UserProfile


class DebateClient(Protocol):
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
        """Generate one structured argument."""



