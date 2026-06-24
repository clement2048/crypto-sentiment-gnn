"""Shared payload helpers for LLM debate prompts."""

from __future__ import annotations

import json
from typing import Any

from agent.schema import Argument, Camp
from data.schema import CommentBlock, datetime_to_str
from profiles.user_profile import UserProfile


def build_user_prompt(
    block: CommentBlock,
    profiles: dict[str, UserProfile],
    camp: Camp,
    role: str,
    round_index: int,
    seq: int,
    prior_arguments: list[Argument],
    phase: str,
    available_target_ids: list[str] | None,
    expected_argument_id: str,
    expected_agent_id: str,
) -> str:
    target_ids = available_target_ids
    if target_ids is None:
        target_ids = [item.argument_id for item in prior_arguments if item.camp != camp]
    payload = {
        "task": "Generate exactly one structured debate argument.",
        "phase": phase,
        "phase_instructions": phase_instructions(phase),
        "required_metadata": {
            "argument_id": expected_argument_id,
            "agent_id": expected_agent_id,
            "role": role,
            "round": round_index,
            "seq": seq,
            "phase": phase,
        },
        "available_target_ids": target_ids,
        "comment_block": {
            "block_id": block.block_id,
            "post_id": block.post_id,
            "product": block.product,
            "market_type": block.market_type,
            "t0": datetime_to_str(block.t0),
            "t_window": block.t_window,
            "root_comment": comment_for_prompt(block.root_comment),
            "replies": [comment_for_prompt(reply) for reply in block.replies[:20]],
            "case_context_comments": [
                comment_for_prompt(comment)
                for comment in getattr(block, "case_context_comments", [])[:20]
            ],
            "post_content": block.post_content[:2000],
        },
        "time_safe_profiles": {
            author: profile.to_dict()
            for author, profile in sorted(profiles.items())
        },
        "prior_arguments": [
            {
                "argument_id": item.argument_id,
                "agent_id": item.agent_id,
                "role": item.role,
                "phase": item.phase,
                "claim": item.claim,
                "confidence": item.confidence,
                "target_args": item.target_args,
            }
            for item in prior_arguments[-16:]
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def phase_instructions(phase: str) -> str:
    if phase == "initial_argument":
        return "Generate an independent opening argument. Do not target previous arguments."
    if phase == "rebuttal":
        return (
            "Generate a concise targeted rebuttal. Use only target_args from available_target_args, "
            "answer the opponent's latest claim directly, and ground the response in supplied text "
            "or time-safe profile signals."
        )
    if phase.startswith("reflection_supplement"):
        return (
            "Generate a concise supplement that repairs weak dimensions identified by Judge. "
            "Use only target_args from available_target_args and do not use future labels or prices."
        )
    return "Follow the requested debate phase and use only provided data."


def comment_for_prompt(comment: Any) -> dict[str, Any]:
    return {
        "comment_id": comment.comment_id,
        "original_comment_id": comment.original_comment_id,
        "author": comment.author,
        "text": comment.text,
        "post_time": datetime_to_str(comment.post_time),
        "replies": [comment_for_prompt(reply) for reply in comment.replies[:20]],
    }


def normalize_argument_metadata(
    argument: Argument,
    expected_argument_id: str,
    expected_agent_id: str,
    camp: Camp,
    role: str,
    round_index: int,
    seq: int,
    phase: str,
    available_target_ids: list[str] | None = None,
) -> Argument:
    argument.argument_id = expected_argument_id
    argument.agent_id = expected_agent_id
    argument.camp = camp
    argument.role = role
    argument.round = round_index
    argument.seq = seq
    argument.phase = phase
    if available_target_ids is not None:
        allowed = set(available_target_ids)
        argument.target_args = [target_id for target_id in argument.target_args if target_id in allowed]
    return argument
