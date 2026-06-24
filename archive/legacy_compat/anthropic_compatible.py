"""Anthropic-compatible LLM clients (debate + judge).
为走 Anthropic Messages 兼容协议的 provider 提供共享 HTTP 层。每个 provider 只需要写一层 ~10 行的"薄配置"。

API key 由 config.py 从项目根目录 .env 或系统环境变量读取。

注意:不引入 ``anthropic`` Python SDK,沿用项目惯例手写 ``urllib``。
"""

from __future__ import annotations

import copy
import hashlib
import http.client
import json
from pathlib import Path
import socket
import ssl
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from agent.output_parser import parse_argument_json
from agent.prompts import get_agent_system_prompt
from agent.schema import Argument, Camp
from config import (
    DEEPSEEK_ANTHROPIC_BASE_URL,
    DEEPSEEK_ANTHROPIC_VERSION,
    DEEPSEEK_API_KEY,
    DEEPSEEK_CACHE_DIR,
    DEEPSEEK_CACHE_ENABLED,
    DEEPSEEK_HTTP_RETRIES,
    DEEPSEEK_MAX_TOKENS,
    DEEPSEEK_MODEL,
    DEEPSEEK_TEMPERATURE,
    DEEPSEEK_THINKING_TYPE,
    DEEPSEEK_TIMEOUT_SECONDS,
)
from data.schema import CommentBlock, datetime_to_str
from profiles.user_profile import UserProfile

# Judge 端的 import 故意留在函数内部 lazy import,避免:
#   agent/__init__.py → agent.anthropic_compatible → judge.consistency →
#   judge/__init__.py → judge.client_factory → agent.anthropic_compatible
# 这种循环。
# 以下是 judge 端会用到的符号,仅在 AnthropicCompatibleJudgeClient 方法内 import:
#   - judge.consistency.check_judge_consistency
#   - judge.judge_parser.parse_judge_json
#   - judge.judge_prompt.JUDGE_OUTPUT_SCHEMA_PROMPT
#   - judge.judge_schema.JudgeOutput (type hint, 由 from __future__ import annotations 推迟)
#   - debate_graph.schema.HeteroGraph (type hint, 同上)
#   - model.model_summary.ModelOutputSummary (type hint, 同上)
#   - agent.schema.DebateTranscript (type hint, 同上)


Transport = Callable[[dict[str, Any]], dict[str, Any]]


# =============================================================================
# Debate 基类
# =============================================================================


class AnthropicCompatibleDebateClient:
    """所有 Anthropic 兼容 provider 的共享 HTTP 层。

    子类(具体 provider)只需要传 ``base_url`` / ``model`` / ``configured_api_key`` /
    ``cache_dir`` 等常量,所有 HTTP / 重试 / 缓存 / 解析逻辑都从这里继承。
    """

    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        model: str,
        max_tokens: int,
        temperature: float,
        anthropic_version: str,
        include_thinking: bool,
        thinking_type: str,
        timeout_seconds: float,
        http_retries: int,
        cache_enabled: bool,
        cache_dir: str | Path,
        configured_api_key: str,
        api_key_label: str,
        transport: Transport | None,
    ):
        self.api_key = api_key or configured_api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.anthropic_version = anthropic_version
        self.include_thinking = include_thinking
        self.thinking_type = thinking_type
        self.timeout_seconds = timeout_seconds
        self.http_retries = http_retries
        self.cache_enabled = cache_enabled
        self.cache_dir = Path(cache_dir)
        self.api_key_label = api_key_label
        self.transport = transport

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
        """调用 Anthropic-compatible API,把返回 JSON 解析成 Argument。"""
        expected_argument_id = f"{block.block_id}:r{round_index}:s{seq}:{camp}"
        expected_agent_id = f"{camp}_{role}"
        payload = self._build_payload(
            block=block,
            profiles=profiles,
            camp=camp,
            role=role,
            round_index=round_index,
            seq=seq,
            prior_arguments=prior_arguments,
            phase=phase,
            available_target_ids=available_target_ids,
            expected_argument_id=expected_argument_id,
            expected_agent_id=expected_agent_id,
        )
        response = self._send_payload(payload)
        text = _extract_text(response)
        try:
            argument = parse_argument_json(text)
        except (ValueError, json.JSONDecodeError):
            repair_payload = self._build_repair_payload(
                invalid_text=text,
                original_payload=payload,
                expected_argument_id=expected_argument_id,
                expected_agent_id=expected_agent_id,
                camp=camp,
                role=role,
                round_index=round_index,
                seq=seq,
                phase=phase,
            )
            repair_response = self._send_payload(repair_payload)
            argument = parse_argument_json(_extract_text(repair_response))
        return _normalize_argument_metadata(
            argument=argument,
            expected_argument_id=expected_argument_id,
            expected_agent_id=expected_agent_id,
            camp=camp,
            role=role,
            round_index=round_index,
            seq=seq,
            phase=phase,
            available_target_ids=available_target_ids,
        )

    def _build_payload(
        self,
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
    ) -> dict[str, Any]:
        user_prompt = _build_user_prompt(
            block=block,
            profiles=profiles,
            camp=camp,
            role=role,
            round_index=round_index,
            seq=seq,
            prior_arguments=prior_arguments,
            phase=phase,
            available_target_ids=available_target_ids,
            expected_argument_id=expected_argument_id,
            expected_agent_id=expected_agent_id,
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": get_agent_system_prompt(role),
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}],
                }
            ],
        }
        if self.include_thinking:
            payload["thinking"] = {"type": self.thinking_type}
        return payload

    def _build_repair_payload(
        self,
        invalid_text: str,
        original_payload: dict[str, Any],
        expected_argument_id: str,
        expected_agent_id: str,
        camp: Camp,
        role: str,
        round_index: int,
        seq: int,
        phase: str,
    ) -> dict[str, Any]:
        """要求模型把上一次非法输出修成可解析 JSON。"""
        payload = copy.deepcopy(original_payload)
        payload["temperature"] = 0.0
        payload["messages"] = [
            *payload["messages"],
            {
                "role": "assistant",
                "content": [{"type": "text", "text": invalid_text[:4000]}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "The previous assistant response was invalid JSON. "
                            "Repair it into exactly one valid JSON object matching the Argument schema. "
                            "Return only JSON, with no markdown and no explanation. "
                            f"Use argument_id={expected_argument_id}, agent_id={expected_agent_id}, "
                            f"camp={camp}, role={role}, round={round_index}, seq={seq}, phase={phase}."
                        ),
                    }
                ],
            },
        ]
        return payload

    def _send_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.transport:
            return self.transport(payload)
        cached = _read_cached_response(self.cache_dir, self.cache_enabled, payload)
        if cached is not None:
            return cached
        response = self._post_messages(payload)
        _write_cached_response(self.cache_dir, self.cache_enabled, payload, response)
        return response

    def _post_messages(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError(
                f"Missing Anthropic-compatible API key. "
                f"Set {self.api_key_label} in your .env file."
            )

        request = urllib.request.Request(
            url=f"{self.base_url}/v1/messages",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": self.anthropic_version,
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(1, self.http_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code < 500 or attempt == self.http_retries:
                    raise RuntimeError(f"Anthropic-compatible API HTTP {exc.code}: {body}") from exc
                last_error = exc
            except urllib.error.URLError as exc:
                if attempt == self.http_retries:
                    raise RuntimeError(
                        f"Anthropic-compatible API request failed after {self.http_retries} attempts: {exc}"
                    ) from exc
                last_error = exc
            except (TimeoutError, socket.timeout) as exc:
                if attempt == self.http_retries:
                    raise RuntimeError(
                        f"Anthropic-compatible API timed out after {self.http_retries} attempts: {exc}"
                    ) from exc
                last_error = exc
            except (http.client.IncompleteRead, http.client.RemoteDisconnected, ConnectionResetError, ssl.SSLError) as exc:
                if attempt == self.http_retries:
                    raise RuntimeError(
                        f"Anthropic-compatible API connection closed after {self.http_retries} attempts: {exc}"
                    ) from exc
                last_error = exc
            time.sleep(0.8 * attempt)
        else:
            raise RuntimeError(f"Anthropic-compatible API request failed: {last_error}")
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Anthropic-compatible response must be a JSON object")
        if "error" in parsed:
            raise RuntimeError(f"Anthropic-compatible API error: {parsed['error']}")
        return parsed


# =============================================================================
# Judge 基类
# =============================================================================


class AnthropicCompatibleJudgeClient:
    """所有 Anthropic 兼容 provider 的共享 judge 层。"""

    def __init__(self, http_client: AnthropicCompatibleDebateClient):
        self.http = http_client

    def judge(
        self,
        transcript: DebateTranscript,
        model_summary: ModelOutputSummary,
        graph: HeteroGraph,
    ) -> JudgeOutput:
        """在辩论和模型演化都完成后调用 LLM 法官。"""
        # lazy import 避免循环:judge.consistency / judge.judge_parser 会在
        # judge/__init__.py 加载时尝试拉取 agent.anthropic_compatible。
        from judge.consistency import check_judge_consistency
        from judge.judge_parser import parse_judge_json

        payload = self._build_payload(transcript, model_summary, graph)
        response = self.http._send_payload(payload)
        text = _extract_text(response)
        try:
            output = parse_judge_json(text)
        except (ValueError, json.JSONDecodeError):
            repair_payload = self._build_repair_payload(payload, text)
            repair_response = self.http._send_payload(repair_payload)
            output = parse_judge_json(_extract_text(repair_response))
        output.consistency_flags = sorted(set([*output.consistency_flags, *check_judge_consistency(output)]))
        return output

    def _build_payload(
        self,
        transcript: DebateTranscript,
        model_summary: ModelOutputSummary,
        graph: HeteroGraph,
    ) -> dict[str, Any]:
        # lazy import 避免循环
        from judge.judge_prompt import JUDGE_OUTPUT_SCHEMA_PROMPT

        payload: dict[str, Any] = {
            "model": self.http.model,
            "max_tokens": self.http.max_tokens,
            "temperature": self.http.temperature,
            "system": JUDGE_OUTPUT_SCHEMA_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                _judge_input(transcript, model_summary, graph),
                                ensure_ascii=False,
                                indent=2,
                            ),
                        }
                    ],
                }
            ],
        }
        if self.http.include_thinking:
            payload["thinking"] = {"type": self.http.thinking_type}
        return payload

    def _build_repair_payload(self, original_payload: dict[str, Any], invalid_text: str) -> dict[str, Any]:
        payload = copy.deepcopy(original_payload)
        payload["temperature"] = 0.0
        payload["messages"] = [
            *payload["messages"],
            {
                "role": "assistant",
                "content": [{"type": "text", "text": invalid_text[:4000]}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "The previous response was not a valid JudgeOutput JSON object. "
                            "Repair it into exactly one valid JSON object with verdict, confidence, "
                            "report, score_vector, and consistency_flags. Return only JSON. "
                            "The verdict must be exactly BULLISH or BEARISH."
                        ),
                    }
                ],
            },
        ]
        return payload


# =============================================================================
# Provider 薄配置:DeepSeek
# =============================================================================


class DeepSeekAnthropicDebateClient(AnthropicCompatibleDebateClient):
    """DeepSeek Anthropic-compatible 协议辩论 client(薄配置)。"""

    def __init__(
        self,
        api_key: str | None = None,
        transport: Transport | None = None,
        **overrides: Any,
    ):
        super().__init__(
            api_key=api_key,
            base_url=DEEPSEEK_ANTHROPIC_BASE_URL,
            model=DEEPSEEK_MODEL,
            max_tokens=DEEPSEEK_MAX_TOKENS,
            temperature=DEEPSEEK_TEMPERATURE,
            anthropic_version=DEEPSEEK_ANTHROPIC_VERSION,
            include_thinking=True,
            thinking_type=DEEPSEEK_THINKING_TYPE,
            timeout_seconds=DEEPSEEK_TIMEOUT_SECONDS,
            http_retries=DEEPSEEK_HTTP_RETRIES,
            cache_enabled=DEEPSEEK_CACHE_ENABLED,
            cache_dir=DEEPSEEK_CACHE_DIR,
            configured_api_key=DEEPSEEK_API_KEY,
            api_key_label="DEEPSEEK_API_KEY",
            transport=transport,
            **overrides,
        )


class DeepSeekJudgeClient(AnthropicCompatibleJudgeClient):
    """DeepSeek 法官 client(薄配置)。"""

    def __init__(
        self,
        transport: Transport | None = None,
        **overrides: Any,
    ):
        super().__init__(
            http_client=DeepSeekAnthropicDebateClient(transport=transport, **overrides)
        )


def _build_user_prompt(
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
        "phase_instructions": _phase_instructions(phase),
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
            "root_comment": _comment_for_prompt(block.root_comment),
            # Agent receives the root comment plus a bounded reply view. This is
            # the LLM-readable side of the same CommentBlock that graph builders
            # later convert into comment nodes.
            "replies": [_comment_for_prompt(reply) for reply in block.replies[:20]],
            "case_context_comments": [
                _comment_for_prompt(comment)
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


def _phase_instructions(phase: str) -> str:
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


def _comment_for_prompt(comment) -> dict[str, Any]:
    """给 LLM 的评论视图：不暴露 p1/label 等未来验证字段。"""
    return {
        "comment_id": comment.comment_id,
        "original_comment_id": comment.original_comment_id,
        "author": comment.author,
        "text": comment.text,
        "post_time": datetime_to_str(comment.post_time),
        "replies": [_comment_for_prompt(reply) for reply in comment.replies[:20]],
    }


def _extract_text(response: dict[str, Any]) -> str:
    content = response.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                texts.append(str(item.get("text") or ""))
            elif isinstance(item, dict) and isinstance(item.get("content"), str):
                texts.append(str(item["content"]))
            elif isinstance(item, str):
                texts.append(item)
        if texts:
            return "\n".join(texts)
    completion = response.get("completion")
    if isinstance(completion, str):
        return completion
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(first.get("text"), str):
                return first["text"]
    raise ValueError(f"Anthropic-compatible response missing text content: {_describe_response_shape(response)}")


def _describe_response_shape(response: dict[str, Any]) -> dict[str, Any]:
    """返回不含正文的响应结构摘要,方便排查 provider 格式差异。"""
    summary: dict[str, Any] = {
        "keys": sorted(response.keys()),
        "content_type": type(response.get("content")).__name__,
        "stop_reason": response.get("stop_reason"),
    }
    content = response.get("content")
    if isinstance(content, list):
        summary["content_blocks"] = [
            sorted(item.keys()) if isinstance(item, dict) else type(item).__name__
            for item in content[:5]
        ]
        summary["content_block_types"] = [
            item.get("type") if isinstance(item, dict) else None
            for item in content[:5]
        ]
    choices = response.get("choices")
    if isinstance(choices, list):
        summary["choices_len"] = len(choices)
        if choices and isinstance(choices[0], dict):
            summary["first_choice_keys"] = sorted(choices[0].keys())
    return summary


def _read_cached_response(
    cache_dir: Path, cache_enabled: bool, payload: dict[str, Any]
) -> dict[str, Any] | None:
    if not cache_enabled:
        return None
    path = _cache_path(cache_dir, payload)
    if not path.exists():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return cached if isinstance(cached, dict) else None


def _write_cached_response(
    cache_dir: Path, cache_enabled: bool, payload: dict[str, Any], response: dict[str, Any]
) -> None:
    if not cache_enabled:
        return
    path = _cache_path(cache_dir, payload)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        return


def _cache_path(cache_dir: Path, payload: dict[str, Any]) -> Path:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"{digest}.json"


def _normalize_argument_metadata(
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
    """固定协议字段,避免 LLM 小幅写错 id 导致下游图构建不稳定。"""
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


def _judge_input(
    transcript: DebateTranscript,
    model_summary: ModelOutputSummary,
    graph: HeteroGraph,
) -> dict[str, Any]:
    """给法官的输入视图：不包含真实 label、p1 等未来验证字段。"""
    return {
        "task": "Produce final JudgeOutput after comparing debate graph and Bi-ODE/model summary.",
        "block_id": transcript.block_id,
        "t0": transcript.t0.strftime("%Y-%m-%d %H:%M:%S"),
        "model_summary": model_summary.explained_dict(),
        "debate_arguments": [
            {
                "argument_id": item.argument_id,
                "agent_id": item.agent_id,
                "camp": item.camp,
                "role": item.role,
                "claim": item.claim,
                "evidence": [evidence.to_dict() for evidence in item.evidence],
                "confidence": item.confidence,
                "target_args": item.target_args,
                "round": item.round,
                "seq": item.seq,
            }
            for item in transcript.arguments
        ],
        "graph": {
            "graph_id": graph.graph_id,
            "node_counts": graph.node_counts(),
            "relation_counts": graph.relation_counts(),
            "nodes": [
                {
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "ref_id": node.ref_id,
                    "text": node.text[:500],
                    "attrs": node.attrs,
                }
                for node in graph.nodes
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "relation": edge.relation,
                    "weight": edge.weight,
                }
                for edge in graph.edges
            ],
        },
        "rules": [
            "Do not use ground-truth labels, p1, or future prices.",
            "Base the decision only on debate logic and model/ODE summary.",
            "Read model_summary.field_descriptions and model_summary.interpretation_notes before using numeric model fields.",
            "If debate and model disagree, explain the disagreement, lower confidence, and still choose BULLISH or BEARISH.",
        ],
    }
