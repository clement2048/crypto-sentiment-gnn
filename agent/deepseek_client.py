"""DeepSeek Anthropic-format debate client.

这个客户端只负责“让一个在线 LLM 生成一个结构化 Argument”。
它不会直接参与辩论顺序，辩论顺序仍由 DebateOrchestrator 控制。
"""

from __future__ import annotations

import copy
import hashlib
import http.client
import json
import os
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
    DEEPSEEK_API_KEY_ENV,
    DEEPSEEK_CACHE_DIR,
    DEEPSEEK_CACHE_ENABLED,
    DEEPSEEK_FALLBACK_API_KEY_ENV,
    DEEPSEEK_HTTP_RETRIES,
    DEEPSEEK_MAX_TOKENS,
    DEEPSEEK_MODEL,
    DEEPSEEK_TEMPERATURE,
    DEEPSEEK_THINKING_TYPE,
    DEEPSEEK_TIMEOUT_SECONDS,
    PROJECT_ROOT,
)
from data.schema import CommentBlock, datetime_to_str
from profiles.user_profile import UserProfile


Transport = Callable[[dict[str, Any]], dict[str, Any]]


class DeepSeekAnthropicDebateClient:
    """使用 DeepSeek 的 Anthropic-compatible Messages API 生成辩论论点。

    API key 默认从环境变量读取：
    1. DEEPSEEK_API_KEY
    2. ANTHROPIC_API_KEY
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEEPSEEK_ANTHROPIC_BASE_URL,
        model: str = DEEPSEEK_MODEL,
        max_tokens: int = DEEPSEEK_MAX_TOKENS,
        temperature: float = DEEPSEEK_TEMPERATURE,
        thinking_type: str = DEEPSEEK_THINKING_TYPE,
        timeout_seconds: float = DEEPSEEK_TIMEOUT_SECONDS,
        http_retries: int = DEEPSEEK_HTTP_RETRIES,
        transport: Transport | None = None,
    ):
        self.api_key = api_key or _load_api_key()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thinking_type = thinking_type
        self.timeout_seconds = timeout_seconds
        self.http_retries = http_retries
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
        """调用 DeepSeek，并把返回 JSON 解析成 Argument。"""
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
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "thinking": {"type": self.thinking_type},
            "system": get_agent_system_prompt(camp, role),
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}],
                }
            ],
        }

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
        cached = _read_cached_response(payload)
        if cached is not None:
            return cached
        response = self._post_messages(payload)
        _write_cached_response(payload, response)
        return response

    def _post_messages(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError(
                f"Missing DeepSeek API key. Set {DEEPSEEK_API_KEY_ENV} "
                f"or {DEEPSEEK_FALLBACK_API_KEY_ENV} in your environment."
            )

        request = urllib.request.Request(
            url=f"{self.base_url}/v1/messages",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": DEEPSEEK_ANTHROPIC_VERSION,
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
                    raise RuntimeError(f"DeepSeek API HTTP {exc.code}: {body}") from exc
                last_error = exc
            except urllib.error.URLError as exc:
                if attempt == self.http_retries:
                    raise RuntimeError(f"DeepSeek API request failed after {self.http_retries} attempts: {exc}") from exc
                last_error = exc
            except (TimeoutError, socket.timeout) as exc:
                if attempt == self.http_retries:
                    raise RuntimeError(f"DeepSeek API timed out after {self.http_retries} attempts: {exc}") from exc
                last_error = exc
            except (http.client.IncompleteRead, http.client.RemoteDisconnected, ConnectionResetError, ssl.SSLError) as exc:
                if attempt == self.http_retries:
                    raise RuntimeError(f"DeepSeek API connection closed after {self.http_retries} attempts: {exc}") from exc
                last_error = exc
            time.sleep(0.8 * attempt)
        else:
            raise RuntimeError(f"DeepSeek API request failed: {last_error}")
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("DeepSeek response must be a JSON object")
        if "error" in parsed:
            raise RuntimeError(f"DeepSeek API error: {parsed['error']}")
        return parsed


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
            "camp": camp,
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
                "camp": item.camp,
                "role": item.role,
                "phase": item.phase,
                "claim": item.claim,
                "confidence": item.confidence,
                "targets": item.targets,
            }
            for item in prior_arguments[-16:]
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _phase_instructions(phase: str) -> str:
    instructions = {
        "initial_argument": (
            "Generate an independent opening argument. Do not target previous arguments."
        ),
        "intra_reflection": (
            "Act as the camp's reflection agent. Read same-camp opening arguments, point out "
            "logical gaps, evidence weakness, and useful shared support. Target same-camp argument ids."
        ),
        "intra_response": (
            "Act as a core camp agent responding to the reflection agent. Revise or strengthen "
            "your camp's position using the reflection critique. You must target the reflection "
            "argument id if it is available."
        ),
        "cross_response": (
            "Read opponent arguments and generate a targeted response/rebuttal. "
            "Choose target ids only from available_target_ids."
        ),
        "counter_reflection": (
            "Act as the camp's reflection agent after opponent attacks. Identify the most damaging "
            "opponent attack, what your camp must repair, and which counterpoint is strongest."
        ),
        "counter_rebuttal": (
            "Answer the opponent's latest cross-response arguments after reading your camp's "
            "counter_reflection. You must target the counter_reflection argument id if it is available, "
            "and may also target opponent attack ids."
        ),
        "reflection_summary": (
            "Summarize this round's debate quality for your camp. Mention strongest support, "
            "weakest evidence, and unresolved uncertainty."
        ),
    }
    return instructions.get(phase, "Follow the requested debate phase and use only provided data.")


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
    raise ValueError(f"DeepSeek response missing text content: {_describe_response_shape(response)}")


def _describe_response_shape(response: dict[str, Any]) -> dict[str, Any]:
    """返回不含正文的响应结构摘要，方便排查 provider 格式差异。"""
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


def _read_cached_response(payload: dict[str, Any]) -> dict[str, Any] | None:
    if not DEEPSEEK_CACHE_ENABLED:
        return None
    path = _cache_path(payload)
    if not path.exists():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return cached if isinstance(cached, dict) else None


def _write_cached_response(payload: dict[str, Any], response: dict[str, Any]) -> None:
    if not DEEPSEEK_CACHE_ENABLED:
        return
    path = _cache_path(payload)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        return


def _cache_path(payload: dict[str, Any]) -> Path:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return Path(DEEPSEEK_CACHE_DIR) / f"{digest}.json"


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
    """固定协议字段，避免 LLM 小幅写错 id 导致下游图构建不稳定。"""
    argument.argument_id = expected_argument_id
    argument.agent_id = expected_agent_id
    argument.camp = camp
    argument.role = role
    argument.round = round_index
    argument.seq = seq
    argument.phase = phase
    if available_target_ids is not None:
        allowed = set(available_target_ids)
        argument.targets = [target_id for target_id in argument.targets if target_id in allowed]
    return argument


def _load_api_key() -> str | None:
    if _looks_like_api_key(DEEPSEEK_API_KEY_ENV):
        return DEEPSEEK_API_KEY_ENV
    if _looks_like_api_key(DEEPSEEK_FALLBACK_API_KEY_ENV):
        return DEEPSEEK_FALLBACK_API_KEY_ENV
    env_key = os.getenv(DEEPSEEK_API_KEY_ENV) or os.getenv(DEEPSEEK_FALLBACK_API_KEY_ENV)
    if env_key:
        return env_key
    return _load_api_key_from_dotenv(PROJECT_ROOT / ".env")


def _looks_like_api_key(value: str | None) -> bool:
    """兼容把真实 key 直接写到 config 的情况；更推荐用 .env 或环境变量。"""
    return bool(value and value.startswith("sk-"))


def _load_api_key_from_dotenv(path: Path) -> str | None:
    """从项目根目录 .env 读取 API key，避免把真实密钥写进 config.py。"""
    if not path.exists():
        return None
    return _load_api_key_from_dotenv_text(path.read_text(encoding="utf-8"))


def _load_api_key_from_dotenv_text(text: str) -> str | None:
    """解析 .env 文本；拆出来方便单元测试不接触真实密钥文件。"""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        supported_names = {
            "DEEPSEEK_API_KEY",
            "ANTHROPIC_API_KEY",
            DEEPSEEK_API_KEY_ENV,
            DEEPSEEK_FALLBACK_API_KEY_ENV,
        }
        if name not in supported_names:
            continue
        return value.strip().strip('"').strip("'")
    return None
