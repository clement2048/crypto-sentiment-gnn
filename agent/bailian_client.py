"""Aliyun Bailian OpenAI-compatible debate client."""

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

from agent.deepseek_client import _build_user_prompt
from agent.output_parser import parse_argument_json
from agent.prompts import get_agent_system_prompt
from agent.schema import Argument, Camp
from config import (
    BAILIAN_API_KEY_ENV,
    BAILIAN_CACHE_DIR,
    BAILIAN_CACHE_ENABLED,
    BAILIAN_ENABLE_THINKING,
    BAILIAN_HTTP_RETRIES,
    BAILIAN_MAX_TOKENS,
    BAILIAN_MODEL,
    BAILIAN_OPENAI_BASE_URL,
    BAILIAN_TEMPERATURE,
    BAILIAN_TIMEOUT_SECONDS,
    PROJECT_ROOT,
)
from data.schema import CommentBlock
from profiles.user_profile import UserProfile


Transport = Callable[[dict[str, Any]], dict[str, Any]]
DEFAULT_BAILIAN_API_KEY_ENV_NAME = "DASHSCOPE_API_KEY"


class BailianOpenAICompatibleDebateClient:
    """通过阿里云百炼 OpenAI 兼容接口生成结构化辩论论点。

    默认读取环境变量 `DASHSCOPE_API_KEY`，默认模型为 `deepseek-v4-flash`。
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = BAILIAN_OPENAI_BASE_URL,
        model: str = BAILIAN_MODEL,
        max_tokens: int = BAILIAN_MAX_TOKENS,
        temperature: float = BAILIAN_TEMPERATURE,
        enable_thinking: bool = BAILIAN_ENABLE_THINKING,
        timeout_seconds: float = BAILIAN_TIMEOUT_SECONDS,
        http_retries: int = BAILIAN_HTTP_RETRIES,
        transport: Transport | None = None,
    ):
        self.api_key = api_key or _load_api_key()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_thinking = enable_thinking
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
        """调用百炼，并把 OpenAI-compatible 响应解析成 Argument。"""
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
        text = _extract_openai_text(response)
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
            argument = parse_argument_json(_extract_openai_text(repair_response))
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
            "enable_thinking": self.enable_thinking,
            "messages": [
                {"role": "system", "content": get_agent_system_prompt(camp, role)},
                {"role": "user", "content": user_prompt},
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
        payload = copy.deepcopy(original_payload)
        payload["temperature"] = 0.0
        payload["messages"] = [
            *payload["messages"],
            {"role": "assistant", "content": invalid_text[:4000]},
            {
                "role": "user",
                "content": (
                    "The previous assistant response was invalid JSON. "
                    "Repair it into exactly one valid JSON object matching the Argument schema. "
                    "Return only JSON, with no markdown and no explanation. "
                    f"Use argument_id={expected_argument_id}, agent_id={expected_agent_id}, "
                    f"camp={camp}, role={role}, round={round_index}, seq={seq}, phase={phase}."
                ),
            },
        ]
        return payload

    def _send_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.transport:
            return self.transport(payload)
        cached = _read_cached_response(payload)
        if cached is not None:
            return cached
        response = self._post_chat_completions(payload)
        _write_cached_response(payload, response)
        return response

    def _post_chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError(
                "Missing Bailian API key. Set DASHSCOPE_API_KEY in your environment "
                "or configure BAILIAN_API_KEY_ENV with the key value."
            )

        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {self.api_key}",
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
                    raise RuntimeError(f"Bailian API HTTP {exc.code}: {body}") from exc
                last_error = exc
            except urllib.error.URLError as exc:
                if attempt == self.http_retries:
                    raise RuntimeError(f"Bailian API request failed after {self.http_retries} attempts: {exc}") from exc
                last_error = exc
            except (TimeoutError, socket.timeout) as exc:
                if attempt == self.http_retries:
                    raise RuntimeError(f"Bailian API timed out after {self.http_retries} attempts: {exc}") from exc
                last_error = exc
            except (http.client.IncompleteRead, http.client.RemoteDisconnected, ConnectionResetError, ssl.SSLError) as exc:
                if attempt == self.http_retries:
                    raise RuntimeError(f"Bailian API connection closed after {self.http_retries} attempts: {exc}") from exc
                last_error = exc
            time.sleep(0.8 * attempt)
        else:
            raise RuntimeError(f"Bailian API request failed: {last_error}")
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Bailian response must be a JSON object")
        if "error" in parsed:
            raise RuntimeError(f"Bailian API error: {parsed['error']}")
        return parsed


def _extract_openai_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(first.get("text"), str):
                return first["text"]
    raise ValueError(f"Bailian response missing choices[0].message.content: {sorted(response.keys())}")


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
    if _looks_like_api_key(BAILIAN_API_KEY_ENV):
        return BAILIAN_API_KEY_ENV
    env_key = os.getenv(BAILIAN_API_KEY_ENV)
    if env_key:
        return env_key
    env_key = os.getenv(DEFAULT_BAILIAN_API_KEY_ENV_NAME)
    if env_key:
        return env_key
    return _load_api_key_from_dotenv(PROJECT_ROOT / ".env")


def _load_api_key_from_dotenv(path: Path) -> str | None:
    if not path.exists():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() in {BAILIAN_API_KEY_ENV, DEFAULT_BAILIAN_API_KEY_ENV_NAME}:
            return value.strip().strip('"').strip("'")
    return None


def _looks_like_api_key(value: str | None) -> bool:
    return bool(value and value.startswith("sk-"))


def _read_cached_response(payload: dict[str, Any]) -> dict[str, Any] | None:
    if not BAILIAN_CACHE_ENABLED:
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
    if not BAILIAN_CACHE_ENABLED:
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
    return Path(BAILIAN_CACHE_DIR) / f"{digest}.json"
