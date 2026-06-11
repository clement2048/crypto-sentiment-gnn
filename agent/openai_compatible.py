"""OpenAI-compatible LLM clients (debate + judge).

合并自原 agent/bailian_client.py 与 judge/bailian_judge_client.py,为
阿里云百炼等所有走 OpenAI Chat Completions 兼容协议的 provider 提供共享
HTTP 层。每个 provider 只需要写一层 ~10 行的"薄配置"。

API key 加载优先级(每一项依次尝试):
1. 调用方传入的 ``api_key`` 参数
2. ``api_key_env``:若值以 ``sk-`` 开头则视为真实 key,否则视为 env 变量名
3. ``fallback_api_key_env``:同上
4. 项目根目录的 ``.env`` 文件

注意:不引入 ``openai`` Python SDK,沿用项目惯例手写 ``urllib``。
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

from agent.anthropic_compatible import (
    _extract_text,  # 复用:fallback 兼容响应也走同一解析
    _load_api_key,  # 复用:env 加载逻辑与 anthropic 一族一致
)
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
    SILICONFLOW_API_KEY_ENV,
    SILICONFLOW_CACHE_DIR,
    SILICONFLOW_CACHE_ENABLED,
    SILICONFLOW_ENABLE_THINKING,
    SILICONFLOW_HTTP_RETRIES,
    SILICONFLOW_MAX_TOKENS,
    SILICONFLOW_MODEL,
    SILICONFLOW_OPENAI_BASE_URL,
    SILICONFLOW_TEMPERATURE,
    SILICONFLOW_TIMEOUT_SECONDS,
)
from data.schema import CommentBlock
from profiles.user_profile import UserProfile

# Judge 端 import 故意留在函数内部 lazy import,避免与 judge/__init__.py
# 形成循环。详见 agent/anthropic_compatible.py 顶部同名注释。


Transport = Callable[[dict[str, Any]], dict[str, Any]]
DEFAULT_BAILIAN_API_KEY_ENV_NAME = "DASHSCOPE_API_KEY"
DEFAULT_SILICONFLOW_API_KEY_ENV_NAME = "SILICONFLOW_API_KEY"


# =============================================================================
# Debate 基类
# =============================================================================


class OpenAICompatibleDebateClient:
    """所有 OpenAI Chat Completions 兼容 provider 的共享 HTTP 层。

    子类(具体 provider)只需要传 ``base_url`` / ``model`` / ``api_key_env`` /
    ``cache_dir`` 等常量。
    """

    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        model: str,
        max_tokens: int,
        temperature: float,
        enable_thinking: bool,
        timeout_seconds: float,
        http_retries: int,
        cache_enabled: bool,
        cache_dir: str | Path,
        api_key_env: str,
        fallback_api_key_env: str | None,
        transport: Transport | None,
    ):
        self.api_key = api_key or _load_api_key(api_key_env, fallback_api_key_env)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.timeout_seconds = timeout_seconds
        self.http_retries = http_retries
        self.cache_enabled = cache_enabled
        self.cache_dir = Path(cache_dir)
        self.api_key_env = api_key_env
        self.fallback_api_key_env = fallback_api_key_env
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
        """调用 OpenAI-compatible API,把响应解析成 Argument。"""
        # prompt 拼装复用 anthropic_compatible 的 helper(协议无关)
        from agent.anthropic_compatible import _build_user_prompt, _normalize_argument_metadata

        expected_argument_id = f"{block.block_id}:r{round_index}:s{seq}:{camp}"
        expected_agent_id = f"{camp}_{role}"
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
        payload = self._build_payload(
            user_prompt=user_prompt,
            camp=camp,
            role=role,
        )
        response = self._send_payload(payload)
        text = _extract_openai_text(response)
        try:
            argument = parse_argument_json(text)
        except (ValueError, json.JSONDecodeError):
            repair_payload = self._build_repair_payload(
                original_payload=payload,
                invalid_text=text,
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

    def _build_payload(self, user_prompt: str, camp: Camp, role: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "enable_thinking": self.enable_thinking,
            "messages": [
                {"role": "system", "content": get_agent_system_prompt(camp, role)},
                {"role": "user", "content": user_prompt},
            ],
        }
        return payload

    def _build_repair_payload(
        self,
        original_payload: dict[str, Any],
        invalid_text: str,
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
        cached = _read_cached_response(self.cache_dir, self.cache_enabled, payload)
        if cached is not None:
            return cached
        response = self._post_chat_completions(payload)
        _write_cached_response(self.cache_dir, self.cache_enabled, payload, response)
        return response

    def _post_chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError(
                f"Missing OpenAI-compatible API key. "
                f"Set {self.api_key_env}"
                + (f" or {self.fallback_api_key_env}" if self.fallback_api_key_env else "")
                + " in your environment or .env file."
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
                    raise RuntimeError(f"OpenAI-compatible API HTTP {exc.code}: {body}") from exc
                last_error = exc
            except urllib.error.URLError as exc:
                if attempt == self.http_retries:
                    raise RuntimeError(
                        f"OpenAI-compatible API request failed after {self.http_retries} attempts: {exc}"
                    ) from exc
                last_error = exc
            except (TimeoutError, socket.timeout) as exc:
                if attempt == self.http_retries:
                    raise RuntimeError(
                        f"OpenAI-compatible API timed out after {self.http_retries} attempts: {exc}"
                    ) from exc
                last_error = exc
            except (http.client.IncompleteRead, http.client.RemoteDisconnected, ConnectionResetError, ssl.SSLError) as exc:
                if attempt == self.http_retries:
                    raise RuntimeError(
                        f"OpenAI-compatible API connection closed after {self.http_retries} attempts: {exc}"
                    ) from exc
                last_error = exc
            time.sleep(0.8 * attempt)
        else:
            raise RuntimeError(f"OpenAI-compatible API request failed: {last_error}")
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("OpenAI-compatible response must be a JSON object")
        if "error" in parsed:
            raise RuntimeError(f"OpenAI-compatible API error: {parsed['error']}")
        return parsed


# =============================================================================
# Judge 基类
# =============================================================================


class OpenAICompatibleJudgeClient:
    """所有 OpenAI 兼容 provider 的共享 judge 层。"""

    def __init__(self, http_client: OpenAICompatibleDebateClient):
        self.http = http_client

    def judge(
        self,
        transcript: DebateTranscript,
        model_summary: ModelOutputSummary,
        graph: HeteroGraph,
    ) -> JudgeOutput:
        # lazy import 避免循环,见 agent/anthropic_compatible.py 注释
        from judge.consistency import check_judge_consistency
        from judge.judge_parser import parse_judge_json

        payload = self._build_payload(transcript, model_summary, graph)
        response = self.http._send_payload(payload)
        text = _extract_openai_text(response)
        try:
            output = parse_judge_json(text)
        except (ValueError, json.JSONDecodeError):
            repair_payload = self._build_repair_payload(payload, text)
            repair_response = self.http._send_payload(repair_payload)
            output = parse_judge_json(_extract_openai_text(repair_response))
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

        return {
            "model": self.http.model,
            "max_tokens": self.http.max_tokens,
            "temperature": self.http.temperature,
            "enable_thinking": self.http.enable_thinking,
            "messages": [
                {"role": "system", "content": JUDGE_OUTPUT_SCHEMA_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        _judge_input(transcript, model_summary, graph),
                        ensure_ascii=False,
                        indent=2,
                    ),
                },
            ],
        }

    def _build_repair_payload(self, original_payload: dict[str, Any], invalid_text: str) -> dict[str, Any]:
        payload = copy.deepcopy(original_payload)
        payload["temperature"] = 0.0
        payload["messages"] = [
            *payload["messages"],
            {"role": "assistant", "content": invalid_text[:4000]},
            {
                "role": "user",
                "content": (
                    "The previous response was not a valid JudgeOutput JSON object. "
                    "Repair it into exactly one valid JSON object with verdict, confidence, "
                    "report, score_vector, and consistency_flags. Return only JSON."
                ),
            },
        ]
        return payload


# =============================================================================
# Provider 薄配置:Bailian(阿里云百炼)
# =============================================================================


class BailianOpenAICompatibleDebateClient(OpenAICompatibleDebateClient):
    """阿里云百炼 OpenAI 兼容接口辩论 client(薄配置)。

    默认读取环境变量 ``DASHSCOPE_API_KEY``,默认模型为 ``deepseek-v4-flash``。
    """

    def __init__(
        self,
        api_key: str | None = None,
        transport: Transport | None = None,
        **overrides: Any,
    ):
        super().__init__(
            api_key=api_key,
            base_url=BAILIAN_OPENAI_BASE_URL,
            model=BAILIAN_MODEL,
            max_tokens=BAILIAN_MAX_TOKENS,
            temperature=BAILIAN_TEMPERATURE,
            enable_thinking=BAILIAN_ENABLE_THINKING,
            timeout_seconds=BAILIAN_TIMEOUT_SECONDS,
            http_retries=BAILIAN_HTTP_RETRIES,
            cache_enabled=BAILIAN_CACHE_ENABLED,
            cache_dir=BAILIAN_CACHE_DIR,
            api_key_env=BAILIAN_API_KEY_ENV,
            fallback_api_key_env=DEFAULT_BAILIAN_API_KEY_ENV_NAME,
            transport=transport,
            **overrides,
        )


class BailianJudgeClient(OpenAICompatibleJudgeClient):
    """阿里云百炼 OpenAI 兼容接口法官 client(薄配置)。"""

    def __init__(
        self,
        transport: Transport | None = None,
        **overrides: Any,
    ):
        super().__init__(
            http_client=BailianOpenAICompatibleDebateClient(transport=transport, **overrides)
        )


# =============================================================================
# Provider 薄配置:SiliconFlow(硅基流动)
# =============================================================================


class SiliconFlowOpenAICompatibleDebateClient(OpenAICompatibleDebateClient):
    """硅基流动 OpenAI 兼容接口辩论 client(薄配置)。"""

    def __init__(
        self,
        api_key: str | None = None,
        transport: Transport | None = None,
        **overrides: Any,
    ):
        super().__init__(
            api_key=api_key,
            base_url=SILICONFLOW_OPENAI_BASE_URL,
            model=SILICONFLOW_MODEL,
            max_tokens=SILICONFLOW_MAX_TOKENS,
            temperature=SILICONFLOW_TEMPERATURE,
            enable_thinking=SILICONFLOW_ENABLE_THINKING,
            timeout_seconds=SILICONFLOW_TIMEOUT_SECONDS,
            http_retries=SILICONFLOW_HTTP_RETRIES,
            cache_enabled=SILICONFLOW_CACHE_ENABLED,
            cache_dir=SILICONFLOW_CACHE_DIR,
            api_key_env=SILICONFLOW_API_KEY_ENV,
            fallback_api_key_env=DEFAULT_SILICONFLOW_API_KEY_ENV_NAME,
            transport=transport,
            **overrides,
        )


class SiliconFlowJudgeClient(OpenAICompatibleJudgeClient):
    """硅基流动 OpenAI 兼容接口法官 client(薄配置)。"""

    def __init__(
        self,
        transport: Transport | None = None,
        **overrides: Any,
    ):
        super().__init__(
            http_client=SiliconFlowOpenAICompatibleDebateClient(transport=transport, **overrides)
        )


# =============================================================================
# 模块级 helper(OpenAI 响应解析、缓存)
# =============================================================================


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
    raise ValueError(f"OpenAI-compatible response missing choices[0].message.content: {sorted(response.keys())}")


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
        "model_summary": model_summary.to_dict(),
        "debate_arguments": [
            {
                "argument_id": item.argument_id,
                "agent_id": item.agent_id,
                "camp": item.camp,
                "role": item.role,
                "phase": item.phase,
                "claim": item.claim,
                "evidence": [evidence.to_dict() for evidence in item.evidence],
                "confidence": item.confidence,
                "targets": item.targets,
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
                    "attrs": edge.attrs,
                }
                for edge in graph.edges
            ],
        },
        "rules": [
            "Do not use ground-truth labels, p1, or future prices.",
            "Base the decision only on debate logic and model/ODE summary.",
            "If debate and model disagree, explain the disagreement and lower confidence.",
        ],
    }
