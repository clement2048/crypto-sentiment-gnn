"""LLM Agent — 调用 DeepSeek Anthropic 兼容 API 分析用户情绪。"""

import json
import os
import re
import time
from dataclasses import dataclass

from anthropic import Anthropic
from anthropic import APIStatusError

from agent.user_profile import UserProfile
from config import (
    LLM_MODEL,
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_API_KEY_ENV,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT,
)
from data_loader.loader import CommentNode
from logger import logger


@dataclass
class AgentResult:
    """单个 Agent 的分析结果。"""
    author: str
    comment_text: str
    sentiment_label: str        # "bullish" | "bearish" | "neutral"
    confidence: float           # 0.0 ~ 1.0
    sentiment_vector: list[float]  # [bullish, bearish, neutral, confidence]
    reason: str                 # 推理简述
    matched_keywords: list[str]


SYSTEM_PROMPT = """分析加密货币评论情绪。输出纯JSON，不要markdown包裹。

规则：看涨→bullish 看跌→bearish 提问/中性→neutral
confidence: 0.8+确定 0.5-0.8可能 <0.5不确定

格式：{"sentiment":"bullish|bearish|neutral","confidence":0.5,"reason":"简短中文"}"""


def _build_client() -> Anthropic:
    """创建 Anthropic 客户端，指向 DeepSeek 兼容接口。"""
    api_key = os.environ.get(LLM_API_KEY_ENV, LLM_API_KEY_ENV)
    if not api_key:
        raise ValueError(f"环境变量 {LLM_API_KEY_ENV} 未设置")

    return Anthropic(
        api_key=api_key,
        base_url=LLM_BASE_URL,
        timeout=LLM_TIMEOUT,
        max_retries=LLM_MAX_RETRIES,
    )


def _extract_text_from_response(response) -> str:
    """从响应中提取文本：优先 TextBlock，其次 ThinkingBlock。"""
    for block in response.content:
        if hasattr(block, "text") and getattr(block, "text", "").strip():
            return block.text.strip()
    for block in response.content:
        if hasattr(block, "thinking") and getattr(block, "thinking", "").strip():
            return block.thinking.strip()
    return ""


def _parse_json_response(response_text: str) -> dict:
    """从文本中提取情绪分析 JSON，支持截断修复。

    抛出:
        json.JSONDecodeError: 无法提取有效 JSON
    """
    # 去掉 markdown 代码块包裹
    text = response_text
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        text = text.strip()

    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 正则提取 JSON 对象
    match = re.search(r'\{[^{}]*"sentiment"[^{}]*\}', text)
    if not match:
        raise json.JSONDecodeError("未找到 sentiment JSON", text, 0)

    raw = match.group()
    # 修复截断的 JSON
    raw = re.sub(r',\s*"[^"]*"\s*$', '', raw)   # 截断的键值对
    raw = re.sub(r',\s*"[^"]*$', '', raw)        # 截断的字符串
    raw = re.sub(r',\s*$', '', raw)              # 尾部逗号
    raw = raw.rstrip()
    if not raw.endswith('}'):
        raw += '}'

    return json.loads(raw)


def analyze_comment(
    node: CommentNode,
    post_content: str,
    parent_text: str | None = None,
    user_profile: UserProfile | None = None,
) -> AgentResult:
    """调用 LLM 分析单条评论的用户情绪。

    异常:
        ValueError: API key 未设置 / LLM 返回空响应
        Exception: API 调用失败 / JSON 解析失败
    """
    text = node.text

    # 构建用户消息
    user_parts = [f"新闻内容：{post_content}"]
    if parent_text:
        user_parts.append(f"被回复的评论：{parent_text}")
    user_parts.append(f"用户 {node.author} 的评论：{text}")
    if user_profile and user_profile.total_comments > 0:
        user_parts.append(
            f"用户画像：历史{user_profile.total_comments}条评论，"
            f"立场偏差={user_profile.stance_bias:.2f}"
            f"（-1=极度看跌 +1=极度看涨），"
            f"一致性={user_profile.consistency:.2f}"
        )

    user_message = "\n".join(user_parts)
    client = _build_client()

    # 调用 API（含重试：无 text 响应时重试最多 3 次）
    max_attempts = 3
    last_error = None

    for attempt in range(max_attempts):
        try:
            logger.debug("LLM 请求: model=%s, user=%s, attempt=%d", LLM_MODEL, node.author, attempt + 1)
            response = client.messages.create(
                model=LLM_MODEL,
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )

            response_text = _extract_text_from_response(response)
            if not response_text:
                block_types = [getattr(b, "type", "?") for b in response.content]
                logger.warning("LLM 无文本响应 (attempt %d/%d), blocks: %s", attempt + 1, max_attempts, block_types)
                last_error = ValueError(f"无文本响应: {block_types}")
                time.sleep(1.0 * (attempt + 1))
                continue

            result = _parse_json_response(response_text)
            break  # 成功

        except (json.JSONDecodeError, APIStatusError) as e:
            logger.warning("LLM 调用失败 (attempt %d/%d): %s", attempt + 1, max_attempts, e)
            last_error = e
            time.sleep(1.0 * (attempt + 1))
    else:
        # 所有重试都失败
        raise last_error  # type: ignore[misc]

    sentiment_label = result.get("sentiment", "neutral")
    confidence = float(result.get("confidence", 0.0))
    reason = result.get("reason", "")

    # 验证 label 合法
    if sentiment_label not in ("bullish", "bearish", "neutral"):
        sentiment_label = "neutral"
        confidence = 0.0

    # 限制 confidence 范围
    confidence = max(0.0, min(1.0, confidence))

    # 转为向量
    if sentiment_label == "bullish":
        sentiment_vector = [confidence, 0.0, 0.0, confidence]
    elif sentiment_label == "bearish":
        sentiment_vector = [0.0, confidence, 0.0, confidence]
    else:
        sentiment_vector = [0.0, 0.0, 1.0, confidence]

    return AgentResult(
        author=node.author,
        comment_text=text,
        sentiment_label=sentiment_label,
        confidence=confidence,
        sentiment_vector=sentiment_vector,
        reason=f"[LLM] {reason}",
        matched_keywords=[],
    )
