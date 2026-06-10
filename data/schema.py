"""Data contracts for the v2 comment-block pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from config import TIMESTAMP_MILLISECONDS_THRESHOLD


DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def parse_datetime(value: Any) -> datetime | None:
    """Parse known timestamp formats from the source JSONL."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        # Binance source commonly includes millisecond epoch fields elsewhere.
        timestamp = float(value)
        if timestamp > TIMESTAMP_MILLISECONDS_THRESHOLD:
            timestamp /= 1000.0
        return datetime.fromtimestamp(timestamp)

    text = str(value).strip()
    if not text:
        return None
    text = text.replace("T", " ").replace("Z", "")
    for fmt in (DATETIME_FORMAT, "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def datetime_to_str(value: datetime | None) -> str | None:
    """Serialize a datetime using the source-friendly format."""
    if value is None:
        return None
    return value.strftime(DATETIME_FORMAT)


@dataclass
class RawComment:
    comment_id: str 
    original_comment_id: str
    author: str
    text: str
    post_time: datetime | None
    replies: list["RawComment"] = field(default_factory=list)
    t0: datetime | None = None
    t_window: str | None = None
    p0: float | None = None
    p1: float | None = None
    label: int | None = None
    comment_error: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RawComment":
        replies = [cls.from_dict(item) for item in (data.get("replies") or [])]
        return cls(
            comment_id=str(data.get("comment_id") or ""),
            original_comment_id=str(data.get("original_comment_id") or ""),
            author=str(data.get("author") or ""),
            text=str(data.get("text") or ""),
            post_time=parse_datetime(data.get("post_time")),
            replies=replies,
            t0=parse_datetime(data.get("t0")),
            t_window=str(data.get("t_window") or "") or None,
            p0=_to_float(data.get("p0")),
            p1=_to_float(data.get("p1")),
            label=_to_int(data.get("label")),
            comment_error=str(data.get("comment_error") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "comment_id": self.comment_id,
            "original_comment_id": self.original_comment_id,
            "author": self.author,
            "text": self.text,
            "post_time": datetime_to_str(self.post_time),
            "replies": [reply.to_dict() for reply in self.replies],
            "t0": datetime_to_str(self.t0),
            "t_window": self.t_window,
            "p0": self.p0,
            "p1": self.p1,
            "label": self.label,
            "comment_error": self.comment_error,
        }


@dataclass
class PostRecord:
    post_id: str
    post_content: str
    post_time: datetime | None
    products: list[str]
    first_product: str | None
    market_type: str | None
    comments: list[RawComment]
    post_author: str = ""
    label_error: str = ""
    source_file: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], source_file: str | None = None) -> "PostRecord":
        products = data.get("products") or []
        if not isinstance(products, list):
            products = [str(products)]
        comments = [RawComment.from_dict(item) for item in (data.get("comments") or [])]
        return cls(
            post_id=str(data.get("post_id") or data.get("postId") or ""),
            post_content=str(data.get("post_content") or data.get("content") or ""),
            post_time=parse_datetime(data.get("post_time") or data.get("timestamp")),
            products=[str(item) for item in products],
            first_product=_optional_str(data.get("first_product")),
            market_type=_optional_str(data.get("market_type")),
            comments=comments,
            post_author=str(data.get("post_author") or data.get("author") or ""),
            label_error=str(data.get("label_error") or ""),
            source_file=source_file,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "post_id": self.post_id,
            "post_content": self.post_content,
            "post_time": datetime_to_str(self.post_time),
            "products": self.products,
            "first_product": self.first_product,
            "market_type": self.market_type,
            "comments": [comment.to_dict() for comment in self.comments],
            "post_author": self.post_author,
            "label_error": self.label_error,
            "source_file": self.source_file,
        }


@dataclass
class CommentBlock:
    block_id: str       # 固定为 post_id:comment_id，后续图、辩论、输出文件都用这个 ID 对齐。
    post_id: str        # 原 post_id，方便后续分析时按帖子聚合样本。注意一个帖子可能有多个 block（多个根评论），但一个 block 只对应一个根评论。
    post_content: str   # 帖子正文，后续可以用来做文本特征或直接输入模型。
    products: list[str] # 帖子里提到的所有代币币种
    root_comment: RawComment
    replies: list[RawComment]
    t0: datetime        # 根评论发布时间
    t_window: str       # 根评论发布后经过的时间窗口
    p0: float           # 根评论发布时的价格
    p1: float           # 根评论发布后 t_window 时间点的价格
    label: int          # 根评论的标签，1/0 分别表示正/负样本
    product: str | None # 根评论提到的代币币种
    market_type: str | None = None
    post_time: datetime | None = None

    def participants(self) -> list[str]:
        authors: list[str] = []
        for comment in [self.root_comment, *flatten_replies(self.replies)]:
            if comment.author and comment.author not in authors:
                authors.append(comment.author)
        return authors

    def all_comments(self) -> list[RawComment]:
        return [self.root_comment, *flatten_replies(self.replies)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "post_id": self.post_id,
            "post_content": self.post_content,
            "products": self.products,
            "root_comment": self.root_comment.to_dict(),
            "replies": [reply.to_dict() for reply in self.replies],
            "t0": datetime_to_str(self.t0),
            "t_window": self.t_window,
            "p0": self.p0,
            "p1": self.p1,
            "label": self.label,
            "product": self.product,
            "market_type": self.market_type,
            "post_time": datetime_to_str(self.post_time),
        }


@dataclass
class FilterIssue:
    post_id: str
    comment_id: str | None
    reason: str
    detail: str = ""
    source_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "post_id": self.post_id,
            "comment_id": self.comment_id,
            "reason": self.reason,
            "detail": self.detail,
            "source_file": self.source_file,
        }


def flatten_replies(replies: list[RawComment]) -> list[RawComment]:
    flat: list[RawComment] = []
    for reply in replies:
        flat.append(reply)
        flat.extend(flatten_replies(reply.replies))
    return flat


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None



