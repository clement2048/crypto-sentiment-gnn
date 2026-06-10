"""JSONL loader for post records."""

from __future__ import annotations

import glob
import json
from pathlib import Path

from data.schema import PostRecord


# -----------------------------
# 已检查没有问题
# -----------------------------

def load_posts(path_or_glob: str | Path) -> list[PostRecord]:
    """Load UTF-8 JSONL posts from a file, directory, or glob pattern."""
    paths = _resolve_input_paths(path_or_glob)  # 解析输入路径，支持单文件、目录和 glob 模式。
    posts: list[PostRecord] = []   
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
                posts.append(PostRecord.from_dict(data, source_file=str(path)))
    return posts


def _resolve_input_paths(path_or_glob: str | Path) -> list[Path]:
    value = str(path_or_glob)
    path = Path(value)
    # 如果是目录，读取目录下所有 JSONL 文件；如果是 glob 模式，解析出所有匹配的文件；否则当作单个文件路径。
    if path.is_dir():
        paths = sorted(path.glob("*.jsonl"))
    elif any(char in value for char in "*?[]"):
        paths = sorted(Path(item) for item in glob.glob(value))
    else:
        paths = [path]
    # 检验是否有存在的文件路径，避免后续打开文件时才发现问题。注意这里不允许路径不存在或是目录。
    existing = [item for item in paths if item.exists() and item.is_file()]
    if not existing:
        raise FileNotFoundError(f"No JSONL files found for input: {path_or_glob}")
    return existing



