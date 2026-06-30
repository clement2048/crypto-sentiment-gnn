"""数据集统计工具：标签分布、按 first_product / 月份 / t_window 切分。

默认输入是 ``dataset/final.jsonl``。可以通过 ``--input`` 切换输入路径。

用法：

    python scripts/dataset_stats.py                          # 默认 dataset/final.jsonl
    python scripts/dataset_stats.py --input dataset/final.jsonl
    python scripts/dataset_stats.py --limit 50
    python scripts/dataset_stats.py --json outputs/dataset_stats.json

也可作为模块导入：

    from scripts.dataset_stats import compute_dataset_stats
    stats = compute_dataset_stats("dataset/final.jsonl")
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import build_comment_blocks, load_posts  # noqa: E402


DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "final.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "dataset_stats.json"


def _per_product(blocks, label_counter: Counter) -> dict[str, dict[str, int]]:
    product_counter: Counter = Counter()
    product_label_counter: dict[str, Counter] = {}
    for block in blocks:
        product = block.product or "(unknown)"
        product_counter[product] += 1
        bucket = product_label_counter.setdefault(product, Counter())
        bucket[int(block.label)] += 1
    out: dict[str, dict[str, int]] = {}
    for product, total in sorted(product_counter.items(), key=lambda item: -item[1]):
        bull = product_label_counter[product].get(1, 0)
        bear = product_label_counter[product].get(-1, 0)
        out[product] = {
            "blocks": int(total),
            "bullish": int(bull),
            "bearish": int(bear),
        }
    return out


def _per_month(blocks) -> dict[str, dict[str, int]]:
    month_counter: Counter = Counter()
    month_label_counter: dict[str, Counter] = {}
    for block in blocks:
        if block.t0 is None:
            month = "(unknown)"
        else:
            month = block.t0.strftime("%Y-%m")
        month_counter[month] += 1
        bucket = month_label_counter.setdefault(month, Counter())
        bucket[int(block.label)] += 1
    out: dict[str, dict[str, int]] = {}
    for month, total in sorted(month_counter.items()):
        bull = month_label_counter[month].get(1, 0)
        bear = month_label_counter[month].get(-1, 0)
        out[month] = {
            "blocks": int(total),
            "bullish": int(bull),
            "bearish": int(bear),
        }
    return out


def _per_t_window(blocks) -> dict[str, dict[str, int]]:
    window_counter: Counter = Counter()
    window_label_counter: dict[str, Counter] = {}
    for block in blocks:
        window = block.t_window or "(unknown)"
        window_counter[window] += 1
        bucket = window_label_counter.setdefault(window, Counter())
        bucket[int(block.label)] += 1
    out: dict[str, dict[str, int]] = {}
    for window, total in sorted(window_counter.items(), key=lambda item: (-item[1], item[0])):
        bull = window_label_counter[window].get(1, 0)
        bear = window_label_counter[window].get(-1, 0)
        out[window] = {
            "blocks": int(total),
            "bullish": int(bull),
            "bearish": int(bear),
        }
    return out


def compute_dataset_stats(input_path: str | Path = DEFAULT_INPUT, limit: int | None = None) -> dict:
    """加载 JSONL，构建 CommentBlock，返回一份完整统计字典。

    不读 `block.label` / `block.p0` / `block.p1` 之外的字段做派生指标；
    只对 `label` 做分布统计。
    """
    input_path = Path(input_path)
    posts = load_posts(input_path)
    if limit is not None:
        posts = posts[:limit]
    blocks, issues = build_comment_blocks(posts)

    label_counter: Counter = Counter()
    for block in blocks:
        label_counter[int(block.label)] += 1
    bullish = int(label_counter.get(1, 0))
    bearish = int(label_counter.get(-1, 0))

    post_err_counter: Counter = Counter()
    for post in posts:
        if post.label_error:
            post_err_counter[post.label_error] += 1

    issue_reason_counter: Counter = Counter()
    for issue in issues:
        issue_reason_counter[issue.reason] += 1

    return {
        "input": str(input_path),
        "limit": limit,
        "posts": len(posts),
        "comment_blocks": len(blocks),
        "filter_issues": len(issues),
        "filter_issue_reasons": dict(sorted(issue_reason_counter.items())),
        "posts_with_label_error": dict(sorted(post_err_counter.items())),
        "labels": {
            "bullish_1": bullish,
            "bearish_minus_1": bearish,
            "bullish_ratio": bullish / (bullish + bearish) if (bullish + bearish) else 0.0,
            "bearish_ratio": bearish / (bullish + bearish) if (bullish + bearish) else 0.0,
        },
        "by_product": _per_product(blocks, label_counter),
        "by_month": _per_month(blocks),
        "by_t_window": _per_t_window(blocks),
    }


def _print_stats(stats: dict) -> None:
    labels = stats["labels"]
    print(f"=== Dataset stats: {stats['input']} (limit={stats['limit']}) ===")
    print(
        f"posts={stats['posts']}  CommentBlocks={stats['comment_blocks']}  "
        f"filter_issues={stats['filter_issues']}"
    )
    print(
        f"label=1  (bullish)  : {labels['bullish_1']:>5d}  ({labels['bullish_ratio']:.1%})"
    )
    print(
        f"label=-1 (bearish)  : {labels['bearish_minus_1']:>5d}  ({labels['bearish_ratio']:.1%})"
    )
    if stats["posts_with_label_error"]:
        print(f"posts with label_error: {stats['posts_with_label_error']}")
    if stats["filter_issue_reasons"]:
        print(f"filter issue reasons  : {stats['filter_issue_reasons']}")

    print("\n--- by first_product ---")
    for product, item in stats["by_product"].items():
        total = item["blocks"]
        bull = item["bullish"]
        bear = item["bearish"]
        print(f"  {product:<12s} blocks={total:>5d}  bull={bull:>4d}  bear={bear:>4d}")

    print("\n--- by month ---")
    for month, item in stats["by_month"].items():
        total = item["blocks"]
        bull = item["bullish"]
        bear = item["bearish"]
        print(f"  {month:<10s} blocks={total:>5d}  bull={bull:>4d}  bear={bear:>4d}")

    print("\n--- by t_window ---")
    for window, item in stats["by_t_window"].items():
        total = item["blocks"]
        bull = item["bullish"]
        bear = item["bearish"]
        print(f"  {window:<8s} blocks={total:>5d}  bull={bull:>4d}  bear={bear:>4d}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="统计 dataset/final.jsonl 的 bullish / bearish 标签分布与子集切分。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--limit", type=int, default=None, help="只取前 N 条 post 做烟测。")
    parser.add_argument("--json", dest="json_path", default=None, help="把结果也写到这份 JSON。")
    args = parser.parse_args()

    stats = compute_dataset_stats(args.input, limit=args.limit)
    _print_stats(stats)

    if args.json_path:
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote JSON: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
