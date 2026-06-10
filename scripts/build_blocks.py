"""Build v2 CommentBlock samples from source JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from config import DEFAULT_INPUT_PATH, PRINT_SAMPLES
from data import build_comment_blocks, load_posts, temporal_split_blocks
from profiles import ProfileStore


DEFAULT_INPUT = DEFAULT_INPUT_PATH


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Build CommentBlock samples from JSONL data.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="JSONL file, directory, or glob pattern.")
    parser.add_argument("--limit", type=int, default=None, help="Limit loaded posts before block building.")
    parser.add_argument("--print-samples", type=int, default=PRINT_SAMPLES, help="Number of block samples to print.")
    parser.add_argument("--output-jsonl", type=str, default=None, help="Optional path to write blocks JSONL.")
    args = parser.parse_args()

    posts = load_posts(args.input)
    if args.limit is not None:
        posts = posts[: args.limit]

    blocks, issues = build_comment_blocks(posts)
    splits = temporal_split_blocks(blocks)
    profile_store = ProfileStore.from_blocks(blocks)

    print(f"Loaded posts: {len(posts)}")
    print(f"Built blocks: {len(blocks)}")
    print(f"Filter issues: {len(issues)}")
    if issues:
        counts = Counter(issue.reason for issue in issues)
        print("Filter issue counts:")
        for reason, count in sorted(counts.items()):
            print(f"  {reason}: {count}")

    print(
        "Temporal split: "
        f"train={len(splits.train)} val={len(splits.val)} test={len(splits.test)}"
    )

    sample_count = min(args.print_samples, len(blocks))
    if sample_count:
        print(f"First {sample_count} CommentBlock samples:")
    for block in blocks[:sample_count]:
        profiles = profile_store.get_profiles_for_block(block)
        root_text = block.root_comment.text.replace("\n", " ")[:120]
        print(
            f"- {block.block_id} | t0={block.t0:%Y-%m-%d %H:%M:%S} "
            f"| p0={block.p0} | p1={block.p1} | label={block.label} "
            f"| product={block.product} | profiles={len(profiles)} | text={root_text}"
        )

    if args.output_jsonl:
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for block in blocks:
                handle.write(json.dumps(block.to_dict(), ensure_ascii=False) + "\n")
        print(f"Wrote blocks JSONL: {output_path}")


if __name__ == "__main__":
    main()


