"""Run offline debates for v2 CommentBlock samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent import DebateOrchestrator, create_debate_client
from agent.llm_client import DebateClient
from config import DEFAULT_DEBATE_ROUNDS, DEFAULT_INPUT_PATH, DEFAULT_LIMIT_BLOCKS, PRINT_SAMPLES
from data import build_comment_blocks, load_posts
from profiles import ProfileStore


DEFAULT_INPUT = DEFAULT_INPUT_PATH


def run_debate_pipeline(
    input_path: str,
    limit_posts: int | None = None,
    limit_blocks: int | None = None,
    rounds: int = DEFAULT_DEBATE_ROUNDS,
    mode: str = "siliconflow",
    client: DebateClient | None = None,
) -> list[dict[str, object]]:
    posts = load_posts(input_path)
    if limit_posts is not None:
        posts = posts[:limit_posts]
    blocks, _issues = build_comment_blocks(posts)
    if limit_blocks is not None:
        blocks = blocks[:limit_blocks]

    profile_store = ProfileStore.from_blocks(blocks)
    orchestrator = DebateOrchestrator(client=client or create_debate_client(mode))

    records: list[dict[str, object]] = []
    for block in blocks:
        profiles = profile_store.get_profiles_for_block(block)
        transcript = orchestrator.run(block, profiles, rounds=rounds)
        records.append(
            {
                "block": block.to_dict(),
                "profiles": {author: profile.to_dict() for author, profile in profiles.items()},
                "debate": transcript.to_dict(),
            }
        )
    return records


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Run online debate pipeline.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="JSONL file, directory, or glob pattern.")
    parser.add_argument("--limit-posts", type=int, default=None)
    parser.add_argument("--limit-blocks", type=int, default=DEFAULT_LIMIT_BLOCKS)
    parser.add_argument("--rounds", type=int, default=DEFAULT_DEBATE_ROUNDS)
    parser.add_argument("--mode", choices=["siliconflow"], default="siliconflow")
    parser.add_argument("--output-jsonl", type=str, default=None)
    args = parser.parse_args()

    records = run_debate_pipeline(
        input_path=args.input,
        limit_posts=args.limit_posts,
        limit_blocks=args.limit_blocks,
        rounds=args.rounds,
        mode=args.mode,
    )

    print(f"Debate records: {len(records)}")
    for record in records[: min(len(records), PRINT_SAMPLES)]:
        block = record["block"]
        debate = record["debate"]
        assert isinstance(block, dict)
        assert isinstance(debate, dict)
        print(
            f"- {block['block_id']} | arguments={len(debate['arguments'])}"
        )

    if args.output_jsonl:
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Wrote debate JSONL: {output_path}")


if __name__ == "__main__":
    main()



