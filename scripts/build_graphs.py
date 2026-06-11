"""Build heterogeneous graphs from comment blocks and online debates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from agent.llm_client import DebateClient
from config import DEFAULT_DEBATE_ROUNDS, DEFAULT_LIMIT_BLOCKS
from debate_graph import build_hetero_graph
from debate_graph.diffusion_ops import normalized_relation_adjacency
from scripts.run_debate import DEFAULT_INPUT, run_debate_pipeline
from agent.schema import DebateTranscript
from data.schema import CommentBlock


def build_graph_records(
    input_path: str,
    limit_blocks: int | None = DEFAULT_LIMIT_BLOCKS,
    rounds: int = DEFAULT_DEBATE_ROUNDS,
    mode: str = "minimax",
    client: DebateClient | None = None,
) -> list[dict[str, object]]:
    debate_records = run_debate_pipeline(
        input_path=input_path,
        limit_blocks=limit_blocks,
        rounds=rounds,
        mode=mode,
        client=client,
    )
    graph_records: list[dict[str, object]] = []
    for record in debate_records:
        block = _comment_block_from_record(record["block"])
        transcript = DebateTranscript.from_dict(record["debate"])
        graph = build_hetero_graph(block, transcript)
        graph_records.append(
            {
                "block_id": block.block_id,
                "graph": graph.to_dict(),
                "normalized_adjacency": {
                    relation: [
                        {"source": source, "target": target, "weight": weight}
                        for source, target, weight in triples
                    ]
                    for relation, triples in normalized_relation_adjacency(graph).items()
                },
            }
        )
    return graph_records


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Build hetero graphs from debate records.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="JSONL file, directory, or glob pattern.")
    parser.add_argument("--limit-blocks", type=int, default=DEFAULT_LIMIT_BLOCKS)
    parser.add_argument("--rounds", type=int, default=DEFAULT_DEBATE_ROUNDS)
    parser.add_argument("--mode", choices=["deepseek", "bailian", "minimax", "siliconflow"], default="minimax")
    parser.add_argument("--output-jsonl", type=str, default=None)
    args = parser.parse_args()

    records = build_graph_records(args.input, limit_blocks=args.limit_blocks, rounds=args.rounds, mode=args.mode)
    relation_totals: Counter[str] = Counter()
    print(f"Graph records: {len(records)}")
    for record in records:
        graph = record["graph"]
        assert isinstance(graph, dict)
        relation_counts = graph["relation_counts"]
        assert isinstance(relation_counts, dict)
        relation_totals.update(relation_counts)
        print(
            f"- {record['block_id']} | nodes={len(graph['nodes'])} "
            f"| edges={len(graph['edges'])} | relations={relation_counts}"
        )
    if relation_totals:
        print(f"Relation totals: {dict(sorted(relation_totals.items()))}")

    if args.output_jsonl:
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Wrote graph JSONL: {output_path}")


def _comment_block_from_record(data: object) -> CommentBlock:
    if not isinstance(data, dict):
        raise ValueError("block record must be a dict")
    from data.schema import RawComment, parse_datetime

    root_data = data.get("root_comment")
    if not isinstance(root_data, dict):
        raise ValueError("block record missing root_comment")
    t0 = parse_datetime(data.get("t0"))
    if t0 is None:
        raise ValueError("block record missing t0")
    return CommentBlock(
        block_id=str(data.get("block_id") or ""),
        post_id=str(data.get("post_id") or ""),
        post_content=str(data.get("post_content") or ""),
        products=[str(item) for item in data.get("products", [])],
        root_comment=RawComment.from_dict(root_data),
        replies=[RawComment.from_dict(item) for item in data.get("replies", [])],
        t0=t0,
        t_window=str(data.get("t_window") or ""),
        p0=float(data.get("p0")),
        p1=float(data.get("p1")),
        label=int(data.get("label")),
        product=str(data.get("product")) if data.get("product") is not None else None,
        market_type=str(data.get("market_type")) if data.get("market_type") is not None else None,
        post_time=parse_datetime(data.get("post_time")),
    )


if __name__ == "__main__":
    main()



