from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from config import (
    DEFAULT_DEBATE_ROUNDS,
    DEFAULT_LIMIT_BLOCKS,
    PRINT_SAMPLES,
)
from data import build_comment_blocks, load_posts, temporal_split_blocks
from profiles import ProfileStore
from scripts.build_graphs import build_graph_records
from scripts.run_case_study import render_case_markdown, run_case_study
from scripts.run_debate import DEFAULT_INPUT, run_debate_pipeline


# 第一阶段只保留数据检查 + debate + graphs + case-study 这几条入口。
# 第二阶段启动时再把 judge / evaluation 之类的子命令接回 main.py。
# 已暂停的 train / full / evaluate / split-experiment 等旧链路全部移到 archive/。

def main() -> None:
    """解析命令行参数，并分发到对应阶段。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Crypto sentiment analysis active commands (blocks / debate / graphs / case-study).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # blocks：最早的数据阶段，只检查 JSONL -> CommentBlock 是否正确。
    blocks = subparsers.add_parser("blocks", help="Build CommentBlock samples from JSONL.")
    _add_input_arg(blocks)
    blocks.add_argument("--limit", type=int, default=None)
    blocks.add_argument("--print-samples", type=int, default=PRINT_SAMPLES)
    blocks.add_argument("--output-jsonl", type=str, default=None)
    blocks.set_defaults(func=_cmd_blocks)

    # debate：只生成辩论，不调用模型和法官。第二阶段启用。
    debate = subparsers.add_parser("debate", help="Run online multi-agent debate.")
    _add_input_arg(debate)
    debate.add_argument("--limit-blocks", type=int, default=DEFAULT_LIMIT_BLOCKS)
    debate.add_argument("--rounds", type=int, default=DEFAULT_DEBATE_ROUNDS)
    debate.add_argument("--mode", choices=["siliconflow"], default="siliconflow")
    debate.add_argument("--output-jsonl", type=str, default=None)
    debate.set_defaults(func=_cmd_debate)

    # graphs：把评论树和辩论论点融合成图。第二阶段启用。
    graphs = subparsers.add_parser("graphs", help="Build heterogeneous debate/comment graphs.")
    _add_input_arg(graphs)
    graphs.add_argument("--limit-blocks", type=int, default=DEFAULT_LIMIT_BLOCKS)
    graphs.add_argument("--rounds", type=int, default=DEFAULT_DEBATE_ROUNDS)
    graphs.add_argument("--mode", choices=["siliconflow"], default="siliconflow")
    graphs.add_argument("--output-jsonl", type=str, default=None)
    graphs.set_defaults(func=_cmd_graphs)

    # case-study：选一个评论较多的帖子，生成可阅读的辩论过程报告。
    case_study = subparsers.add_parser("case-study", help="Run and render a readable debate case study.")
    _add_input_arg(case_study)
    case_study.add_argument("--post-id", default=None)
    case_study.add_argument("--block-id", default=None)
    case_study.add_argument("--max-blocks", type=int, default=None)
    case_study.add_argument("--rounds", type=int, default=DEFAULT_DEBATE_ROUNDS)
    case_study.add_argument("--debate-mode", choices=["siliconflow"], default="siliconflow")
    case_study.add_argument("--judge-mode", choices=["siliconflow"], default="siliconflow")
    case_study.add_argument("--seed", type=int, default=42)
    case_study.add_argument("--output-json", default=None)
    case_study.add_argument("--output-md", default=None)
    case_study.set_defaults(func=_cmd_case_study)

    args = parser.parse_args()
    args.func(args)


def _add_input_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", default=DEFAULT_INPUT, help="JSONL file, directory, or glob pattern.")


def _cmd_blocks(args: argparse.Namespace) -> None:
    """加载 JSONL，打印评论块、过滤原因和时间切分统计。"""
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
        print(f"Filter issue counts: {dict(sorted(counts.items()))}")
    print(f"Temporal split: train={len(splits.train)} val={len(splits.val)} test={len(splits.test)}")

    for block in blocks[: min(args.print_samples, len(blocks))]:
        profiles = profile_store.get_profiles_for_block(block)
        root_text = block.root_comment.text.replace("\n", " ")[:100]
        print(
            f"- {block.block_id} | t0={block.t0:%Y-%m-%d %H:%M:%S} "
            f"| label={block.label} | product={block.product} "
            f"| profiles={len(profiles)} | text={root_text}"
        )

    if args.output_jsonl:
        _write_jsonl(args.output_jsonl, [block.to_dict() for block in blocks])


def _cmd_debate(args: argparse.Namespace) -> None:
    """只运行辩论；这里不会调用法官。"""
    records = run_debate_pipeline(args.input, limit_blocks=args.limit_blocks, rounds=args.rounds, mode=args.mode)
    print(f"Debate records: {len(records)}")
    for record in records[: min(len(records), PRINT_SAMPLES)]:
        block = record["block"]
        debate = record["debate"]
        assert isinstance(block, dict)
        assert isinstance(debate, dict)
        print(f"- {block['block_id']} | arguments={len(debate['arguments'])}")
    if args.output_jsonl:
        _write_jsonl(args.output_jsonl, records)


def _cmd_graphs(args: argparse.Namespace) -> None:
    """从辩论结果构建异构图。"""
    records = build_graph_records(args.input, limit_blocks=args.limit_blocks, rounds=args.rounds, mode=args.mode)
    print(f"Graph records: {len(records)}")
    for record in records[: min(len(records), PRINT_SAMPLES)]:
        graph = record["graph"]
        assert isinstance(graph, dict)
        print(
            f"- {record['block_id']} | nodes={len(graph['nodes'])} "
            f"| edges={len(graph['edges'])} | relations={graph['relation_counts']}"
        )
    if args.output_jsonl:
        _write_jsonl(args.output_jsonl, records)


def _cmd_case_study(args: argparse.Namespace) -> None:
    """运行并导出可读案例报告。"""
    result = run_case_study(
        input_path=args.input,
        post_id=args.post_id,
        block_id=args.block_id,
        max_blocks=args.max_blocks,
        rounds=args.rounds,
        debate_mode=args.debate_mode,
        judge_mode=args.judge_mode,
        seed=args.seed,
    )
    print(f"Case study post: {result['config']['post_id']}")
    print(f"Root comments in post: {len(result['post']['comments'])}")
    print(f"Debated blocks: {len(result['records'])}")
    for record in result["records"]:
        block = record["block"]
        judge = record["judge"]
        assert isinstance(block, dict)
        assert isinstance(judge, dict)
        print(
            f"- {block['block_id']} | label={block['label']} "
            f"| verdict={judge['verdict']} | confidence={judge['confidence']:.3f}"
        )
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote case JSON: {output_json}")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_case_markdown(result), encoding="utf-8")
        print(f"Wrote case Markdown: {output_md}")


def _write_jsonl(path: str, records: list[dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote JSONL: {output_path}")


if __name__ == "__main__":
    main()
