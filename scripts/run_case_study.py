"""Run and render a readable DeepSeek debate case study."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from agent import DebateOrchestrator, DebateTranscript, create_debate_client
from agent.llm_client import DebateClient
from config import DEFAULT_DEBATE_ROUNDS
from data import build_comment_blocks, load_posts
from data.schema import CommentBlock, PostRecord, RawComment, datetime_to_str
from debate_graph import HeteroGraph, build_hetero_graph, graph_to_tensor
from debate_graph.graph_batch import NODE_FEATURE_DIM
from judge import create_judge_client
from model import GraphSentimentModel
from profiles import ProfileStore
from scripts.run_debate import DEFAULT_INPUT


@dataclass
class CaseBlockRecord:
    block: CommentBlock
    transcript: DebateTranscript
    graph: HeteroGraph
    model_summary: Any
    judge: Any


def run_case_study(
    input_path: str = DEFAULT_INPUT,
    post_id: str | None = None,
    block_id: str | None = None,
    max_blocks: int | None = None,
    rounds: int = 1,
    debate_mode: str = "deepseek",
    judge_mode: str = "siliconflow",
    seed: int = 42,
    debate_client: DebateClient | None = None,
    judge_client: object | None = None,
) -> dict[str, Any]:
    """选择一个多评论帖子，运行辩论并返回可序列化结果。"""
    torch.manual_seed(seed)
    posts = load_posts(input_path)
    all_blocks, issues = build_comment_blocks(posts)
    post = _select_post(posts, post_id=post_id, block_id=block_id)
    post_blocks = [block for block in all_blocks if block.post_id == post.post_id]
    if block_id:
        post_blocks = [block for block in post_blocks if block.block_id == block_id]
    post_blocks = sorted(post_blocks, key=lambda item: item.t0)
    if max_blocks is not None:
        post_blocks = post_blocks[:max_blocks]
    if not post_blocks:
        raise ValueError("No CommentBlock found for the requested case")

    # 当前数据没有 replies。为了案例阅读，把同帖其他 root comments 作为额外上下文传给 LLM，
    # 但不把它们伪装成 reply 边，也不把 label/p1 等未来字段传给 LLM。
    for block in post_blocks:
        setattr(block, "case_context_comments", [comment for comment in post.comments if comment.comment_id != block.root_comment.comment_id])

    profile_store = ProfileStore.from_blocks(all_blocks)
    orchestrator = DebateOrchestrator(client=debate_client or create_debate_client(debate_mode))
    model = GraphSentimentModel(input_dim=NODE_FEATURE_DIM)
    judge = judge_client or create_judge_client(judge_mode)

    block_records: list[CaseBlockRecord] = []
    for block in post_blocks:
        profiles = profile_store.get_profiles_for_block(block)
        transcript = orchestrator.run(block, profiles, rounds=rounds)
        graph = build_hetero_graph(block, transcript)
        model_summary = model.summarize(graph_to_tensor(graph, label=block.label))
        judge_output = judge.judge(transcript, model_summary, graph)
        block_records.append(
            CaseBlockRecord(
                block=block,
                transcript=transcript,
                graph=graph,
                model_summary=model_summary,
                judge=judge_output,
            )
        )

    return {
        "config": {
            "input_path": input_path,
            "post_id": post.post_id,
            "block_id": block_id,
            "max_blocks": max_blocks,
            "rounds": rounds,
            "debate_mode": debate_mode,
            "judge_mode": judge_mode,
            "seed": seed,
            "filter_issues": len(issues),
            "note": "case_context_comments are same-post root comments supplied for case reading only; they are not reply edges.",
        },
        "post": _post_for_output(post),
        "records": [_record_to_dict(item) for item in block_records],
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Run a readable case study for a post with many comments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--post-id", default=None)
    parser.add_argument("--block-id", default=None)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--debate-mode", choices=["deepseek", "bailian", "siliconflow"], default="deepseek")
    parser.add_argument("--judge-mode", choices=["deepseek", "bailian", "siliconflow"], default="siliconflow")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

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
        judge = record["judge"]
        block = record["block"]
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


def render_case_markdown(result: dict[str, Any]) -> str:
    """把 case JSON 渲染成便于阅读的 Markdown。"""
    post = result["post"]
    config = result["config"]
    lines: list[str] = [
        f"# Debate Case Study: post {post['post_id']}",
        "",
        "## Case Config",
        "",
        f"- debate_mode: `{config['debate_mode']}`",
        f"- judge_mode: `{config['judge_mode']}`",
        f"- rounds: `{config['rounds']}`",
        f"- seed: `{config['seed']}`",
        f"- note: {config['note']}",
        "",
        "## Post",
        "",
        f"- product: `{post.get('first_product')}`",
        f"- market_type: `{post.get('market_type')}`",
        f"- post_time: `{post.get('post_time')}`",
        f"- root_comments: `{len(post['comments'])}`",
        "",
        _quote(post.get("post_content") or ""),
        "",
        "## Same-Post Comments",
        "",
    ]
    for comment in post["comments"]:
        lines.extend([
            f"- `{comment['comment_id']}` author=`{comment['author']}` time=`{comment['post_time']}`",
            f"  - {_one_line(comment['text'])}",
        ])

    for record in result["records"]:
        block = record["block"]
        judge = record["judge"]
        graph = record["graph"]
        model_summary = record["model_summary"]
        lines.extend([
            "",
            f"## Block {block['block_id']}",
            "",
            f"- true label: `{block['label']}`",
            f"- t0: `{block['t0']}`",
            f"- root comment: {_one_line(block['root_comment']['text'])}",
            f"- graph nodes: `{len(graph['nodes'])}`, edges: `{len(graph['edges'])}`, relations: `{graph['relation_counts']}`",
            f"- model bullish_probability: `{model_summary['bullish_probability']:.3f}`",
            f"- judge verdict: `{judge['verdict']}`, confidence: `{judge['confidence']:.3f}`",
            "",
            "### Judge Report",
            "",
            judge["report"],
            "",
            "### Debate Process",
            "",
        ])
        for argument in record["debate"]["arguments"]:
            lines.extend([
                f"#### {argument['seq']}. {argument['camp']} / {argument['role']}",
                "",
                f"- confidence: `{argument['confidence']:.3f}`",
                f"- targets: `{argument['targets']}`",
                f"- claim: {argument['claim']}",
                "- evidence:",
            ])
            for evidence in argument["evidence"]:
                lines.append(
                    f"  - `{evidence['source_type']}` `{evidence['source_id']}` "
                    f"relevance=`{evidence['relevance']:.2f}`: {_one_line(evidence['quote'])}"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _select_post(posts: list[PostRecord], post_id: str | None, block_id: str | None) -> PostRecord:
    if post_id:
        for post in posts:
            if post.post_id == post_id:
                return post
        raise ValueError(f"Post not found: {post_id}")
    if block_id:
        target_post_id = block_id.split(":", 1)[0]
        for post in posts:
            if post.post_id == target_post_id:
                return post
        raise ValueError(f"Post not found for block: {block_id}")
    return max(posts, key=lambda item: len(item.comments))


def _post_for_output(post: PostRecord) -> dict[str, Any]:
    return {
        "post_id": post.post_id,
        "post_content": post.post_content,
        "post_time": datetime_to_str(post.post_time),
        "products": post.products,
        "first_product": post.first_product,
        "market_type": post.market_type,
        "comments": [_comment_for_output(comment) for comment in post.comments],
    }


def _comment_for_output(comment: RawComment) -> dict[str, Any]:
    return {
        "comment_id": comment.comment_id,
        "author": comment.author,
        "text": comment.text,
        "post_time": datetime_to_str(comment.post_time),
    }


def _record_to_dict(item: CaseBlockRecord) -> dict[str, Any]:
    return {
        "block": item.block.to_dict(),
        "debate": item.transcript.to_dict(),
        "graph": item.graph.to_dict(),
        "model_summary": item.model_summary.to_dict(),
        "judge": item.judge.to_dict(),
    }


def _quote(text: str) -> str:
    return "\n".join(f"> {line}" for line in text.splitlines() if line.strip())


def _one_line(text: str, limit: int = 240) -> str:
    compact = " ".join(str(text).split())
    return compact[:limit] + ("..." if len(compact) > limit else "")


if __name__ == "__main__":
    main()


