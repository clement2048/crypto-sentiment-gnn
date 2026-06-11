"""运行符合 v2 文档顺序的完整原型流程。

顺序是：
CommentBlock -> Profile -> Debate -> HeteroGraph -> ModelSummary -> JudgeOutput

注意：法官在模型摘要之后才运行，不做“辩论后先判一次”。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from agent import DebateOrchestrator, DebateTranscript, create_debate_client
from config import DEFAULT_DEBATE_ROUNDS, DEFAULT_LIMIT_BLOCKS, FULL_PIPELINE_TRAIN_EPOCHS, LEARNING_RATE, PRINT_SAMPLES
from data import build_comment_blocks, load_posts
from data.schema import CommentBlock
from debate_graph import HeteroGraph, build_hetero_graph, graph_to_tensor
from debate_graph.graph_batch import GraphTensor, NODE_FEATURE_DIM
from judge import create_judge_client
from model import GraphSentimentModel
from model.losses import classification_loss
from profiles import ProfileStore
from scripts.run_debate import DEFAULT_INPUT


@dataclass
class PipelineContext:
    """一个样本在完整流程中的中间产物集合。
    把这些对象放在一起，是为了避免每个阶段重复构建上游结果。
    """

    block: CommentBlock
    profiles: dict[str, object]
    transcript: DebateTranscript
    graph: HeteroGraph
    graph_tensor: GraphTensor


def run_full_pipeline(
    input_path: str,
    limit_blocks: int | None = DEFAULT_LIMIT_BLOCKS,
    rounds: int = DEFAULT_DEBATE_ROUNDS,
    train_epochs: int = FULL_PIPELINE_TRAIN_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    debate_mode: str = "mock",
    judge_mode: str = "mock",
) -> list[dict[str, object]]:
    """运行完整 pipeline，并返回可 JSON 序列化的记录列表。"""
    contexts = _build_contexts(
        input_path=input_path,
        limit_blocks=limit_blocks,
        rounds=rounds,
        debate_mode=debate_mode,
    )
    model = GraphSentimentModel(input_dim=NODE_FEATURE_DIM)
    if train_epochs > 0:
        # 这是原型 smoke 训练，不是正式训练系统。
        _train_model(model, [context.graph_tensor for context in contexts], train_epochs, learning_rate)

    judge = create_judge_client(judge_mode)
    records: list[dict[str, object]] = []
    for context in contexts:
        # 模型先给出 ODE/calibrator 摘要，然后法官基于该摘要和辩论结构做最终分析。
        model_summary = model.summarize(context.graph_tensor)
        judge_output = judge.judge(context.transcript, model_summary, context.graph)
        records.append(
            {
                "block": context.block.to_dict(),
                "profiles": {
                    author: profile.to_dict()
                    for author, profile in context.profiles.items()
                    if hasattr(profile, "to_dict")
                },
                "debate": context.transcript.to_dict(),
                "graph": context.graph.to_dict(),
                "model_summary": model_summary.to_dict(),
                "judge": judge_output.to_dict(),
            }
        )
    return records


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Run debate -> graph -> ODE summary -> judge pipeline.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--limit-blocks", type=int, default=DEFAULT_LIMIT_BLOCKS)
    parser.add_argument("--rounds", type=int, default=DEFAULT_DEBATE_ROUNDS)
    parser.add_argument("--train-epochs", type=int, default=FULL_PIPELINE_TRAIN_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--debate-mode", choices=["mock", "deepseek", "bailian", "minimax"], default="mock")
    parser.add_argument("--judge-mode", choices=["mock", "deepseek", "bailian", "minimax"], default="mock")
    parser.add_argument("--output-jsonl", type=str, default=None)
    args = parser.parse_args()

    records = run_full_pipeline(
        input_path=args.input,
        limit_blocks=args.limit_blocks,
        rounds=args.rounds,
        train_epochs=args.train_epochs,
        learning_rate=args.learning_rate,
        debate_mode=args.debate_mode,
        judge_mode=args.judge_mode,
    )
    print(f"Full pipeline records: {len(records)}")
    for record in records[: min(len(records), PRINT_SAMPLES)]:
        block = record["block"]
        model_summary = record["model_summary"]
        judge = record["judge"]
        assert isinstance(block, dict)
        assert isinstance(model_summary, dict)
        assert isinstance(judge, dict)
        print(
            f"- {block['block_id']} | model_prob={model_summary['bullish_probability']:.3f} "
            f"| verdict={judge['verdict']} | confidence={judge['confidence']:.3f}"
        )

    if args.output_jsonl:
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Wrote full pipeline JSONL: {output_path}")


def _build_contexts(
    input_path: str,
    limit_blocks: int | None,
    rounds: int,
    debate_mode: str,
) -> list[PipelineContext]:
    """构建完整流程需要的中间对象，但暂不运行模型和法官。"""
    posts = load_posts(input_path)
    blocks, _issues = build_comment_blocks(posts)
    if limit_blocks is not None:
        blocks = blocks[:limit_blocks]

    profile_store = ProfileStore.from_blocks(blocks)
    orchestrator = DebateOrchestrator(client=create_debate_client(debate_mode))
    contexts: list[PipelineContext] = []
    for block in blocks:
        # 每个 block 独立生成画像、辩论和异构图。
        profiles = profile_store.get_profiles_for_block(block)
        transcript = orchestrator.run(block, profiles, rounds=rounds)
        graph = build_hetero_graph(block, transcript)
        contexts.append(
            PipelineContext(
                block=block,
                profiles=profiles,
                transcript=transcript,
                graph=graph,
                graph_tensor=graph_to_tensor(graph, label=block.label),
            )
        )
    return contexts


def _train_model(
    model: GraphSentimentModel,
    tensors: list[GraphTensor],
    epochs: int,
    learning_rate: float,
) -> None:
    """用当前样本训练几步模型，仅用于证明可反向传播。"""
    # TODO:为什么使用的是什么AdamW优化器？这个是实验结论还是随便选的？后续可以对比不同优化器的效果。
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for _epoch in range(epochs):
        total_loss = torch.tensor(0.0)
        for graph in tensors:
            assert graph.label is not None
            total_loss = total_loss + classification_loss(model(graph), graph.label)
        loss = total_loss / max(len(tensors), 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    main()




