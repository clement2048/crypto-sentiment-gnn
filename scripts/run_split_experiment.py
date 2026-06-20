"""Run a small chronological train/val/test experiment."""

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
from config import DEFAULT_DEBATE_ROUNDS, LEARNING_RATE, PRINT_SAMPLES, TRAIN_CHECKPOINT_DIR
from data import build_comment_blocks, load_posts
from data.schema import CommentBlock
from debate_graph import HeteroGraph, build_hetero_graph, graph_to_tensor
from debate_graph.graph_batch import GraphTensor
from judge import create_judge_client
from model import GraphSentimentModel, TrainingConfig, train_graph_model
from profiles import ProfileStore
from scripts.evaluate_pipeline import EvaluationMetrics, compute_metrics
from scripts.run_debate import DEFAULT_INPUT
from verification import verify_market_behavior


@dataclass
class ExperimentContext:
    block: CommentBlock
    profiles: dict[str, object]
    transcript: DebateTranscript
    graph: HeteroGraph
    graph_tensor: GraphTensor


def run_split_experiment(
    input_path: str = DEFAULT_INPUT,
    train_count: int = 9,
    val_count: int = 3,
    test_count: int = 3,
    rounds: int = DEFAULT_DEBATE_ROUNDS,
    epochs: int = 5,
    learning_rate: float = LEARNING_RATE,
    debate_mode: str = "siliconflow",
    judge_mode: str = "siliconflow",
    seed: int = 42,
    embedding_backend: str | None = None,
    debate_client: DebateClient | None = None,
    judge_client: object | None = None,
) -> dict[str, Any]:
    """按时间顺序运行 train/val/test 小实验。"""
    if train_count <= 0 or val_count < 0 or test_count <= 0:
        raise ValueError("train_count and test_count must be positive; val_count can be zero")

    torch.manual_seed(seed)
    total_needed = train_count + val_count + test_count
    posts = load_posts(input_path)
    blocks, issues = build_comment_blocks(posts)
    sorted_blocks = sorted(blocks, key=lambda item: item.t0)
    if len(sorted_blocks) < total_needed:
        raise ValueError(f"Need {total_needed} blocks, but only found {len(sorted_blocks)}")
    selected = sorted_blocks[:total_needed]

    train_blocks = selected[:train_count]
    val_blocks = selected[train_count : train_count + val_count]
    test_blocks = selected[train_count + val_count :]

    profile_store = ProfileStore.from_blocks(sorted_blocks)
    orchestrator = DebateOrchestrator(client=debate_client or create_debate_client(debate_mode))
    contexts = [
        _build_context(block, profile_store, orchestrator, rounds, embedding_backend)
        for block in selected
    ]
    train_contexts = contexts[:train_count]
    val_contexts = contexts[train_count : train_count + val_count]
    test_contexts = contexts[train_count + val_count :]

    model = GraphSentimentModel(input_dim=contexts[0].graph_tensor.x.shape[1])
    training = train_graph_model(
        model,
        [item.graph_tensor for item in train_contexts],
        TrainingConfig(
            epochs=epochs,
            learning_rate=learning_rate,
            checkpoint_path=str(Path(TRAIN_CHECKPOINT_DIR) / "split_experiment.pt"),
        ),
    )

    judge = judge_client or create_judge_client(judge_mode)
    train_records = _records_for_contexts(model, judge, train_contexts)
    val_records = _records_for_contexts(model, judge, val_contexts)
    test_records = _records_for_contexts(model, judge, test_contexts)

    return {
        "config": {
            "input_path": str(input_path),
            "train_count": train_count,
            "val_count": val_count,
            "test_count": test_count,
            "rounds": rounds,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "debate_mode": debate_mode,
            "judge_mode": judge_mode,
            "seed": seed,
            "filter_issues": len(issues),
            "selected_time_range": {
                "start": selected[0].t0.strftime("%Y-%m-%d %H:%M:%S"),
                "end": selected[-1].t0.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "split_block_ids": {
                "train": [block.block_id for block in train_blocks],
                "val": [block.block_id for block in val_blocks],
                "test": [block.block_id for block in test_blocks],
            },
        },
        "train_losses": training.train_losses,
        "training": training.to_dict(),
        "metrics": {
            "train": compute_metrics(train_records).to_dict(),
            "val": compute_metrics(val_records).to_dict() if val_records else None,
            "test": compute_metrics(test_records).to_dict(),
        },
        "records": {
            "train": train_records,
            "val": val_records,
            "test": test_records,
        },
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Run chronological train/val/test experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--train-count", type=int, default=9)
    parser.add_argument("--val-count", type=int, default=3)
    parser.add_argument("--test-count", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=DEFAULT_DEBATE_ROUNDS)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--debate-mode", choices=["deepseek", "bailian", "siliconflow"], default="siliconflow")
    parser.add_argument("--judge-mode", choices=["deepseek", "bailian", "siliconflow"], default="siliconflow")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--embedding-backend",
        choices=["none", "sentencebert", "finbert", "sentencebert_finbert"],
        default=None,
    )
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    result = run_split_experiment(
        input_path=args.input,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
        rounds=args.rounds,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        debate_mode=args.debate_mode,
        judge_mode=args.judge_mode,
        seed=args.seed,
        embedding_backend=args.embedding_backend,
    )
    _print_result(result)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote experiment JSON: {output_path}")


def _build_context(
    block: CommentBlock,
    profile_store: ProfileStore,
    orchestrator: DebateOrchestrator,
    rounds: int,
    embedding_backend: str | None,
) -> ExperimentContext:
    profiles = profile_store.get_profiles_for_block(block)
    transcript = orchestrator.run(block, profiles, rounds=rounds)
    graph = build_hetero_graph(block, transcript)
    return ExperimentContext(
        block=block,
        profiles=profiles,
        transcript=transcript,
        graph=graph,
        graph_tensor=graph_to_tensor(graph, label=block.label, embedding_backend=embedding_backend),
    )


def _records_for_contexts(
    model: GraphSentimentModel,
    judge,
    contexts: list[ExperimentContext],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for context in contexts:
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
                "market_verification": verify_market_behavior(
                    context.block.p0,
                    context.block.p1,
                    judge_output.verdict,
                ).to_dict(),
            }
        )
    return records


def _print_result(result: dict[str, Any]) -> None:
    config = result["config"]
    print(
        "Split experiment complete: "
        f"train={config['train_count']} val={config['val_count']} test={config['test_count']} "
        f"rounds={config['rounds']} epochs={config['epochs']} "
        f"debate_mode={config['debate_mode']} judge_mode={config['judge_mode']} seed={config['seed']}"
    )
    print(
        "Selected time range: "
        f"{config['selected_time_range']['start']} -> {config['selected_time_range']['end']}"
    )
    losses = result["train_losses"]
    if losses:
        print(f"Train loss: first={losses[0]:.4f} last={losses[-1]:.4f}")
    for split_name in ("train", "val", "test"):
        metrics = result["metrics"][split_name]
        if metrics is None:
            continue
        _print_metrics(split_name, metrics)
    for split_name in ("train", "val", "test"):
        records = result["records"][split_name]
        for record in records[: min(len(records), PRINT_SAMPLES)]:
            block = record["block"]
            judge = record["judge"]
            assert isinstance(block, dict)
            assert isinstance(judge, dict)
            print(
                f"{split_name}: {block['block_id']} | true={block['label']} "
                f"| pred={judge['verdict']} | confidence={judge['confidence']:.3f}"
            )
            print(f"{split_name} judge_report: {judge.get('report', '')}")


def _print_metrics(split_name: str, metrics: dict[str, Any]) -> None:
    print(
        f"{split_name.upper()} metrics: "
        f"n={metrics['total']} "
        f"accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"bull_f1={metrics['bullish']['f1']:.4f} "
            f"bear_f1={metrics['bearish']['f1']:.4f}"
        )
    print(f"{split_name.upper()} confusion: {metrics['confusion_matrix']}")


if __name__ == "__main__":
    main()

