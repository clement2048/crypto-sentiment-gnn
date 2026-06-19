"""Train a tiny graph sentiment prototype on debate graphs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent import DebateOrchestrator, create_debate_client
from agent.llm_client import DebateClient
from config import (
    DEFAULT_DEBATE_ROUNDS,
    LEARNING_RATE,
    TRAIN_CHECKPOINT_DIR,
    TRAIN_PROTOTYPE_EPOCHS,
    TRAIN_PROTOTYPE_LIMIT_BLOCKS,
)
from data import build_comment_blocks, load_posts
from debate_graph import build_hetero_graph, graph_to_tensor
from debate_graph.graph_batch import GraphTensor, NODE_FEATURE_DIM
from model import GraphSentimentModel, TrainingConfig, train_graph_model
from profiles import ProfileStore
from scripts.run_debate import DEFAULT_INPUT


def build_training_tensors(
    input_path: str,
    limit_blocks: int | None = TRAIN_PROTOTYPE_LIMIT_BLOCKS,
    rounds: int = DEFAULT_DEBATE_ROUNDS,
    mode: str = "siliconflow",
    client: DebateClient | None = None,
) -> list[GraphTensor]:
    posts = load_posts(input_path)
    blocks, _issues = build_comment_blocks(posts)
    if limit_blocks is not None:
        blocks = blocks[:limit_blocks]
    profile_store = ProfileStore.from_blocks(blocks)
    orchestrator = DebateOrchestrator(client=client or create_debate_client(mode))

    tensors: list[GraphTensor] = []
    for block in blocks:
        profiles = profile_store.get_profiles_for_block(block)
        transcript = orchestrator.run(block, profiles, rounds=rounds)
        graph = build_hetero_graph(block, transcript)
        tensors.append(graph_to_tensor(graph, label=block.label))
    return tensors


def train_prototype(
    input_path: str,
    limit_blocks: int | None = TRAIN_PROTOTYPE_LIMIT_BLOCKS,
    rounds: int = DEFAULT_DEBATE_ROUNDS,
    epochs: int = TRAIN_PROTOTYPE_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    mode: str = "siliconflow",
    client: DebateClient | None = None,
    checkpoint_path: str | None = None,
) -> dict[str, float | str]:
    tensors = build_training_tensors(
        input_path,
        limit_blocks=limit_blocks,
        rounds=rounds,
        mode=mode,
        client=client,
    )
    if not tensors:
        raise ValueError("No graph tensors available for training")

    model = GraphSentimentModel(input_dim=NODE_FEATURE_DIM)
    if checkpoint_path is None:
        checkpoint_path = str(Path(TRAIN_CHECKPOINT_DIR) / "train_prototype.pt")
    training = train_graph_model(
        model,
        tensors,
        TrainingConfig(
            epochs=epochs,
            learning_rate=learning_rate,
            checkpoint_path=checkpoint_path,
        ),
    )

    import torch

    with torch.no_grad():
        probs = [float(model(graph)) for graph in tensors]
    last = training.history[-1] if training.history else None
    return {
        "graphs": float(len(tensors)),
        "final_loss": float(last.total_loss if last else 0.0),
        "classification_loss": float(last.classification if last else 0.0),
        "initial_alignment_loss": float(last.initial_alignment if last else 0.0),
        "smoothness_loss": float(last.smoothness if last else 0.0),
        "mutual_exclusion_loss": float(last.mutual_exclusion if last else 0.0),
        "regression_loss": float(last.regression if last else 0.0),
        "epochs_ran": float(training.epochs_ran),
        "best_loss": float(training.best_loss),
        "mean_probability": sum(probs) / len(probs),
        "checkpoint_path": training.checkpoint_path or "",
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Train the minimal graph sentiment prototype.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--limit-blocks", type=int, default=TRAIN_PROTOTYPE_LIMIT_BLOCKS)
    parser.add_argument("--rounds", type=int, default=DEFAULT_DEBATE_ROUNDS)
    parser.add_argument("--mode", choices=["deepseek", "bailian", "siliconflow"], default="siliconflow")
    parser.add_argument("--epochs", type=int, default=TRAIN_PROTOTYPE_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--checkpoint-path", default=None)
    args = parser.parse_args()

    metrics = train_prototype(
        input_path=args.input,
        limit_blocks=args.limit_blocks,
        rounds=args.rounds,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        mode=args.mode,
        checkpoint_path=args.checkpoint_path,
    )
    print(
        "Prototype training complete: "
        f"graphs={metrics['graphs']:.0f} "
        f"final_loss={metrics['final_loss']:.4f} "
        f"mean_probability={metrics['mean_probability']:.4f} "
        f"checkpoint={metrics['checkpoint_path']}"
    )


if __name__ == "__main__":
    main()




