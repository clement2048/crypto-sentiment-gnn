"""Train a tiny graph sentiment prototype on debate graphs."""

from __future__ import annotations

import argparse
import sys

import torch

from agent import DebateOrchestrator, create_debate_client
from agent.llm_client import DebateClient
from config import DEFAULT_DEBATE_ROUNDS, LEARNING_RATE, TRAIN_PROTOTYPE_EPOCHS, TRAIN_PROTOTYPE_LIMIT_BLOCKS
from data import build_comment_blocks, load_posts
from debate_graph import build_hetero_graph, graph_to_tensor
from debate_graph.graph_batch import GraphTensor, NODE_FEATURE_DIM
from model import GraphSentimentModel
from model.losses import classification_loss
from profiles import ProfileStore
from scripts.run_debate import DEFAULT_INPUT


def build_training_tensors(
    input_path: str,
    limit_blocks: int | None = TRAIN_PROTOTYPE_LIMIT_BLOCKS,
    rounds: int = DEFAULT_DEBATE_ROUNDS,
    mode: str = "minimax",
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
    mode: str = "minimax",
    client: DebateClient | None = None,
) -> dict[str, float]:
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    last_loss = 0.0
    for _epoch in range(epochs):
        total_loss = torch.tensor(0.0)
        for graph in tensors:
            assert graph.label is not None
            prob = model(graph)
            total_loss = total_loss + classification_loss(prob, graph.label)
        loss = total_loss / len(tensors)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())

    with torch.no_grad():
        probs = [float(model(graph)) for graph in tensors]
    return {
        "graphs": float(len(tensors)),
        "final_loss": last_loss,
        "mean_probability": sum(probs) / len(probs),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Train the minimal graph sentiment prototype.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--limit-blocks", type=int, default=TRAIN_PROTOTYPE_LIMIT_BLOCKS)
    parser.add_argument("--rounds", type=int, default=DEFAULT_DEBATE_ROUNDS)
    parser.add_argument("--mode", choices=["deepseek", "bailian", "minimax", "siliconflow"], default="minimax")
    parser.add_argument("--epochs", type=int, default=TRAIN_PROTOTYPE_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    metrics = train_prototype(
        input_path=args.input,
        limit_blocks=args.limit_blocks,
        rounds=args.rounds,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        mode=args.mode,
    )
    print(
        "Prototype training complete: "
        f"graphs={metrics['graphs']:.0f} "
        f"final_loss={metrics['final_loss']:.4f} "
        f"mean_probability={metrics['mean_probability']:.4f}"
    )


if __name__ == "__main__":
    main()




