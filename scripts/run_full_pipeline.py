"""Run the v3 prototype pipeline.

Order:
CommentBlock -> Profile -> Debate -> interact graph -> Bi-ODE summary -> Judge
-> optional judge-guided debater reflection -> market verification.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from agent import DebateOrchestrator, DebateTranscript, create_debate_client
from agent.llm_client import DebateClient
from agent.reflection import should_continue_reflection
from config import (
    DEFAULT_DEBATE_ROUNDS,
    DEFAULT_LIMIT_BLOCKS,
    FULL_PIPELINE_TRAIN_EPOCHS,
    LEARNING_RATE,
    PRINT_SAMPLES,
    TRAIN_CHECKPOINT_DIR,
)
from data import build_comment_blocks, load_posts
from data.schema import CommentBlock
from debate_graph import HeteroGraph, build_hetero_graph, graph_to_tensor
from debate_graph.graph_batch import GraphTensor, NODE_FEATURE_DIM
from judge import create_judge_client, reflection_signal_from_judge
from model import GraphSentimentModel, TrainingConfig, train_graph_model
from profiles import ProfileStore
from scripts.run_debate import DEFAULT_INPUT
from verification import verify_market_behavior


@dataclass
class PipelineContext:
    """Intermediate artifacts for one sample; kept together to avoid rebuilding upstream stages."""

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
    debate_mode: str = "siliconflow",
    judge_mode: str = "siliconflow",
    reflection_rounds: int = 0,
    debate_client: DebateClient | None = None,
    judge_client: object | None = None,
) -> list[dict[str, object]]:
    """Run the full pipeline and return JSON-serializable records."""
    orchestrator = DebateOrchestrator(client=debate_client or create_debate_client(debate_mode))
    contexts = _build_contexts(
        input_path=input_path,
        limit_blocks=limit_blocks,
        rounds=rounds,
        orchestrator=orchestrator,
    )
    model = GraphSentimentModel(input_dim=NODE_FEATURE_DIM)
    training_summary: dict[str, object] | None = None
    if train_epochs > 0:
        training = train_graph_model(
            model,
            [context.graph_tensor for context in contexts],
            TrainingConfig(
                epochs=train_epochs,
                learning_rate=learning_rate,
                checkpoint_path=str(Path(TRAIN_CHECKPOINT_DIR) / "full_pipeline.pt"),
            ),
        )
        training_summary = training.to_dict()

    judge = judge_client or create_judge_client(judge_mode)
    records: list[dict[str, object]] = []
    for context in contexts:
        model_summary = model.summarize(context.graph_tensor)
        judge_output = judge.judge(context.transcript, model_summary, context.graph)
        reflection_history: list[dict[str, object]] = []

        for _reflection_index in range(max(reflection_rounds, 0)):
            # The reflection signal is derived from Judge's report and model summary only.
            # It must not include label, p1, or any future market field.
            signal = reflection_signal_from_judge(
                judge_output,
                model_summary,
                mean_argument_confidence=_mean_argument_confidence(context.transcript),
            )
            reflection_history.append(signal.to_dict())
            if not should_continue_reflection(signal):
                break
            context.transcript = orchestrator.add_reflection_rounds(
                context.block,
                context.profiles,
                context.transcript,
                signal,
                reflection_rounds=1,
            )
            context.graph = build_hetero_graph(context.block, context.transcript)
            context.graph_tensor = graph_to_tensor(context.graph, label=context.block.label)
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
                "training": training_summary,
                "reflection_history": reflection_history,
                "market_verification": verify_market_behavior(
                    context.block.p0,
                    context.block.p1,
                    judge_output.verdict,
                ).to_dict(),
            }
        )
    return records


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Run debate -> graph -> Bi-ODE summary -> judge pipeline.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--limit-blocks", type=int, default=DEFAULT_LIMIT_BLOCKS)
    parser.add_argument("--rounds", type=int, default=DEFAULT_DEBATE_ROUNDS)
    parser.add_argument("--train-epochs", type=int, default=FULL_PIPELINE_TRAIN_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--debate-mode", choices=["deepseek", "bailian", "siliconflow"], default="siliconflow")
    parser.add_argument("--judge-mode", choices=["deepseek", "bailian", "siliconflow"], default="siliconflow")
    parser.add_argument("--reflection-rounds", type=int, default=0)
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
        reflection_rounds=args.reflection_rounds,
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
    orchestrator: DebateOrchestrator,
) -> list[PipelineContext]:
    posts = load_posts(input_path)
    blocks, _issues = build_comment_blocks(posts)
    if limit_blocks is not None:
        blocks = blocks[:limit_blocks]

    profile_store = ProfileStore.from_blocks(blocks)
    contexts: list[PipelineContext] = []
    for block in blocks:
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


def _mean_argument_confidence(transcript: DebateTranscript) -> float | None:
    values = [argument.confidence for argument in transcript.arguments]
    if not values:
        return None
    return sum(values) / len(values)


if __name__ == "__main__":
    main()
