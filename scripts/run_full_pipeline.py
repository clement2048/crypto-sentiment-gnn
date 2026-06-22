"""Run the v3 prototype pipeline.

This module is the clearest place to understand the end-to-end data flow:

1. Raw input
   `input_path` points to a JSONL file, directory, or glob. `load_posts(...)`
   parses it into `PostRecord` objects. At this stage the code is still close
   to the source data: one post contains post text, product metadata, root
   comments, and replies.

2. Sample construction
   `build_comment_blocks(...)` converts posts into `CommentBlock` samples.
   One `CommentBlock` is the unit used everywhere downstream: one root comment
   plus its replies, with root-level `t0/p0/p1/label` already supplied by the
   dataset. The full pipeline does not regenerate labels.

3. Time-safe profile store
   `ProfileStore.from_blocks(...)` builds a searchable history index from all
   available blocks. For each target block, `get_profiles_for_block(block)` is
   the only approved access path: it returns user profiles using records with
   timestamp strictly earlier than `block.t0`.

4. Debate generation
   `DebateOrchestrator.run(...)` sends the `CommentBlock`, its time-safe user
   profiles, and prior debate arguments to the bull/bear LLM agents. The output
   is a `DebateTranscript`: structured `Argument` objects with claims,
   evidence, confidence, `target_args`, round/seq, phase, and `t_index`.

5. Graph construction
   `build_hetero_graph(block, transcript)` fuses two views:
   - comment nodes from the raw comment tree; reply structure is stored in
     comment node attrs as `parent_id`;
   - argument nodes from the debate transcript; argument interactions become
     single-relation `interact` edges based on `target_args`.

6. Tensorization
   `graph_to_tensor(...)` converts graph nodes and edges into tensors:
   - node texts/attrs become node feature matrix `x`;
   - normalized `interact` adjacency becomes `relation_adjs`;
   - `CommentBlock.label` becomes graph-level tensor label when training.
   Optional text embeddings can be appended here, but the default path remains
   the structural 12-feature representation.

7. Optional graph-model training
   If `train_epochs > 0`, `train_graph_model(...)` trains `GraphSentimentModel`
   on the graph tensors built above. If `train_epochs == 0`, the model is used
   untrained, which is useful for checking the full I/O chain without changing
   parameters.

8. Model summary for Judge
   `model.summarize(graph_tensor)` runs the Bi-ODE graph model and exports a
   compact, serializable `ModelOutputSummary`. This is what the LLM Judge sees;
   it does not receive labels or future prices.

9. LLM Judge
   `judge.judge(transcript, model_summary, graph)` receives the raw debate
   graph plus model summary and returns final structured judgment fields:
   verdict, confidence, report, score vector, weak dimensions, suggestions,
   and consistency flags.

10. Optional reflection loop
    When `reflection_rounds > 0`, Judge feedback is converted into a safe
    reflection signal. The debaters append extra arguments, the graph is rebuilt,
    the model summary is recomputed, and Judge runs again. The loop appends to
    the existing debate instead of replacing it.

11. Output record
    Each returned record contains the block, profiles, debate transcript, graph,
    model summary, Judge output, optional training summary, reflection history,
    and post-hoc market verification. The output record may contain `label/p1`
    for analysis, but LLM Agent/Judge payload builders must not expose those
    fields.
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
from debate_graph.graph_batch import GraphTensor
from judge import create_judge_client, reflection_signal_from_judge
from model import GraphSentimentModel, TrainingConfig, train_graph_model
from profiles import ProfileStore
from scripts.run_debate import DEFAULT_INPUT
from verification import verify_market_behavior


@dataclass
class PipelineContext:
    """All reusable artifacts for one `CommentBlock`.

    The pipeline builds these artifacts in dependency order:
    `block -> profiles -> transcript -> graph -> graph_tensor`.
    Keeping them together avoids rebuilding upstream LLM outputs when later
    stages need to rerun model summary or Judge logic.
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
    debate_mode: str = "siliconflow",
    judge_mode: str = "siliconflow",
    reflection_rounds: int = 0,
    embedding_backend: str | None = None,
    debate_client: DebateClient | None = None,
    judge_client: object | None = None,
) -> list[dict[str, object]]:
    """Run debate, graph construction, model summary, Judge, and verification.

    The returned list is JSON-serializable so scripts can write it directly as
    JSONL. This function is also used by evaluation scripts, so it deliberately
    keeps every intermediate artifact that downstream metrics may need.
    """
    # Debate and Judge providers are created once and reused across samples.
    # Tests can pass fake clients here to exercise the whole pipeline without
    # external API calls.
    orchestrator = DebateOrchestrator(client=debate_client or create_debate_client(debate_mode))

    # Build every sample up to the graph tensor stage before model creation.
    # This matters because optional text embeddings change graph_tensor.x.shape[1],
    # and the model input dimension must match the actual tensor feature width.
    contexts = _build_contexts(
        input_path=input_path,
        limit_blocks=limit_blocks,
        rounds=rounds,
        orchestrator=orchestrator,
        embedding_backend=embedding_backend,
    )
    model = GraphSentimentModel(input_dim=_graph_input_dim(contexts))
    training_summary: dict[str, object] | None = None

    # Optional training stage. The tensors already contain labels, but those
    # labels are used only by the trainable graph model; they are not passed to
    # LLM agents or the LLM Judge.
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
        # The graph model converts graph_tensor into numeric evolution features.
        # The LLM Judge receives this summary plus the raw debate/graph objects.
        model_summary = model.summarize(context.graph_tensor)
        judge_output = judge.judge(context.transcript, model_summary, context.graph)
        reflection_history: list[dict[str, object]] = []

        # Optional Judge-guided reflection. Each iteration appends new debater
        # arguments, rebuilds all graph/tensor artifacts that depend on the
        # transcript, then asks the Judge again on the updated state.
        for _reflection_index in range(max(reflection_rounds, 0)):
            signal = reflection_signal_from_judge(
                judge_output,
                model_summary,
                mean_argument_confidence=_mean_argument_confidence(context.transcript),
            )
            reflection_history.append(signal.to_dict())
            if not should_continue_reflection(signal):
                break
            # Transcript changes first. Everything derived from the transcript
            # must then be rebuilt in order: hetero graph -> graph tensor ->
            # model summary -> Judge output.
            context.transcript = orchestrator.add_reflection_rounds(
                context.block,
                context.profiles,
                context.transcript,
                signal,
                reflection_rounds=1,
            )
            context.graph = build_hetero_graph(context.block, context.transcript)
            # Text embeddings are applied only when tensorizing the finished
            # debate graph for Bi-ODE; agent prompts stay raw text based.
            context.graph_tensor = graph_to_tensor(
                context.graph,
                label=context.block.label,
                embedding_backend=embedding_backend,
            )
            model_summary = model.summarize(context.graph_tensor)
            judge_output = judge.judge(context.transcript, model_summary, context.graph)

        # This is the persisted analysis record. It intentionally stores both
        # model-facing and human-facing artifacts so later scripts can compute
        # metrics, inspect cases, export reports, or replay decisions without
        # rerunning expensive LLM calls.
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
    parser.add_argument(
        "--embedding-backend",
        choices=["none", "sentencebert", "finbert", "sentencebert_finbert"],
        default=None,
    )
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
        embedding_backend=args.embedding_backend,
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
    embedding_backend: str | None,
) -> list[PipelineContext]:
    """Build per-sample contexts up to the tensor stage.

    This is the upstream half of the pipeline:
    source JSONL -> PostRecord -> CommentBlock -> user profiles -> transcript
    -> hetero graph -> graph tensor.
    """
    posts = load_posts(input_path)
    blocks, _issues = build_comment_blocks(posts)
    if limit_blocks is not None:
        blocks = blocks[:limit_blocks]

    # The profile store may index all selected blocks, but every profile lookup
    # below still uses strict t < block.t0 slicing.
    profile_store = ProfileStore.from_blocks(blocks)
    contexts: list[PipelineContext] = []
    for block in blocks:
        # Profiles are computed before debate because they are part of the LLM
        # agent input. The raw block text and safe profiles together form the
        # debate context.
        profiles = profile_store.get_profiles_for_block(block)
        transcript = orchestrator.run(block, profiles, rounds=rounds)
        graph = build_hetero_graph(block, transcript)
<<<<<<< Updated upstream
        # Tensorization is the handoff from symbolic/LLM artifacts to PyTorch.
        # The label is attached here for model training, not for LLM prompting.
=======
        # The optional embedding backend affects only this graph tensor for
        # Bi-ODE. It is not part of the bull/bear agent input.
>>>>>>> Stashed changes
        contexts.append(
            PipelineContext(
                block=block,
                profiles=profiles,
                transcript=transcript,
                graph=graph,
                graph_tensor=graph_to_tensor(graph, label=block.label, embedding_backend=embedding_backend),
            )
        )
    return contexts


def _graph_input_dim(contexts: list[PipelineContext]) -> int:
    if not contexts:
        raise ValueError("No graph contexts available")
    return int(contexts[0].graph_tensor.x.shape[1])


def _mean_argument_confidence(transcript: DebateTranscript) -> float | None:
    values = [argument.confidence for argument in transcript.arguments]
    if not values:
        return None
    return sum(values) / len(values)


if __name__ == "__main__":
    main()
