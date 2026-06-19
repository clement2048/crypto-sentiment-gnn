"""Build v3 single-relation debate graphs from debate transcripts."""

from __future__ import annotations

from collections import defaultdict

from agent.schema import Argument, DebateTranscript
from debate_graph.schema import GraphEdge, GraphNode, argument_node_id


def build_debate_graph(transcript: DebateTranscript) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Convert arguments into nodes and target_args into interact edges."""
    max_seq_by_round = _max_seq_by_round(transcript.arguments)
    nodes = [
        _argument_node(argument, max_seq_by_round.get(argument.round, 1))
        for argument in transcript.arguments
    ]
    by_id = {argument.argument_id: argument for argument in transcript.arguments}
    return nodes, _interact_edges(transcript.arguments, by_id)


def _argument_node(argument: Argument, max_seq: int) -> GraphNode:
    t_index = argument.t_index or _argument_time(argument, max_seq)
    return GraphNode(
        node_id=argument_node_id(argument.argument_id),
        node_type="argument",
        ref_id=argument.argument_id,
        text=argument.claim,
        attrs={
            "agent_id": argument.agent_id,
            "camp": argument.camp,
            "stance": argument.camp,
            "role": argument.role,
            "confidence": argument.confidence,
            "round": argument.round,
            "seq": argument.seq,
            "phase": argument.phase,
            "target_args": argument.target_args,
            "evidence": [item.to_dict() for item in argument.evidence],
            "t_index": t_index,
        },
    )


def _interact_edges(arguments: list[Argument], by_id: dict[str, Argument]) -> list[GraphEdge]:
    """Build only interact edges; support/rebuttal/respond semantics stay in node attrs."""
    edges: list[GraphEdge] = []
    max_seq_by_round = _max_seq_by_round(arguments)
    for argument in arguments:
        source_time = argument.t_index or _argument_time(argument, max_seq_by_round.get(argument.round, 1))
        for target_id in argument.target_args:
            target = by_id.get(target_id)
            if target is None:
                continue
            target_time = target.t_index or _argument_time(target, max_seq_by_round.get(target.round, 1))
            edges.append(
                GraphEdge(
                    source=argument_node_id(argument.argument_id),
                    target=argument_node_id(target_id),
                    relation="interact",
                    weight=argument.confidence,
                    attrs={
                        "delta_t": source_time - target_time,
                        "source_camp": argument.camp,
                        "target_camp": target.camp,
                    },
                )
            )
    return edges


def _max_seq_by_round(arguments: list[Argument]) -> dict[int, int]:
    result: dict[int, int] = defaultdict(int)
    for argument in arguments:
        result[argument.round] = max(result[argument.round], argument.seq)
    return dict(result)


def _argument_time(argument: Argument, max_seq: int) -> float:
    return float(argument.round - 1) + float(argument.seq - 1) / max(max_seq, 1)
