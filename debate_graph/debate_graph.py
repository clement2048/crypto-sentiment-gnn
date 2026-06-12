"""Build argument relation graphs from debate transcripts."""

from __future__ import annotations

from collections import defaultdict

from agent.schema import Argument, DebateTranscript
from debate_graph.schema import (
    GraphEdge,
    GraphNode,
    argument_node_id,
)


def build_debate_graph(transcript: DebateTranscript) -> tuple[list[GraphNode], list[GraphEdge]]:
    """把辩论 transcript 转成“论点节点 + 回应关系边”。

    简化版双 agent 辩论不再把 evidence/citation 建成图边；引用内容保留在
    Argument 文本和 evidence 字段里。图结构只表达“哪条论点回应哪条论点”。
    """
    max_seq_by_round = _max_seq_by_round(transcript.arguments)
    nodes = [
        _argument_node(argument, max_seq_by_round.get(argument.round, 1))
        for argument in transcript.arguments
    ]
    by_id = {argument.argument_id: argument for argument in transcript.arguments}
    return nodes, _respond_edges(transcript.arguments, by_id)


def _argument_node(argument: Argument, max_seq: int) -> GraphNode:
    """创建论点节点，并保存论文中的时间属性。"""
    return GraphNode(
        node_id=argument_node_id(argument.argument_id),
        node_type="argument",
        ref_id=argument.argument_id,
        text=argument.claim,
        attrs={
            "agent_id": argument.agent_id,
            "camp": argument.camp,
            "role": argument.role,
            "confidence": argument.confidence,
            "round": argument.round,
            "seq": argument.seq,
            "phase": argument.phase,
            "relative_time": _argument_time(argument, max_seq),
        },
    )


def _respond_edges(arguments: list[Argument], by_id: dict[str, Argument]) -> list[GraphEdge]:
    """根据 argument.targets 生成 respond 边。"""
    edges: list[GraphEdge] = []
    max_seq_by_round = _max_seq_by_round(arguments)
    for argument in arguments:
        for target_id in argument.targets:
            target = by_id.get(target_id)
            if target is None:
                continue
            attrs = {
                "delta_t": _argument_time(argument, max_seq_by_round.get(argument.round, 1))
                - _argument_time(target, max_seq_by_round.get(target.round, 1))
            }
            edges.append(
                GraphEdge(
                    source=argument_node_id(argument.argument_id),
                    target=argument_node_id(target_id),
                    relation="respond",
                    weight=argument.confidence,
                    attrs=attrs,
                )
            )
    return edges


def _max_seq_by_round(arguments: list[Argument]) -> dict[int, int]:
    result: dict[int, int] = defaultdict(int)
    for argument in arguments:
        result[argument.round] = max(result[argument.round], argument.seq)
    return dict(result)


def _argument_time(argument: Argument, max_seq: int) -> float:
    """把 round/seq 转成连续相对时间，只作为属性使用。"""
    seq_scale = max(max_seq, 1)
    return float(argument.round - 1) + float(argument.seq - 1) / seq_scale
