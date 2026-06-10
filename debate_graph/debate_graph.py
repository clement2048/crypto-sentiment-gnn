"""Build argument relation graphs from debate transcripts."""

from __future__ import annotations

from collections import defaultdict

from agent.schema import Argument, DebateTranscript
from debate_graph.schema import (
    GraphEdge,
    GraphNode,
    argument_node_id,
    comment_node_id,
)


def build_debate_graph(transcript: DebateTranscript) -> tuple[list[GraphNode], list[GraphEdge]]:
    """把辩论 transcript 转成“论点节点 + 语义关系边”。

    论文 v4 中时间信息是节点/边属性，不是独立关系。因此这里不再构造
    precede 边，而是在 argument 节点上保存 round/seq/phase/relative_time，
    在部分边上保存 delta_t。
    """
    max_seq_by_round = _max_seq_by_round(transcript.arguments)
    nodes = [
        _argument_node(argument, max_seq_by_round.get(argument.round, 1))
        for argument in transcript.arguments
    ]
    by_id = {argument.argument_id: argument for argument in transcript.arguments}

    edges: list[GraphEdge] = []
    edges.extend(_cite_edges(transcript.arguments))
    edges.extend(_target_edges(transcript.arguments, by_id))
    edges.extend(_support_edges(transcript.arguments))
    edges.extend(_propose_edges(transcript.arguments))
    return nodes, edges


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


def _cite_edges(arguments: list[Argument]) -> list[GraphEdge]:
    """论点引用评论证据时，生成 argument -> comment 的 cite 边。"""
    edges: list[GraphEdge] = []
    for argument in arguments:
        for comment_id in argument.cited_comment_ids:
            edges.append(
                GraphEdge(
                    source=argument_node_id(argument.argument_id),
                    target=comment_node_id(comment_id),
                    relation="cite",
                    weight=1.0,
                )
            )
    return edges


def _target_edges(arguments: list[Argument], by_id: dict[str, Argument]) -> list[GraphEdge]:
    """根据 argument.targets 生成语义边。

    同阵营 target 表示支持/修正，记为 support。
    跨阵营 target 表示攻击，并额外保留 respond 表达“回应关系”。
    """
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
            relation = "attack" if target.camp != argument.camp else "support"
            edges.append(
                GraphEdge(
                    source=argument_node_id(argument.argument_id),
                    target=argument_node_id(target_id),
                    relation=relation,
                    weight=argument.confidence,
                    attrs=attrs,
                )
            )
            if target.camp != argument.camp:
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


def _support_edges(arguments: list[Argument]) -> list[GraphEdge]:
    """补充同阵营内部连续论点的支持链。

    这条链表达“同一阵营内论证逐步累积”，不是时间先后本身；时间先后只在
    delta_t/relative_time 属性里。
    """
    by_camp: dict[str, list[Argument]] = defaultdict(list)
    max_seq_by_round = _max_seq_by_round(arguments)
    for argument in sorted(arguments, key=lambda item: (item.round, item.seq)):
        by_camp[argument.camp].append(argument)

    edges: list[GraphEdge] = []
    for camp_arguments in by_camp.values():
        for previous, current in zip(camp_arguments, camp_arguments[1:]):
            edges.append(
                GraphEdge(
                    source=argument_node_id(previous.argument_id),
                    target=argument_node_id(current.argument_id),
                    relation="support",
                    weight=(previous.confidence + current.confidence) / 2,
                    attrs={
                        "delta_t": _argument_time(current, max_seq_by_round.get(current.round, 1))
                        - _argument_time(previous, max_seq_by_round.get(previous.round, 1))
                    },
                )
            )
    return edges


def _propose_edges(arguments: list[Argument]) -> list[GraphEdge]:
    """连接同一个 agent 在不同阶段提出的连续论点。"""
    by_agent: dict[str, list[Argument]] = defaultdict(list)
    max_seq_by_round = _max_seq_by_round(arguments)
    for argument in sorted(arguments, key=lambda item: (item.round, item.seq)):
        by_agent[argument.agent_id].append(argument)

    edges: list[GraphEdge] = []
    for agent_arguments in by_agent.values():
        for previous, current in zip(agent_arguments, agent_arguments[1:]):
            edges.append(
                GraphEdge(
                    source=argument_node_id(previous.argument_id),
                    target=argument_node_id(current.argument_id),
                    relation="propose",
                    attrs={
                        "delta_t": _argument_time(current, max_seq_by_round.get(current.round, 1))
                        - _argument_time(previous, max_seq_by_round.get(previous.round, 1))
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
    """把 round/seq 转成连续相对时间，只作为属性使用。"""
    seq_scale = max(max_seq, 1)
    return float(argument.round - 1) + float(argument.seq - 1) / seq_scale
