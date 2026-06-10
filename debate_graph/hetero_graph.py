"""融合评论图和辩论图。

评论图提供真实对话结构；辩论图提供 Agent 生成的论点关系。
融合后的 HeteroGraph 是后续图张量化和 ODE 模型的输入。
"""

from __future__ import annotations

from agent.schema import DebateTranscript
from data.schema import CommentBlock
from debate_graph.comment_graph import build_comment_graph
from debate_graph.debate_graph import build_debate_graph
from debate_graph.schema import GraphEdge, GraphNode, HeteroGraph


def build_hetero_graph(block: CommentBlock, transcript: DebateTranscript) -> HeteroGraph:
    """构建一个 block 对应的多关系异构图。"""
    comment_nodes, comment_edges = build_comment_graph(block)
    argument_nodes, argument_edges = build_debate_graph(transcript)

    # comment 和 argument 理论上 ID 前缀不同，但仍做一次去重，避免未来扩展时重复节点。
    nodes = _dedupe_nodes([*comment_nodes, *argument_nodes])
    node_ids = {node.node_id for node in nodes}
    # 只保留两端节点都存在的边。比如某个 argument 引用了不存在的 comment_id，就丢弃该边。
    edges = [
        edge
        for edge in [*comment_edges, *argument_edges]
        if edge.source in node_ids and edge.target in node_ids
    ]
    return HeteroGraph(graph_id=block.block_id, nodes=nodes, edges=edges)


def _dedupe_nodes(nodes: list[GraphNode]) -> list[GraphNode]:
    """按 node_id 去重，同时保留首次出现顺序。"""
    seen: set[str] = set()
    deduped: list[GraphNode] = []
    for node in nodes:
        if node.node_id in seen:
            continue
        seen.add(node.node_id)
        deduped.append(node)
    return deduped



