"""把 HeteroGraph 转成 PyTorch 张量。

- 节点 -> 固定 8 维结构特征矩阵 x
- 关系边 -> 每种关系一个邻接矩阵（当前主要是 reply/respond）
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from config import (
    COMMENT_DEPTH_SCALE,
    DEBATE_ROUND_SCALE,
    DEBATE_SEQUENCE_SCALE,
    NODE_FEATURE_DIM,
)
from debate_graph.diffusion_ops import normalized_relation_adjacency
from debate_graph.schema import HeteroGraph


@dataclass
class GraphTensor:
    """模型实际接收的图数据。

    x: (num_nodes, NODE_FEATURE_DIM)
    relation_adjs: 每个 relation 一个 (num_nodes, num_nodes) 邻接矩阵
    label: 训练时使用，1.0=看涨，0.0=看跌
    """

    graph_id: str
    x: torch.Tensor
    relation_adjs: dict[str, torch.Tensor]
    label: torch.Tensor | None = None

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])


def graph_to_tensor(graph: HeteroGraph, label: int | None = None) -> GraphTensor:
    """将异构图转换成固定维度节点特征和 dense 邻接矩阵。"""
    x = torch.tensor([_node_features(node) for node in graph.nodes], dtype=torch.float32)
    n_nodes = len(graph.nodes)
    relation_adjs: dict[str, torch.Tensor] = {}
    for relation, triples in normalized_relation_adjacency(graph).items():
        # 原型阶段用 dense 矩阵更直观；数据变大后这里应换成 sparse tensor。
        adj = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)
        for source, target, weight in triples:
            adj[source, target] = float(weight)
        relation_adjs[relation] = adj

    y = None
    if label is not None:
        y = torch.tensor([1.0 if label == 1 else 0.0], dtype=torch.float32)
    return GraphTensor(graph_id=graph.graph_id, x=x, relation_adjs=relation_adjs, label=y)


def _node_features(node) -> list[float]:
    """把一个节点压成 8 个结构特征。

    这不是最终特征工程，只是为了让模型原型可训练：
    [是否评论节点, 是否论点节点, 是否 bull, 是否 bear,
     confidence, comment depth, debate round, debate seq]
    """
    attrs = node.attrs
    is_comment = 1.0 if node.node_type == "comment" else 0.0
    is_argument = 1.0 if node.node_type == "argument" else 0.0
    camp = attrs.get("camp")
    is_bull = 1.0 if camp == "bull" else 0.0
    is_bear = 1.0 if camp == "bear" else 0.0
    confidence = _to_float(attrs.get("confidence"))
    depth = min(_to_float(attrs.get("depth")) / COMMENT_DEPTH_SCALE, 1.0)
    round_value = min(_to_float(attrs.get("round")) / DEBATE_ROUND_SCALE, 1.0)
    seq_value = min(_to_float(attrs.get("seq")) / DEBATE_SEQUENCE_SCALE, 1.0)
    return [
        is_comment,
        is_argument,
        is_bull,
        is_bear,
        confidence,
        depth,
        round_value,
        seq_value,
    ]


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


