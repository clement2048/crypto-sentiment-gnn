"""Tensorize v3 single-relation heterogeneous graphs.

Current node features are deterministic structural/text proxy features. Real
Sentence-BERT embeddings can replace the text proxy after the embedding model
and cache policy are fixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from config import COMMENT_DEPTH_SCALE, DEBATE_ROUND_SCALE, DEBATE_SEQUENCE_SCALE, NODE_FEATURE_DIM
from debate_graph.diffusion_ops import normalized_relation_adjacency
from debate_graph.schema import HeteroGraph


@dataclass
class GraphTensor:
    graph_id: str
    x: torch.Tensor
    relation_adjs: dict[str, torch.Tensor]
    label: torch.Tensor | None = None

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])


def graph_to_tensor(graph: HeteroGraph, label: int | None = None) -> GraphTensor:
    x = torch.tensor([_node_features(node) for node in graph.nodes], dtype=torch.float32)
    n_nodes = len(graph.nodes)
    relation_adjs: dict[str, torch.Tensor] = {}
    for relation, triples in normalized_relation_adjacency(graph).items():
        adj = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)
        for source, target, weight in triples:
            adj[source, target] = float(weight)
        relation_adjs[relation] = adj

    y = None
    if label is not None:
        y = torch.tensor([1.0 if label == 1 else 0.0], dtype=torch.float32)
    return GraphTensor(graph_id=graph.graph_id, x=x, relation_adjs=relation_adjs, label=y)


def _node_features(node) -> list[float]:
    attrs = node.attrs
    is_comment = 1.0 if node.node_type == "comment" else 0.0
    is_argument = 1.0 if node.node_type == "argument" else 0.0
    stance = attrs.get("stance") or attrs.get("camp")
    is_bull = 1.0 if stance == "bull" else 0.0
    is_bear = 1.0 if stance == "bear" else 0.0
    confidence = _to_float(attrs.get("confidence"))
    depth = min(_to_float(attrs.get("depth")) / COMMENT_DEPTH_SCALE, 1.0)
    round_value = min(_to_float(attrs.get("round")) / DEBATE_ROUND_SCALE, 1.0)
    seq_value = min(_to_float(attrs.get("seq")) / DEBATE_SEQUENCE_SCALE, 1.0)
    has_parent = 1.0 if attrs.get("parent_id") else 0.0
    t_index = min(_to_float(attrs.get("t_index")) / DEBATE_ROUND_SCALE, 1.0)
    evidence_count = min(float(len(attrs.get("evidence") or [])) / 10.0, 1.0)
    text_signal = _text_signal(node.text)
    features = [
        is_comment,
        is_argument,
        is_bull,
        is_bear,
        confidence,
        depth,
        round_value,
        seq_value,
        has_parent,
        t_index,
        evidence_count,
        text_signal,
    ]
    if len(features) != NODE_FEATURE_DIM:
        raise ValueError(f"NODE_FEATURE_DIM={NODE_FEATURE_DIM} does not match graph features={len(features)}")
    return features


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _text_signal(text: str) -> float:
    if not text:
        return 0.0
    return (sum(ord(char) for char in text[:256]) % 1000) / 1000.0
