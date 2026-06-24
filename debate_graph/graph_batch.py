"""Tensorize v3 single-relation heterogeneous graphs for Bi-ODE.

Bull/bear agents receive raw news, comments, and profile text. Text embeddings
are applied only when converting the finished debate graph into node features
for the graph ODE model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from debate_graph.diffusion_ops import normalized_relation_adjacency
from debate_graph.schema import HeteroGraph
from debate_graph.text_embeddings import encode_texts, normalize_embedding_backend, text_embedding_dim


@dataclass
class GraphTensor:
    graph_id: str
    x: torch.Tensor
    relation_adjs: dict[str, torch.Tensor]
    label: torch.Tensor | None = None

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])


def graph_to_tensor(
    graph: HeteroGraph,
    label: int | None = None,
    embedding_backend: str | None = None,
) -> GraphTensor:
    backend = normalize_embedding_backend(embedding_backend)
    embeddings = encode_texts([node.text for node in graph.nodes], backend)
    x = torch.tensor(embeddings, dtype=torch.float32)
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


def get_node_feature_dim(embedding_backend: str | None = None) -> int:
    return text_embedding_dim(embedding_backend)
