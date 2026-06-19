"""Common graph schemas for v2 graph construction."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

NodeType = Literal["comment", "argument"]
RelationType = Literal["interact"]


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    node_type: NodeType
    ref_id: str
    text: str = ""
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "ref_id": self.ref_id,
            "text": self.text,
            "attrs": self.attrs,
        }


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    relation: RelationType
    weight: float = 1.0
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": self.weight,
            "attrs": self.attrs,
        }


@dataclass
class HeteroGraph:
    graph_id: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]

    def relation_counts(self) -> dict[str, int]:
        return dict(Counter(edge.relation for edge in self.edges))

    def node_counts(self) -> dict[str, int]:
        return dict(Counter(node.node_type for node in self.nodes))

    def node_index(self) -> dict[str, int]:
        return {node.node_id: index for index, node in enumerate(self.nodes)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "node_counts": self.node_counts(),
            "relation_counts": self.relation_counts(),
        }


def comment_node_id(comment_id: str) -> str:
    return f"comment:{comment_id}"


def argument_node_id(argument_id: str) -> str:
    return f"argument:{argument_id}"



