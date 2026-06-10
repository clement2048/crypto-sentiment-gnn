"""Relation-wise normalized adjacency operators."""

from __future__ import annotations

from collections import defaultdict

from debate_graph.schema import HeteroGraph


def relation_adjacency(graph: HeteroGraph) -> dict[str, list[tuple[int, int, float]]]:
    """Return sparse adjacency triples grouped by relation."""
    index = graph.node_index()
    grouped: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
    for edge in graph.edges:
        grouped[edge.relation].append((index[edge.source], index[edge.target], edge.weight))
    return dict(grouped)


def normalized_relation_adjacency(graph: HeteroGraph) -> dict[str, list[tuple[int, int, float]]]:
    """Row-normalize outgoing edge weights for every relation."""
    grouped = relation_adjacency(graph)
    normalized: dict[str, list[tuple[int, int, float]]] = {}
    for relation, triples in grouped.items():
        row_sums: dict[int, float] = defaultdict(float)
        for source, _target, weight in triples:
            row_sums[source] += abs(weight)
        normalized[relation] = [
            (source, target, weight / row_sums[source] if row_sums[source] else 0.0)
            for source, target, weight in triples
        ]
    return normalized



