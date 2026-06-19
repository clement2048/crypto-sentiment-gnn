"""Single-relation diffusion operators for v3 graphs."""

from __future__ import annotations

from collections import defaultdict
from math import sqrt

from debate_graph.schema import HeteroGraph


def relation_adjacency(graph: HeteroGraph) -> dict[str, list[tuple[int, int, float]]]:
    """Return sparse adjacency triples grouped by relation."""
    index = graph.node_index()
    grouped: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
    for edge in graph.edges:
        grouped[edge.relation].append((index[edge.source], index[edge.target], edge.weight))
    return dict(grouped)


def normalized_relation_adjacency(graph: HeteroGraph) -> dict[str, list[tuple[int, int, float]]]:
    """Return symmetric normalized Laplacian triples per relation.

    v3 uses one relation, interact. We keep the relation-keyed return shape so
    existing tensorization code can continue to consume it.
    """
    grouped = relation_adjacency(graph)
    normalized: dict[str, list[tuple[int, int, float]]] = {}
    n_nodes = len(graph.nodes)
    for relation, triples in grouped.items():
        degrees: dict[int, float] = defaultdict(float)
        for source, target, weight in triples:
            degrees[source] += abs(weight)
            degrees[target] += abs(weight)

        values: dict[tuple[int, int], float] = {}
        for node_idx in range(n_nodes):
            if degrees[node_idx] > 0:
                values[(node_idx, node_idx)] = 1.0

        for source, target, weight in triples:
            denom = sqrt(degrees[source] * degrees[target]) if degrees[source] and degrees[target] else 0.0
            if denom:
                values[(source, target)] = values.get((source, target), 0.0) - weight / denom
        normalized[relation] = [(source, target, weight) for (source, target), weight in sorted(values.items())]
    return normalized



