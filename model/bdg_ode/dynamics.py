"""Bi-Dynamic Graph ODE dynamics for graph node states.

This module receives the output of `DualEmotionEncoder`:

`bull(t)` and `bear(t)` are both tensors shaped `[num_nodes, hidden_dim]`.
They represent two coupled hidden-state channels on the same graph nodes.

One forward call computes derivatives:

1. Combine graph relation matrices into a graph operator.
   In the current v3 graph there is usually one relation, `interact`.
2. Propagate bull and bear states through that graph operator.
   This spreads each node's hidden state along debate interaction edges.
3. Apply four trainable maps:
   - bull -> bull self evolution;
   - bear -> bull cross influence;
   - bull -> bear cross influence;
   - bear -> bear self evolution.
4. Return non-negative derivative tensors for the ODE solver.

The function does not know labels, prices, comments, or LLM text. It only sees
numeric tensors produced by graph tensorization and the encoder.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from config import ODE_DROPOUT, ODE_USE_CONTROL, ODE_USE_GRAPH, RELATION_WEIGHT_INIT


class BDGODEFunc(nn.Module):
    """Derivative function used by the ODE solver.

    It implements the continuous-time transition:
    `(bull(t), bear(t), graph) -> (d_bull/dt, d_bear/dt)`.
    """

    def __init__(
        self,
        hidden_dim: int,
        relation_names: list[str],
        dropout: float = ODE_DROPOUT,
        use_graph: bool = ODE_USE_GRAPH,
        use_control: bool = ODE_USE_CONTROL,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.relation_names = relation_names
        self.use_graph = use_graph
        self.use_control = use_control
        self.dropout_layer = nn.Dropout(dropout)

        # 对应原仓库 ODEFunc1 中的 wt1/wt2/wt3/wt4。
        # bull_from_bull: bull 视角自身演化
        # bull_from_bear: bear 视角对 bull 视角的影响
        # bear_from_bull: bull 视角对 bear 视角的影响
        # bear_from_bear: bear 视角自身演化
        self.bull_from_bull = nn.Linear(hidden_dim, hidden_dim)
        self.bull_from_bear = nn.Linear(hidden_dim, hidden_dim)
        self.bear_from_bull = nn.Linear(hidden_dim, hidden_dim)
        self.bear_from_bear = nn.Linear(hidden_dim, hidden_dim)

        # 原始 BDG-ODE 接收单一图算子 A；我们当前是多关系图，所以先学习每类边
        # 的组合权重，再合成一个图算子。
        self.relation_weights = nn.ParameterDict(
            {name: nn.Parameter(torch.tensor(RELATION_WEIGHT_INIT)) for name in relation_names}
        )

    def forward(
        self,
        bull: torch.Tensor,
        bear: torch.Tensor,
        relation_adjs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Graph propagation stage. `relation_adjs` comes from `graph_to_tensor`.
        # If graph use is disabled, the ODE becomes a per-node channel evolution
        # without neighbor diffusion.
        if self.use_graph:
            graph_operator = self._combine_graph_operator(relation_adjs, bull)
            if graph_operator is not None:
                bull = graph_operator.matmul(bull)
                bear = graph_operator.matmul(bear)

        # Channel coupling stage. Cross terms let the bear channel affect bull
        # dynamics and vice versa. The model learns whether this cross influence
        # behaves like reinforcement, suppression, or correction.
        if self.use_control:
            d_bull = self.bull_from_bull(bull) + self.bull_from_bear(bear)
            d_bear = self.bear_from_bull(bull) + self.bear_from_bear(bear)
        else:
            d_bull = bull
            d_bear = bear

        d_bull = self.dropout_layer(d_bull)
        d_bear = self.dropout_layer(d_bear)
        return F.relu(d_bull), F.relu(d_bear)

    def _combine_graph_operator(
        self,
        relation_adjs: dict[str, torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor | None:
        """Combine relation adjacency matrices into one graph operator."""
        graph_operator = None
        for name in self.relation_names:
            adj = relation_adjs.get(name)
            if adj is None:
                continue
            adj = adj.to(device=reference.device, dtype=reference.dtype)
            weight = self.relation_weights[name]
            graph_operator = weight * adj if graph_operator is None else graph_operator + weight * adj
        return graph_operator


