"""双视角图 ODE 演替模块。

核心结构参考：
tsinghua-fib-lab/Bi-Dynamic-Graph-ODE-for-Opinion-Evolution 的 ``ODEFunc1``。

原仓库的 ODEFunc1 做法是：
1. 把节点状态拆成正向/负向两半；
2. 两半状态分别经过图算子传播；
3. 用四个线性映射建模正负视角之间的相互影响；
4. 拼回双视角状态并经过 ReLU。

这里没有直接绑定原仓库的数据读取脚本，而是把这个演替函数适配到我们当前的
CommentBlock -> HeteroGraph -> GraphTensor 流程。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from config import ODE_DROPOUT, ODE_USE_CONTROL, ODE_USE_GRAPH, RELATION_WEIGHT_INIT


class BDGODEFunc(nn.Module):
    """基于 BDG-ODE 双视角思想的节点状态演替函数。

    输入是每个节点的 bull/bear 两套隐藏状态，以及多关系邻接矩阵。
    输出是 bull/bear 两套状态的导数，供 Euler 或其他 ODE solver 积分。
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
        if self.use_graph:
            graph_operator = self._combine_graph_operator(relation_adjs, bull)
            if graph_operator is not None:
                bull = graph_operator.matmul(bull)
                bear = graph_operator.matmul(bear)

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
        """把多关系邻接矩阵合成为原 BDG-ODE 形式中的单一图算子 A。"""
        graph_operator = None
        for name in self.relation_names:
            adj = relation_adjs.get(name)
            if adj is None:
                continue
            adj = adj.to(device=reference.device, dtype=reference.dtype)
            weight = self.relation_weights[name]
            graph_operator = weight * adj if graph_operator is None else graph_operator + weight * adj
        return graph_operator



