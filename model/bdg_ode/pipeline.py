"""端到端图情绪模型原型。

输入 GraphTensor，输出一个看涨概率。
当前模型是最小可训练版本，用来验证链路和反向传播，不是最终论文级模型。
"""

from __future__ import annotations

import torch
from torch import nn

from config import (
    CLASSIFICATION_THRESHOLD,
    DEFAULT_RELATIONS,
    MODEL_HIDDEN_DIM,
    ODE_STEPS,
)
from debate_graph.graph_batch import GraphTensor
from model.bdg_ode.calibrator import VerdictCalibrator
from model.bdg_ode.dual_encoder import DualEmotionEncoder
from model.bdg_ode.dynamics import BDGODEFunc
from model.bdg_ode.ode_solver import integrate_bdg_ode
from model.bdg_ode.readout import DualReadout
from model.model_summary import ModelOutputSummary


class GraphSentimentModel(nn.Module):
    """评论块级图分类模型。

    计算流程：
    1. DualEmotionEncoder：节点特征 -> bull/bear 两套初始状态
    2. BDGODEFunc + torchdiffeq：在多关系图上演化 bull/bear 状态
    3. DualReadout：把节点状态池化成图级向量
    4. VerdictCalibrator：输出看涨概率
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = MODEL_HIDDEN_DIM,
        relation_names: list[str] | None = None,
        ode_steps: int = ODE_STEPS,
    ):
        super().__init__()
        self.relation_names = relation_names or DEFAULT_RELATIONS
        self.ode_steps = ode_steps
        self.encoder = DualEmotionEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
        self.ode_func = BDGODEFunc(hidden_dim=hidden_dim, relation_names=self.relation_names)
        self.readout = DualReadout()
        self.calibrator = VerdictCalibrator(input_dim=hidden_dim * 5)

    def forward(self, graph: GraphTensor) -> torch.Tensor:
        """训练/推理用前向函数，返回 shape=(1,) 的看涨概率。"""
        bull_t, bear_t = self.forward_states(graph)
        graph_repr = self.readout(bull_t, bear_t)
        return self.calibrator(graph_repr)

    def forward_states(self, graph: GraphTensor) -> tuple[torch.Tensor, torch.Tensor]:
        """只跑到 ODE 演化状态，供 summarize 或调试使用。"""
        device = next(self.parameters()).device
        x = graph.x.to(device)
        relation_adjs = {name: adj.to(device) for name, adj in graph.relation_adjs.items()}
        # bull0 / bear0 是每个节点在初始时刻的双情绪状态。
        bull0, bear0 = self.encoder(x)
        return integrate_bdg_ode(
            self.ode_func,
            bull0,
            bear0,
            relation_adjs,
            steps=self.ode_steps,
        )

    @torch.no_grad()
    def summarize(self, graph: GraphTensor) -> ModelOutputSummary:
        """生成给法官看的模型摘要。

        法官不直接读神经网络张量，而是读这些可解释的统计量：
        - calibrator 的看涨概率
        - ODE 后 bull/bear 均值和最大值
        - bull_bear_margin
        """
        bull_t, bear_t = self.forward_states(graph)
        graph_repr = self.readout(bull_t, bear_t)
        prob = float(self.calibrator(graph_repr))
        bull_mean = float(bull_t.mean())
        bear_mean = float(bear_t.mean())
        return ModelOutputSummary(
            bullish_probability=prob,
            predicted_label=1 if prob >= CLASSIFICATION_THRESHOLD else -1,
            bull_mean=bull_mean,
            bear_mean=bear_mean,
            bull_max=float(bull_t.max()),
            bear_max=float(bear_t.max()),
            bull_bear_margin=bull_mean - bear_mean,
            ode_steps=self.ode_steps,
        )


