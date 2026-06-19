"""End-to-end Bi-ODE graph sentiment model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from config import DEFAULT_RELATIONS, MODEL_HIDDEN_DIM, ODE_STEPS
from debate_graph.graph_batch import GraphTensor
from model.bdg_ode.calibrator import VerdictCalibrator
from model.bdg_ode.dual_encoder import DualEmotionEncoder
from model.bdg_ode.dynamics import BDGODEFunc
from model.bdg_ode.ode_solver import integrate_bdg_ode, integrate_bdg_ode_path
from model.bdg_ode.readout import DualReadout
from model.model_summary import ModelOutputSummary


@dataclass
class GraphForwardDetails:
    probability: torch.Tensor
    graph_repr: torch.Tensor
    bull0: torch.Tensor
    bear0: torch.Tensor
    bull_t: torch.Tensor
    bear_t: torch.Tensor
    bull_path: list[torch.Tensor]
    bear_path: list[torch.Tensor]
    polarity_seed: torch.Tensor


class GraphSentimentModel(nn.Module):
    """Root-comment graph classifier with bull/bear channels and Bi-ODE evolution."""

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
        """Return bullish probability with shape (1,)."""
        return self.forward_details(graph).probability

    def forward_details(self, graph: GraphTensor) -> GraphForwardDetails:
        """Return probability plus intermediate states for auxiliary training losses."""
        device = next(self.parameters()).device
        x = graph.x.to(device)
        relation_adjs = {name: adj.to(device) for name, adj in graph.relation_adjs.items()}
        bull0, bear0, polarity_seed = self.encoder.forward_with_seed(x)
        bull_t, bear_t, bull_path, bear_path = integrate_bdg_ode_path(
            self.ode_func,
            bull0,
            bear0,
            relation_adjs,
            steps=self.ode_steps,
        )
        graph_repr = self.readout(bull_t, bear_t)
        probability = self.calibrator(graph_repr)
        return GraphForwardDetails(
            probability=probability,
            graph_repr=graph_repr,
            bull0=bull0,
            bear0=bear0,
            bull_t=bull_t,
            bear_t=bear_t,
            bull_path=bull_path,
            bear_path=bear_path,
            polarity_seed=polarity_seed,
        )

    def forward_states(self, graph: GraphTensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return terminal bull/bear ODE states for summaries and debugging."""
        device = next(self.parameters()).device
        x = graph.x.to(device)
        relation_adjs = {name: adj.to(device) for name, adj in graph.relation_adjs.items()}
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
        """Create the structured model summary consumed by the LLM judge."""
        bull_t, bear_t = self.forward_states(graph)
        graph_repr = self.readout(bull_t, bear_t)
        prob = float(self.calibrator(graph_repr))
        bull_mean = float(bull_t.mean())
        bear_mean = float(bear_t.mean())
        net_scores = torch.sigmoid(bull_t).mean(dim=1) - torch.sigmoid(bear_t).mean(dim=1)
        return ModelOutputSummary(
            bullish_probability=prob,
            bull_mean=bull_mean,
            bear_mean=bear_mean,
            bull_max=float(bull_t.max()),
            bear_max=float(bear_t.max()),
            bull_bear_margin=bull_mean - bear_mean,
            ode_steps=self.ode_steps,
            net_score_mean=float(net_scores.mean()),
            net_score_max=float(net_scores.abs().max()),
            relation_count=len(graph.relation_adjs),
        )
