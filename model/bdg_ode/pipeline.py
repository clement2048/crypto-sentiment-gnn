"""End-to-end Bi-ODE graph sentiment model.

This module connects tensorized graphs to trainable predictions:

1. Input from graph layer
   `GraphTensor.x` is the node feature matrix and `relation_adjs` contains the
   normalized graph operators for each relation.

2. Dual initialization
   `DualEmotionEncoder` maps each node feature row into two initial hidden
   states: `bull0` and `bear0`.

3. Continuous graph evolution
   `integrate_bdg_ode_path` repeatedly calls `BDGODEFunc` to evolve the two
   channels over graph structure. The result is terminal states `bull_t`,
   `bear_t`, plus sampled paths for auxiliary losses.

4. Graph readout
   `DualReadout` pools terminal node states into one graph-level vector.

5. Probability calibration
   `VerdictCalibrator` maps that graph vector to `bullish_probability`.

6. Judge-facing summary
   `summarize(...)` converts internal tensors into a compact
   `ModelOutputSummary` so the LLM Judge can use numeric model evidence without
   seeing labels or future prices.
"""

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
    """Intermediate tensors returned during training.

    Training losses need more than the final probability:
    - `bull0/bear0` support initial-alignment losses;
    - `bull_path/bear_path` support smoothness losses;
    - `bull_t/bear_t` support mutual-exclusion and summary statistics;
    - `polarity_seed` exposes the learned scalar initialization cue.
    """

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
    """Graph classifier with bull/bear channels and Bi-ODE evolution."""

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
        """Return bullish probability with shape `(1,)`."""
        return self.forward_details(graph).probability

    def forward_details(self, graph: GraphTensor) -> GraphForwardDetails:
        """Run the full differentiable graph-model path.

        Flow inside one forward pass:
        graph tensor -> node feature matrix on model device -> dual encoder
        -> ODE integration -> graph readout -> probability.
        """
        device = next(self.parameters()).device
        x = graph.x.to(device)
        relation_adjs = {name: adj.to(device) for name, adj in graph.relation_adjs.items()}

        # Encode static node features into the initial conditions of the two
        # continuous-time channels.
        bull0, bear0, polarity_seed = self.encoder.forward_with_seed(x)

        # Evolve bull/bear node states over the debate/comment graph. The path
        # is retained so training can regularize the trajectory, not just the
        # terminal probability.
        bull_t, bear_t, bull_path, bear_path = integrate_bdg_ode_path(
            self.ode_func,
            bull0,
            bear0,
            relation_adjs,
            steps=self.ode_steps,
        )

        # Collapse node-level terminal states into one graph-level vector and
        # map it to a calibrated bullish probability.
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
        """Create the structured model summary consumed by the LLM Judge.

        This method intentionally exports only model-derived diagnostics:
        probability, channel means/maxima, bull-bear margins, and relation count.
        It does not expose `graph.label`, `p1`, or any future market value.
        """
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
