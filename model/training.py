"""Training utilities for the Bi-ODE graph sentiment model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from config import (
    LEARNING_RATE,
    LOSS_WEIGHT_CLASSIFICATION,
    LOSS_WEIGHT_INITIAL_ALIGNMENT,
    LOSS_WEIGHT_MUTUAL_EXCLUSION,
    LOSS_WEIGHT_REGRESSION,
    LOSS_WEIGHT_SMOOTHNESS,
    TRAIN_EARLY_STOPPING_PATIENCE,
    TRAIN_MIN_DELTA,
)
from debate_graph.graph_batch import GraphTensor
from model.bdg_ode.pipeline import GraphForwardDetails
from model.losses import (
    classification_loss,
    initial_alignment_loss,
    mutual_exclusion_loss,
    regression_strength_loss,
    smoothness_loss,
)


@dataclass
class TrainingConfig:
    epochs: int
    learning_rate: float = LEARNING_RATE
    weight_classification: float = LOSS_WEIGHT_CLASSIFICATION
    weight_initial_alignment: float = LOSS_WEIGHT_INITIAL_ALIGNMENT
    weight_smoothness: float = LOSS_WEIGHT_SMOOTHNESS
    weight_mutual_exclusion: float = LOSS_WEIGHT_MUTUAL_EXCLUSION
    weight_regression: float = LOSS_WEIGHT_REGRESSION
    early_stopping_patience: int | None = TRAIN_EARLY_STOPPING_PATIENCE
    min_delta: float = TRAIN_MIN_DELTA
    checkpoint_path: str | None = None


@dataclass
class EpochMetrics:
    epoch: int
    total_loss: float
    classification: float
    initial_alignment: float
    smoothness: float
    mutual_exclusion: float
    regression: float
    accuracy: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass
class TrainingResult:
    epochs_ran: int
    best_loss: float
    history: list[EpochMetrics]
    checkpoint_path: str | None = None

    @property
    def train_losses(self) -> list[float]:
        return [item.total_loss for item in self.history]

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs_ran": self.epochs_ran,
            "best_loss": self.best_loss,
            "checkpoint_path": self.checkpoint_path,
            "history": [item.to_dict() for item in self.history],
        }


def train_graph_model(
    model: torch.nn.Module,
    tensors: list[GraphTensor],
    config: TrainingConfig,
) -> TrainingResult:
    """Train a graph model with classification plus implemented auxiliary losses."""
    if config.epochs <= 0:
        return TrainingResult(epochs_ran=0, best_loss=0.0, history=[], checkpoint_path=config.checkpoint_path)
    if not tensors:
        raise ValueError("No graph tensors available for training")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    history: list[EpochMetrics] = []

    for epoch in range(1, config.epochs + 1):
        totals = _empty_loss_totals()
        correct = 0
        optimizer.zero_grad()

        for graph in tensors:
            assert graph.label is not None
            details = model.forward_details(graph)
            losses = compute_loss_components(details, graph, config)
            losses["total"].backward()
            _accumulate(totals, losses)
            pred = 1.0 if float(details.probability.detach()) >= 0.5 else 0.0
            correct += int(pred == float(_label_on_device(graph, details.probability).detach()))

        optimizer.step()
        divisor = float(len(tensors))
        metrics = EpochMetrics(
            epoch=epoch,
            total_loss=totals["total"] / divisor,
            classification=totals["classification"] / divisor,
            initial_alignment=totals["initial_alignment"] / divisor,
            smoothness=totals["smoothness"] / divisor,
            mutual_exclusion=totals["mutual_exclusion"] / divisor,
            regression=totals["regression"] / divisor,
            accuracy=correct / len(tensors),
        )
        history.append(metrics)

        if metrics.total_loss + config.min_delta < best_loss:
            best_loss = metrics.total_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if config.early_stopping_patience is not None and stale_epochs >= config.early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint_path = _save_checkpoint(model, config, history, best_loss) if config.checkpoint_path else None
    return TrainingResult(
        epochs_ran=len(history),
        best_loss=best_loss if history else 0.0,
        history=history,
        checkpoint_path=checkpoint_path,
    )


def compute_loss_components(
    details: GraphForwardDetails,
    graph: GraphTensor,
    config: TrainingConfig,
) -> dict[str, torch.Tensor]:
    """Compute every implemented loss component for one graph."""
    label = _label_on_device(graph, details.probability)
    classification = classification_loss(details.probability, label)

    bull_seed_target = details.polarity_seed.clamp_min(0.0)
    bear_seed_target = (-details.polarity_seed).clamp_min(0.0)
    bull_initial_strength = torch.sigmoid(details.bull0.mean(dim=1, keepdim=True))
    bear_initial_strength = torch.sigmoid(details.bear0.mean(dim=1, keepdim=True))
    initial_alignment = (
        initial_alignment_loss(bull_initial_strength, bull_seed_target)
        + initial_alignment_loss(bear_initial_strength, bear_seed_target)
    ) / 2.0

    state_path = [
        torch.cat([bull_state, bear_state], dim=-1)
        for bull_state, bear_state in zip(details.bull_path, details.bear_path)
    ]
    smoothness = smoothness_loss(state_path)
    mutual_exclusion = mutual_exclusion_loss(torch.sigmoid(details.bull_t), torch.sigmoid(details.bear_t))
    regression = regression_strength_loss(details.probability, label)

    total = (
        config.weight_classification * classification
        + config.weight_initial_alignment * initial_alignment
        + config.weight_smoothness * smoothness
        + config.weight_mutual_exclusion * mutual_exclusion
        + config.weight_regression * regression
    )
    return {
        "total": total,
        "classification": classification,
        "initial_alignment": initial_alignment,
        "smoothness": smoothness,
        "mutual_exclusion": mutual_exclusion,
        "regression": regression,
    }


def _label_on_device(graph: GraphTensor, reference: torch.Tensor) -> torch.Tensor:
    assert graph.label is not None
    return graph.label.to(device=reference.device, dtype=reference.dtype)


def _empty_loss_totals() -> dict[str, float]:
    return {
        "total": 0.0,
        "classification": 0.0,
        "initial_alignment": 0.0,
        "smoothness": 0.0,
        "mutual_exclusion": 0.0,
        "regression": 0.0,
    }


def _accumulate(target: dict[str, float], values: dict[str, torch.Tensor]) -> None:
    for key, value in values.items():
        target[key] = target[key] + float(value.detach())


def _save_checkpoint(
    model: torch.nn.Module,
    config: TrainingConfig,
    history: list[EpochMetrics],
    best_loss: float,
) -> str:
    assert config.checkpoint_path is not None
    path = Path(config.checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "training_config": asdict(config),
            "history": [item.to_dict() for item in history],
            "best_loss": best_loss,
        },
        path,
    )
    return str(path)
