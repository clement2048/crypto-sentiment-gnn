"""Evaluate binary BULLISH/BEARISH predictions against root-comment labels."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.llm_client import DebateClient
from config import DEFAULT_DEBATE_ROUNDS, FULL_PIPELINE_TRAIN_EPOCHS, LEARNING_RATE, PRINT_SAMPLES
from scripts.run_debate import DEFAULT_INPUT
from scripts.run_full_pipeline import run_full_pipeline


LABEL_BULLISH = 1
LABEL_BEARISH = -1


@dataclass
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "support": self.support,
        }


@dataclass
class EvaluationMetrics:
    total: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    bullish: ClassMetrics
    bearish: ClassMetrics
    confusion_matrix: dict[str, dict[str, int]]
    brier: float
    ece: float
    msed: float | None
    correlation: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "bullish": self.bullish.to_dict(),
            "bearish": self.bearish.to_dict(),
            "confusion_matrix": self.confusion_matrix,
            "brier": self.brier,
            "ece": self.ece,
            "msed": self.msed,
            "correlation": self.correlation,
        }


def evaluate_pipeline(
    input_path: str = DEFAULT_INPUT,
    limit_blocks: int | None = None,
    rounds: int = DEFAULT_DEBATE_ROUNDS,
    train_epochs: int = FULL_PIPELINE_TRAIN_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    debate_mode: str = "siliconflow",
    judge_mode: str = "siliconflow",
    embedding_backend: str | None = None,
    debate_client: DebateClient | None = None,
    judge_client: object | None = None,
) -> tuple[list[dict[str, object]], EvaluationMetrics]:
    records = run_full_pipeline(
        input_path=input_path,
        limit_blocks=limit_blocks,
        rounds=rounds,
        train_epochs=train_epochs,
        learning_rate=learning_rate,
        debate_mode=debate_mode,
        judge_mode=judge_mode,
        embedding_backend=embedding_backend,
        debate_client=debate_client,
        judge_client=judge_client,
    )
    return records, compute_metrics(records)


def compute_metrics(records: list[dict[str, object]]) -> EvaluationMetrics:
    """Compute binary accuracy, macro-F1, confusion matrix, and calibration metrics."""
    # Metric flow:
    # 1. Each record is produced by `run_full_pipeline`.
    # 2. `_true_label(record)` reads the root-comment label from record["block"].
    # 3. `_predicted_label(record)` converts the Judge verdict into 1/-1.
    # 4. Class metrics are computed from those `(truth, prediction)` pairs.
    # 5. Probability-style diagnostics read model/Judge confidence fields from
    #    the same records; they do not rerun the model or Judge.
    pairs = [(_true_label(record), _predicted_label(record)) for record in records]
    total = len(pairs)
    correct = sum(1 for truth, pred in pairs if truth == pred)

    bullish = _class_metrics(pairs, LABEL_BULLISH)
    bearish = _class_metrics(pairs, LABEL_BEARISH)
    macro_precision = (bullish.precision + bearish.precision) / 2.0
    macro_recall = (bullish.recall + bearish.recall) / 2.0
    macro_f1 = (bullish.f1 + bearish.f1) / 2.0

    confidences = [_confidence(record) for record in records]
    correctness = [1.0 if truth == pred else 0.0 for truth, pred in pairs]
    probs = [_bullish_probability(record) for record in records]
    truth_binary = [1.0 if truth == LABEL_BULLISH else 0.0 for truth, _pred in pairs]

    return EvaluationMetrics(
        total=total,
        accuracy=_safe_div(correct, total),
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        bullish=bullish,
        bearish=bearish,
        confusion_matrix=_confusion_matrix(pairs),
        brier=_brier(probs, truth_binary),
        ece=_ece(confidences, correctness),
        msed=_msed(probs, truth_binary),
        correlation=_correlation(probs, truth_binary),
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Evaluate binary full-pipeline predictions against CommentBlock labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--limit-blocks", type=int, default=None, help="Limit samples; omit for all blocks.")
    parser.add_argument("--rounds", type=int, default=DEFAULT_DEBATE_ROUNDS)
    parser.add_argument("--train-epochs", type=int, default=FULL_PIPELINE_TRAIN_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--debate-mode", choices=["siliconflow"], default="siliconflow")
    parser.add_argument("--judge-mode", choices=["siliconflow"], default="siliconflow")
    parser.add_argument(
        "--embedding-backend",
        choices=["none", "sentencebert", "finbert", "sentencebert_finbert"],
        default="sentencebert",
    )
    parser.add_argument("--output-jsonl", type=str, default=None)
    parser.add_argument("--metrics-json", type=str, default=None)
    args = parser.parse_args()

    records, metrics = evaluate_pipeline(
        input_path=args.input,
        limit_blocks=args.limit_blocks,
        rounds=args.rounds,
        train_epochs=args.train_epochs,
        learning_rate=args.learning_rate,
        debate_mode=args.debate_mode,
        judge_mode=args.judge_mode,
        embedding_backend=args.embedding_backend,
    )
    _print_metrics(metrics)

    for record in records[: min(len(records), PRINT_SAMPLES)]:
        block = record["block"]
        judge = record["judge"]
        assert isinstance(block, dict)
        assert isinstance(judge, dict)
        print(
            f"- {block['block_id']} | true={block['label']} "
            f"| pred={judge['verdict']} | confidence={judge['confidence']:.3f}"
        )

    if args.output_jsonl:
        _write_jsonl(args.output_jsonl, records)
    if args.metrics_json:
        _write_json(args.metrics_json, metrics.to_dict())


def _print_metrics(metrics: EvaluationMetrics) -> None:
    print(f"Evaluated samples: {metrics.total}")
    print(f"Accuracy: {metrics.accuracy:.4f}")
    print(
        "Macro: "
        f"precision={metrics.macro_precision:.4f} "
        f"recall={metrics.macro_recall:.4f} "
        f"f1={metrics.macro_f1:.4f}"
    )
    print(
        "Bullish: "
        f"precision={metrics.bullish.precision:.4f} "
        f"recall={metrics.bullish.recall:.4f} "
        f"f1={metrics.bullish.f1:.4f} "
        f"support={metrics.bullish.support}"
    )
    print(
        "Bearish: "
        f"precision={metrics.bearish.precision:.4f} "
        f"recall={metrics.bearish.recall:.4f} "
        f"f1={metrics.bearish.f1:.4f} "
        f"support={metrics.bearish.support}"
    )
    print(f"Confusion matrix: {metrics.confusion_matrix}")
    print(f"Brier: {metrics.brier:.4f}")
    print(f"ECE: {metrics.ece:.4f}")
    if metrics.msed is not None:
        print(f"MSED: {metrics.msed:.4f}")
    if metrics.correlation is not None:
        print(f"Correlation: {metrics.correlation:.4f}")


def _class_metrics(pairs: list[tuple[int, int]], label: int) -> ClassMetrics:
    tp = sum(1 for truth, pred in pairs if truth == label and pred == label)
    fp = sum(1 for truth, pred in pairs if truth != label and pred == label)
    fn = sum(1 for truth, pred in pairs if truth == label and pred != label)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    support = sum(1 for truth, _pred in pairs if truth == label)
    return ClassMetrics(precision=precision, recall=recall, f1=f1, support=support)


def _confusion_matrix(pairs: list[tuple[int, int]]) -> dict[str, dict[str, int]]:
    names = {
        LABEL_BULLISH: "bullish",
        LABEL_BEARISH: "bearish",
    }
    matrix = {
        "bullish": {"bullish": 0, "bearish": 0},
        "bearish": {"bullish": 0, "bearish": 0},
    }
    for truth, pred in pairs:
        true_name = names.get(truth)
        pred_name = names.get(pred)
        if true_name in matrix and pred_name in matrix[true_name]:
            matrix[true_name][pred_name] += 1
    return matrix


def _true_label(record: dict[str, object]) -> int:
    block = record.get("block")
    if not isinstance(block, dict):
        raise ValueError("Evaluation record missing block")
    label = block.get("label")
    if label not in (LABEL_BULLISH, LABEL_BEARISH):
        raise ValueError(f"Unsupported true label for evaluation: {label}")
    return int(label)


def _predicted_label(record: dict[str, object]) -> int:
    judge = record.get("judge")
    if not isinstance(judge, dict):
        raise ValueError("Evaluation record missing judge")
    verdict = judge.get("verdict")
    if verdict == "BULLISH":
        return LABEL_BULLISH
    if verdict == "BEARISH":
        return LABEL_BEARISH
    raise ValueError(f"Unsupported binary judge verdict for evaluation: {verdict}")


def _confidence(record: dict[str, object]) -> float:
    judge = record.get("judge", {})
    if isinstance(judge, dict):
        try:
            return float(judge.get("confidence", 0.0))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _bullish_probability(record: dict[str, object]) -> float:
    model_summary = record.get("model_summary", {})
    if isinstance(model_summary, dict):
        try:
            return float(model_summary.get("bullish_probability", 0.5))
        except (TypeError, ValueError):
            return 0.5
    judge = record.get("judge", {})
    if isinstance(judge, dict):
        verdict = str(judge.get("verdict", "")).upper()
        confidence = _confidence(record)
        if verdict == "BULLISH":
            return confidence
        if verdict == "BEARISH":
            return 1.0 - confidence
    return 0.5


def _brier(probs: list[float], truth: list[float]) -> float:
    return _safe_div(sum((p - y) ** 2 for p, y in zip(probs, truth)), len(probs))


def _msed(probs: list[float], truth: list[float]) -> float | None:
    if not probs:
        return None
    return _brier(probs, truth)


def _ece(confidences: list[float], correctness: list[float], bins: int = 10) -> float:
    if not confidences:
        return 0.0
    total = len(confidences)
    error = 0.0
    for idx in range(bins):
        lower = idx / bins
        upper = (idx + 1) / bins
        selected = [
            (conf, corr)
            for conf, corr in zip(confidences, correctness)
            if (lower <= conf < upper) or (idx == bins - 1 and conf == upper)
        ]
        if not selected:
            continue
        avg_conf = sum(item[0] for item in selected) / len(selected)
        avg_acc = sum(item[1] for item in selected) / len(selected)
        error += (len(selected) / total) * abs(avg_acc - avg_conf)
    return error


def _correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x == 0 or denom_y == 0:
        return None
    return numerator / (denom_x * denom_y)


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _write_jsonl(path: str, records: list[dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote JSONL: {output_path}")


def _write_json(path: str, data: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote metrics JSON: {output_path}")


if __name__ == "__main__":
    main()
