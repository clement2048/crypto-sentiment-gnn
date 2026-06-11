"""Evaluate full pipeline predictions against root-comment labels.

当前评估把 CommentBlock.label 当作真实方向：
- 1 表示看涨
- -1 表示看跌

法官输出的 BULLISH/BEARISH 会映射回 1/-1；NEUTRAL 作为中立/弃权预测单独统计。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import DEFAULT_DEBATE_ROUNDS, FULL_PIPELINE_TRAIN_EPOCHS, LEARNING_RATE, PRINT_SAMPLES
from scripts.run_debate import DEFAULT_INPUT
from scripts.run_full_pipeline import run_full_pipeline


LABEL_BULLISH = 1
LABEL_BEARISH = -1
LABEL_NEUTRAL = 0


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
    directional_accuracy: float | None
    coverage: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    bullish: ClassMetrics
    bearish: ClassMetrics
    confusion_matrix: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "accuracy": self.accuracy,
            "directional_accuracy": self.directional_accuracy,
            "coverage": self.coverage,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "bullish": self.bullish.to_dict(),
            "bearish": self.bearish.to_dict(),
            "confusion_matrix": self.confusion_matrix,
        }


def evaluate_pipeline(
    input_path: str = DEFAULT_INPUT,
    limit_blocks: int | None = None,
    rounds: int = DEFAULT_DEBATE_ROUNDS,
    train_epochs: int = FULL_PIPELINE_TRAIN_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    debate_mode: str = "mock",
    judge_mode: str = "mock",
) -> tuple[list[dict[str, object]], EvaluationMetrics]:
    """运行完整 pipeline，并计算最终法官 verdict 相对真实 label 的指标。"""
    records = run_full_pipeline(
        input_path=input_path,
        limit_blocks=limit_blocks,
        rounds=rounds,
        train_epochs=train_epochs,
        learning_rate=learning_rate,
        debate_mode=debate_mode,
        judge_mode=judge_mode,
    )
    metrics = compute_metrics(records)
    return records, metrics


def compute_metrics(records: list[dict[str, object]]) -> EvaluationMetrics:
    """计算 accuracy、precision、recall、F1 和混淆矩阵。"""
    pairs = [(_true_label(record), _predicted_label(record)) for record in records]
    total = len(pairs)
    correct = sum(1 for truth, pred in pairs if truth == pred)
    non_neutral = [(truth, pred) for truth, pred in pairs if pred != LABEL_NEUTRAL]
    directional_correct = sum(1 for truth, pred in non_neutral if truth == pred)

    bullish = _class_metrics(pairs, LABEL_BULLISH)
    bearish = _class_metrics(pairs, LABEL_BEARISH)
    macro_precision = (bullish.precision + bearish.precision) / 2.0
    macro_recall = (bullish.recall + bearish.recall) / 2.0
    macro_f1 = (bullish.f1 + bearish.f1) / 2.0

    return EvaluationMetrics(
        total=total,
        accuracy=_safe_div(correct, total),
        directional_accuracy=(
            _safe_div(directional_correct, len(non_neutral))
            if non_neutral
            else None
        ),
        coverage=_safe_div(len(non_neutral), total),
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        bullish=bullish,
        bearish=bearish,
        confusion_matrix=_confusion_matrix(pairs),
    )


def main() -> None:
    """命令行入口：默认对全部 CommentBlock 做评估。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Evaluate full pipeline predictions against CommentBlock labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--limit-blocks", type=int, default=None, help="Limit samples; omit for all blocks.")
    parser.add_argument("--rounds", type=int, default=DEFAULT_DEBATE_ROUNDS)
    parser.add_argument("--train-epochs", type=int, default=FULL_PIPELINE_TRAIN_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--debate-mode", choices=["mock", "deepseek", "bailian", "minimax"], default="mock")
    parser.add_argument("--judge-mode", choices=["mock", "deepseek", "bailian", "minimax"], default="mock")
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
    print(f"Coverage(non-neutral): {metrics.coverage:.4f}")
    if metrics.directional_accuracy is None:
        print("Directional accuracy(non-neutral only): N/A")
    else:
        print(f"Directional accuracy(non-neutral only): {metrics.directional_accuracy:.4f}")
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
        LABEL_NEUTRAL: "neutral",
    }
    matrix = {
        "bullish": {"bullish": 0, "bearish": 0, "neutral": 0},
        "bearish": {"bullish": 0, "bearish": 0, "neutral": 0},
    }
    for truth, pred in pairs:
        true_name = names.get(truth)
        pred_name = names.get(pred, "neutral")
        if true_name in matrix:
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
    if verdict == "NEUTRAL":
        return LABEL_NEUTRAL
    raise ValueError(f"Unsupported judge verdict for evaluation: {verdict}")


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


