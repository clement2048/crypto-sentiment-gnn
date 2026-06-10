"""将 split/evaluate 输出 JSON 里的 precision/recall/F1 导出为 CSV。"""


from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def export_metrics_csv(input_json: str, output_csv: str) -> Path:
    """把 split/evaluate 输出 JSON 里的 precision/recall/F1 导出为 CSV。"""
    input_path = Path(input_json)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    rows = _metric_rows(data)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_file",
        "split",
        "class",
        "precision",
        "recall",
        "f1",
        "support",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "coverage",
        "directional_accuracy",
        "confusion_bullish_as_bullish",
        "confusion_bullish_as_bearish",
        "confusion_bullish_as_neutral",
        "confusion_bearish_as_bullish",
        "confusion_bearish_as_bearish",
        "confusion_bearish_as_neutral",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Export precision/recall/F1 metrics to CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    path = export_metrics_csv(args.input_json, args.output_csv)
    print(f"Wrote metrics CSV: {path}")


def _metric_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    source_file = str(data.get("config", {}).get("input_path", ""))
    metrics = data.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Input JSON does not contain a metrics object")

    rows: list[dict[str, Any]] = []
    for split_name, split_metrics in metrics.items():
        if split_metrics is None:
            continue
        for class_name in ("bullish", "bearish"):
            class_metrics = split_metrics.get(class_name, {})
            matrix = split_metrics.get("confusion_matrix", {})
            bull_row = matrix.get("bullish", {})
            bear_row = matrix.get("bearish", {})
            rows.append(
                {
                    "source_file": source_file,
                    "split": split_name,
                    "class": class_name,
                    "precision": class_metrics.get("precision"),
                    "recall": class_metrics.get("recall"),
                    "f1": class_metrics.get("f1"),
                    "support": class_metrics.get("support"),
                    "accuracy": split_metrics.get("accuracy"),
                    "macro_precision": split_metrics.get("macro_precision"),
                    "macro_recall": split_metrics.get("macro_recall"),
                    "macro_f1": split_metrics.get("macro_f1"),
                    "coverage": split_metrics.get("coverage"),
                    "directional_accuracy": split_metrics.get("directional_accuracy"),
                    "confusion_bullish_as_bullish": bull_row.get("bullish"),
                    "confusion_bullish_as_bearish": bull_row.get("bearish"),
                    "confusion_bullish_as_neutral": bull_row.get("neutral"),
                    "confusion_bearish_as_bullish": bear_row.get("bullish"),
                    "confusion_bearish_as_bearish": bear_row.get("bearish"),
                    "confusion_bearish_as_neutral": bear_row.get("neutral"),
                }
            )
    return rows


if __name__ == "__main__":
    main()
