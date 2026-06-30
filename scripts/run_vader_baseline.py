"""VADER lexicon baseline.

For each CommentBlock, run VADER on the root comment text and predict
BULLISH / BEARISH using the standard compound thresholds (>= 0.05 / <= -0.05).
Samples whose compound falls in the neutral band are treated as NEUTRAL; the
script reports metrics on the full set (NEUTRAL bucketed into the majority
class for a single prediction) and on the non-neutral subset only.

Usage:
    python scripts/run_vader_baseline.py
    python scripts/run_vader_baseline.py --limit-blocks 50
    python scripts/run_vader_baseline.py --output-jsonl outputs/vader_predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_INPUT_PATH  # noqa: E402
from data import build_comment_blocks, load_posts  # noqa: E402

DEFAULT_INPUT = DEFAULT_INPUT_PATH

BASELINE_DIR = Path(__file__).resolve().parent.parent / "baseline" / "vaderSentiment"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # noqa: E402

LABEL_BULLISH = 1
LABEL_BEARISH = -1
POS_THRESHOLD = 0.05
NEG_THRESHOLD = -0.05


def _predict_analyzer(analyzer: SentimentIntensityAnalyzer, text: str) -> tuple[str, float, dict[str, float]]:
    scores = analyzer.polarity_scores(text or "")
    compound = float(scores.get("compound", 0.0))
    if compound >= POS_THRESHOLD:
        label = "BULLISH"
    elif compound <= NEG_THRESHOLD:
        label = "BEARISH"
    else:
        label = "NEUTRAL"
    return label, compound, scores


def _majority_label(blocks: list[object]) -> int:
    counts: dict[int, int] = {}
    for block in blocks:
        label = getattr(block, "label", None)
        if label in (LABEL_BULLISH, LABEL_BEARISH):
            counts[label] = counts.get(label, 0) + 1
    if not counts:
        return LABEL_BULLISH
    return max(counts.items(), key=lambda item: item[1])[0]


def _to_predicted_int(label: str, fallback: int) -> int:
    if label == "BULLISH":
        return LABEL_BULLISH
    if label == "BEARISH":
        return LABEL_BEARISH
    return fallback


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _class_metrics(pairs: list[tuple[int, int]], label: int) -> dict[str, float]:
    tp = sum(1 for truth, pred in pairs if truth == label and pred == label)
    fp = sum(1 for truth, pred in pairs if truth != label and pred == label)
    fn = sum(1 for truth, pred in pairs if truth == label and pred != label)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    support = sum(1 for truth, _pred in pairs if truth == label)
    return {"precision": precision, "recall": recall, "f1": f1, "support": support}


def _confusion_matrix(pairs: list[tuple[int, int]]) -> dict[str, dict[str, int]]:
    matrix = {
        "bullish": {"bullish": 0, "bearish": 0},
        "bearish": {"bullish": 0, "bearish": 0},
    }
    for truth, pred in pairs:
        true_name = "bullish" if truth == LABEL_BULLISH else "bearish"
        pred_name = "bullish" if pred == LABEL_BULLISH else "bearish"
        matrix[true_name][pred_name] += 1
    return matrix


def run_vader_baseline(
    input_path: str = DEFAULT_INPUT,
    limit_blocks: int | None = None,
) -> dict[str, object]:
    posts = load_posts(input_path)
    if limit_blocks is not None:
        posts = posts[:limit_blocks]
    blocks, issues = build_comment_blocks(posts)
    if limit_blocks is not None:
        blocks = blocks[:limit_blocks]

    analyzer = SentimentIntensityAnalyzer()
    fallback = _majority_label(blocks)

    rows: list[dict[str, object]] = []
    full_pairs: list[tuple[int, int]] = []
    non_neutral_pairs: list[tuple[int, int]] = []
    neutral_count = 0

    for block in blocks:
        text = block.root_comment.text or ""
        pred_label, compound, scores = _predict_analyzer(analyzer, text)
        truth = int(block.label)
        pred_int = _to_predicted_int(pred_label, fallback)
        full_pairs.append((truth, pred_int))
        if pred_label == "NEUTRAL":
            neutral_count += 1
        else:
            non_neutral_pairs.append((truth, pred_int))

        rows.append(
            {
                "block_id": block.block_id,
                "true_label": truth,
                "pred_label": pred_label,
                "pred_int_with_fallback": pred_int,
                "compound": compound,
                "scores": scores,
                "text": text,
            }
        )

    full_metrics = _compute_metrics(full_pairs)
    non_neutral_metrics = _compute_metrics(non_neutral_pairs) if non_neutral_pairs else None

    return {
        "config": {
            "input": input_path,
            "limit_blocks": limit_blocks,
            "pos_threshold": POS_THRESHOLD,
            "neg_threshold": NEG_THRESHOLD,
            "neutral_fallback_label": fallback,
        },
        "blocks_total": len(blocks),
        "neutral_count": neutral_count,
        "non_neutral_count": len(non_neutral_pairs),
        "full_metrics": full_metrics,
        "non_neutral_metrics": non_neutral_metrics,
        "predictions": rows,
    }


def _compute_metrics(pairs: list[tuple[int, int]]) -> dict[str, object]:
    if not pairs:
        return {
            "total": 0,
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "bullish": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
            "bearish": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
            "confusion_matrix": {
                "bullish": {"bullish": 0, "bearish": 0},
                "bearish": {"bullish": 0, "bearish": 0},
            },
        }
    total = len(pairs)
    correct = sum(1 for truth, pred in pairs if truth == pred)
    bullish = _class_metrics(pairs, LABEL_BULLISH)
    bearish = _class_metrics(pairs, LABEL_BEARISH)
    return {
        "total": total,
        "accuracy": _safe_div(correct, total),
        "macro_precision": (bullish["precision"] + bearish["precision"]) / 2.0,
        "macro_recall": (bullish["recall"] + bearish["recall"]) / 2.0,
        "macro_f1": (bullish["f1"] + bearish["f1"]) / 2.0,
        "bullish": bullish,
        "bearish": bearish,
        "confusion_matrix": _confusion_matrix(pairs),
    }


def _print_metrics(summary: dict[str, object]) -> None:
    cfg = summary["config"]
    print("=== VADER baseline ===")
    print(
        f"input={cfg['input']} limit_blocks={cfg['limit_blocks']} "
        f"fallback_for_neutral={cfg['neutral_fallback_label']}"
    )
    print(
        f"blocks={summary['blocks_total']} "
        f"neutral={summary['neutral_count']} "
        f"non_neutral={summary['non_neutral_count']}"
    )

    full = summary["full_metrics"]
    print(
        f"FULL (n={full['total']}): acc={full['accuracy']:.4f} "
        f"macro_f1={full['macro_f1']:.4f}"
    )
    print(f"  bull f1={full['bullish']['f1']:.4f} bear f1={full['bearish']['f1']:.4f}")
    print(f"  confusion: {full['confusion_matrix']}")

    non_neutral = summary["non_neutral_metrics"]
    if non_neutral is not None:
        print(
            f"NON-NEUTRAL (n={non_neutral['total']}): acc={non_neutral['accuracy']:.4f} "
            f"macro_f1={non_neutral['macro_f1']:.4f}"
        )
        print(
            f"  bull f1={non_neutral['bullish']['f1']:.4f} "
            f"bear f1={non_neutral['bearish']['f1']:.4f}"
        )
        print(f"  confusion: {non_neutral['confusion_matrix']}")
    else:
        print("NON-NEUTRAL: no samples fell outside the neutral band")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="VADER lexicon baseline for root-comment sentiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--limit-blocks", type=int, default=None)
    parser.add_argument("--output-jsonl", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--print-samples", type=int, default=5)
    args = parser.parse_args()

    summary = run_vader_baseline(
        input_path=args.input,
        limit_blocks=args.limit_blocks,
    )
    _print_metrics(summary)

    for row in summary["predictions"][: args.print_samples]:
        text = str(row["text"]).replace("\n", " ")[:80]
        print(
            f"- {row['block_id']} | true={row['true_label']} "
            f"| pred={row['pred_label']} | compound={row['compound']:+.3f} | text={text}"
        )

    if args.output_jsonl:
        path = Path(args.output_jsonl)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in summary["predictions"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote JSONL: {path}")

    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: v for k, v in summary.items() if k != "predictions"}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote metrics JSON: {path}")


if __name__ == "__main__":
    main()