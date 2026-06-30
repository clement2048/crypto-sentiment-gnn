"""统一的 baseline 指标计算工具。

This module does NOT read block.p0 / block.p1 / block.label directly.
所有输入由调用方提供：`rows` 形如 `[{"true_label": int, "pred_label_int": int}, ...]`
这样无论 VADER / Sentence-BERT / Direct LLM 还是后续的 debate+judge，
只要把每条样本的真实标签与预测标签打成 row，都可以用同一份 compute_metrics
输出同一份字段格式（total / accuracy / confusion_matrix / macro_f1 / ...），
方便跨 baseline 在同一张表上横向对比。

设计上是 **baseline-agnostic** 的：不依赖任何 baseline 专属类型，也不依赖
torch / sklearn / numpy。任何能产出 (truth, pred) 的流程（baseline notebook、
debate+judge pipeline、reflection 循环、论文复现脚本）都可以直接 import。
"""

from __future__ import annotations

LABEL_BULLISH = 1
LABEL_BEARISH = -1
VALID_LABELS = (LABEL_BULLISH, LABEL_BEARISH)


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _class_prf(pairs: list[tuple[int, int]], target: int) -> dict[str, float | int]:
    """计算一个类的 precision / recall / f1 / support。"""
    tp = sum(1 for t, p in pairs if t == target and p == target)
    fp = sum(1 for t, p in pairs if t != target and p == target)
    fn = sum(1 for t, p in pairs if t == target and p != target)
    support = sum(1 for t, _ in pairs if t == target)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": int(support),
    }


def _confusion_matrix(pairs: list[tuple[int, int]]) -> dict[str, dict[str, int]]:
    matrix = {
        "bullish": {"bullish": 0, "bearish": 0},
        "bearish": {"bullish": 0, "bearish": 0},
    }
    for t, p in pairs:
        truth_name = "bullish" if int(t) == LABEL_BULLISH else "bearish"
        pred_name = "bullish" if int(p) == LABEL_BULLISH else "bearish"
        matrix[truth_name][pred_name] += 1
    return matrix


def compute_metrics(rows: list[dict[str, int | None]]) -> dict[str, object]:
    """统一指标计算。

    输入：
        rows: 每个元素形如 ``{"true_label": int, "pred_label_int": int}``。
              真实标签与预测标签仅接受 1 / -1，其它值会被忽略，避免非法 label
              让指标崩溃。

    返回：固定字段的字典；空 rows 也能安全返回（所有比率 0.0、total=0）。
    """
    pairs: list[tuple[int, int]] = []
    for row in rows:
        t = row.get("true_label")
        p = row.get("pred_label_int")
        if t not in VALID_LABELS or p not in VALID_LABELS:
            continue
        pairs.append((int(t), int(p)))

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
    correct = sum(1 for t, p in pairs if t == p)
    bull = _class_prf(pairs, LABEL_BULLISH)
    bear = _class_prf(pairs, LABEL_BEARISH)
    return {
        "total": total,
        "accuracy": _safe_div(correct, total),
        "macro_precision": (bull["precision"] + bear["precision"]) / 2.0,
        "macro_recall": (bull["recall"] + bear["recall"]) / 2.0,
        "macro_f1": (bull["f1"] + bear["f1"]) / 2.0,
        "bullish": bull,
        "bearish": bear,
        "confusion_matrix": _confusion_matrix(pairs),
    }
