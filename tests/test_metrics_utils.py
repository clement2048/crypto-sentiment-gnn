from __future__ import annotations

import unittest
from pathlib import Path
import sys

# 让 unittest 在仓库根直接 `python -m unittest` 跑测试时也能 import baseline
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline._metrics_utils import compute_metrics  # noqa: E402


class ComputeMetricsTest(unittest.TestCase):
    def test_balanced_sample_returns_half_accuracy(self) -> None:
        # 2 bull + 2 bear，对/错各半 → accuracy=0.5、macro_f1=0.5
        rows = [
            {"true_label": 1, "pred_label_int": 1},
            {"true_label": 1, "pred_label_int": -1},
            {"true_label": -1, "pred_label_int": 1},
            {"true_label": -1, "pred_label_int": -1},
        ]

        metrics = compute_metrics(rows)

        self.assertEqual(metrics["total"], 4)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["macro_f1"], 0.5)
        self.assertEqual(
            metrics["confusion_matrix"],
            {
                "bullish": {"bullish": 1, "bearish": 1},
                "bearish": {"bullish": 1, "bearish": 1},
            },
        )

    def test_empty_rows_returns_zero_metrics_without_throwing(self) -> None:
        metrics = compute_metrics([])

        self.assertEqual(metrics["total"], 0)
        self.assertEqual(metrics["accuracy"], 0.0)
        self.assertEqual(metrics["macro_precision"], 0.0)
        self.assertEqual(metrics["macro_recall"], 0.0)
        self.assertEqual(metrics["macro_f1"], 0.0)
        # 各类也要在空集时返回 0 / 0 support
        self.assertEqual(metrics["bullish"]["support"], 0)
        self.assertEqual(metrics["bearish"]["support"], 0)
        self.assertEqual(
            metrics["confusion_matrix"],
            {
                "bullish": {"bullish": 0, "bearish": 0},
                "bearish": {"bullish": 0, "bearish": 0},
            },
        )

    def test_all_correct_predictions_yield_full_accuracy(self) -> None:
        rows = [
            {"true_label": 1, "pred_label_int": 1},
            {"true_label": 1, "pred_label_int": 1},
            {"true_label": -1, "pred_label_int": -1},
        ]

        metrics = compute_metrics(rows)

        self.assertEqual(metrics["total"], 3)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        # 对角线全满，无反类
        self.assertEqual(metrics["confusion_matrix"]["bullish"]["bullish"], 2)
        self.assertEqual(metrics["confusion_matrix"]["bullish"]["bearish"], 0)
        self.assertEqual(metrics["confusion_matrix"]["bearish"]["bearish"], 1)
        self.assertEqual(metrics["confusion_matrix"]["bearish"]["bullish"], 0)

    def test_invalid_labels_are_ignored_without_throwing(self) -> None:
        # 1 / -1 之外的真值/预测应该被当噪点忽略
        rows = [
            {"true_label": 1, "pred_label_int": 1},   # valid bull 猜对
            {"true_label": 0, "pred_label_int": 1},   # 非法真值（忽略）
            {"true_label": 1, "pred_label_int": 0},   # 非法预测（忽略）
            {"true_label": -1, "pred_label_int": -1}, # valid bear 猜对
            {"true_label": -1, "pred_label_int": 1},  # valid bear 误判为 bull
        ]

        metrics = compute_metrics(rows)

        # 剩 3 条合法数据：bull 猜对 1，bear 猜对 1，bear 猜错 1 → 2/3
        self.assertEqual(metrics["total"], 3)
        self.assertAlmostEqual(metrics["accuracy"], 2.0 / 3.0)
        # bear 被预测为 bull 这条要进混淆矩阵
        self.assertEqual(metrics["confusion_matrix"]["bearish"]["bullish"], 1)
        self.assertEqual(metrics["confusion_matrix"]["bullish"]["bullish"], 1)
        self.assertEqual(metrics["confusion_matrix"]["bearish"]["bearish"], 1)


if __name__ == "__main__":
    unittest.main()
