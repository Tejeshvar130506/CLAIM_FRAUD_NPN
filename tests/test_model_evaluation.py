"""
Unit & Integration Tests for Model Evaluation Phase 5
"""

import os
import unittest
import pandas as pd


class TestModelEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls.reports_dir = os.path.join(cls.base_dir, "reports")

    def test_model_comparison_csv_exists(self):
        csv_path = os.path.join(self.reports_dir, "model_comparison.csv")
        self.assertTrue(os.path.exists(csv_path), "model_comparison.csv missing.")

        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 2, "model_comparison.csv must contain 2 rows (XGBoost and Random Forest).")
        self.assertIn("Precision", df.columns)
        self.assertIn("Recall", df.columns)
        self.assertIn("F1", df.columns)
        self.assertIn("ROC-AUC", df.columns)
        self.assertIn("TP", df.columns)

    def test_evaluation_plot_files_exist(self):
        plots = [
            "roc_curve.png",
            "precision_recall_curve.png",
            "confusion_matrix.png"
        ]

        for p in plots:
            p_path = os.path.join(self.reports_dir, p)
            self.assertTrue(os.path.exists(p_path), f"Evaluation plot {p} missing from reports/.")

    def test_model_metrics_report_exists(self):
        rep_path = os.path.join(self.reports_dir, "model_metrics.md")
        self.assertTrue(os.path.exists(rep_path), "model_metrics.md missing.")


if __name__ == "__main__":
    unittest.main()
