"""
Unit & Integration Tests for Provider Fraud Model Pipeline (Phase 4)
"""

import os
import unittest
import joblib
import pandas as pd
import numpy as np


class TestProviderFraudModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls.models_dir = os.path.join(cls.base_dir, "models")
        cls.reports_dir = os.path.join(cls.base_dir, "reports")
        cls.processed_dir = os.path.join(cls.base_dir, "data", "processed")
        cls.features_dir = os.path.join(cls.base_dir, "data", "features")

        cls.final_model_path = os.path.join(cls.models_dir, "final_fraud_model.pkl")
        cls.xgb_model_path = os.path.join(cls.models_dir, "xgboost_fraud_model.pkl")
        cls.rf_model_path = os.path.join(cls.models_dir, "random_forest_fraud_model.pkl")

    def test_saved_models_exist_and_load(self):
        self.assertTrue(os.path.exists(self.final_model_path), "final_fraud_model.pkl missing.")
        self.assertTrue(os.path.exists(self.xgb_model_path), "xgboost_fraud_model.pkl missing.")
        self.assertTrue(os.path.exists(self.rf_model_path), "random_forest_fraud_model.pkl missing.")

        model = joblib.load(self.final_model_path)
        self.assertIsNotNone(model, "Failed to load final_fraud_model.pkl.")

    def test_model_prediction_capability(self):
        model = joblib.load(self.final_model_path)
        test_feat_path = os.path.join(self.features_dir, "test_provider_features.parquet")
        df_test = pd.read_parquet(test_feat_path)

        feature_cols = [c for c in df_test.columns if c != 'Provider']
        X_sample = df_test[feature_cols].iloc[:10]

        probs = model.predict_proba(X_sample)[:, 1]
        self.assertEqual(len(probs), 10)
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)), "Probabilities must be in range [0.0, 1.0]")

    def test_test_predictions_dataset(self):
        parquet_out = os.path.join(self.processed_dir, "test_provider_predictions.parquet")
        csv_out = os.path.join(self.reports_dir, "test_provider_predictions.csv")

        self.assertTrue(os.path.exists(parquet_out), "test_provider_predictions.parquet missing.")
        self.assertTrue(os.path.exists(csv_out), "test_provider_predictions.csv missing.")

        df_preds = pd.read_parquet(parquet_out)
        self.assertEqual(len(df_preds), 1353, "TEST predictions must contain exactly 1,353 rows.")
        self.assertIn('Provider', df_preds.columns)
        self.assertIn('fraud_prediction', df_preds.columns)
        self.assertIn('fraud_probability', df_preds.columns)

    def test_model_evaluation_report_exists(self):
        report_path = os.path.join(self.reports_dir, "model_evaluation.md")
        self.assertTrue(os.path.exists(report_path), "model_evaluation.md missing.")


if __name__ == "__main__":
    unittest.main()
