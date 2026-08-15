"""
Unit & Integration Tests for Preprocessing & Feature Engineering Pipeline
"""

import os
import unittest
import pandas as pd


class TestHealthcarePreprocessingAndFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls.processed_dir = os.path.join(cls.base_dir, "data", "processed")
        cls.features_dir = os.path.join(cls.base_dir, "data", "features")
        cls.reports_dir = os.path.join(cls.base_dir, "reports")

    def test_processed_claims_files_exist(self):
        train_path = os.path.join(self.processed_dir, "train_claims_integrated.parquet")
        test_path = os.path.join(self.processed_dir, "test_claims_integrated.parquet")

        self.assertTrue(os.path.exists(train_path), "train_claims_integrated.parquet missing.")
        self.assertTrue(os.path.exists(test_path), "test_claims_integrated.parquet missing.")

        df_tr = pd.read_parquet(train_path)
        df_te = pd.read_parquet(test_path)

        self.assertEqual(len(df_tr), 558211, "Train claims count must equal 558,211.")
        self.assertEqual(len(df_te), 135392, "Test claims count must equal 135,392.")

    def test_provider_features_files_exist(self):
        train_feat_path = os.path.join(self.features_dir, "train_provider_features.parquet")
        test_feat_path = os.path.join(self.features_dir, "test_provider_features.parquet")

        self.assertTrue(os.path.exists(train_feat_path), "train_provider_features.parquet missing.")
        self.assertTrue(os.path.exists(test_feat_path), "test_provider_features.parquet missing.")

        df_tr_f = pd.read_parquet(train_feat_path)
        df_te_f = pd.read_parquet(test_feat_path)

        self.assertEqual(len(df_tr_f), 5410, "Train provider feature count must equal 5,410.")
        self.assertEqual(len(df_te_f), 1353, "Test provider feature count must equal 1,353.")

    def test_target_label_isolation(self):
        train_feat_path = os.path.join(self.features_dir, "train_provider_features.parquet")
        test_feat_path = os.path.join(self.features_dir, "test_provider_features.parquet")

        df_tr_f = pd.read_parquet(train_feat_path)
        df_te_f = pd.read_parquet(test_feat_path)

        self.assertIn("PotentialFraud", df_tr_f.columns, "TRAIN provider features must contain PotentialFraud.")
        self.assertNotIn("PotentialFraud", df_te_f.columns, "TEST provider features must NOT contain PotentialFraud.")
        self.assertEqual(df_tr_f["PotentialFraud"].isnull().sum(), 0, "No missing target values allowed in TRAIN.")
        self.assertEqual(df_tr_f["PotentialFraud"].sum(), 506, "Positive fraud count in TRAIN must equal 506.")

    def test_preprocessing_report_exists(self):
        rep_path = os.path.join(self.reports_dir, "preprocessing_report.md")
        self.assertTrue(os.path.exists(rep_path), "preprocessing_report.md must exist.")


if __name__ == "__main__":
    unittest.main()
