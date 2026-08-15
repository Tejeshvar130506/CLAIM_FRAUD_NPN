"""
Unit & Integration Tests for EDA and Provider Feature Engineering Phase 3
"""

import os
import unittest
import pandas as pd


class TestEDAAndProviderFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls.features_dir = os.path.join(cls.base_dir, "data", "features")
        cls.reports_dir = os.path.join(cls.base_dir, "reports")
        cls.figures_dir = os.path.join(cls.reports_dir, "figures")

        cls.train_feat_path = os.path.join(cls.features_dir, "train_provider_features.parquet")
        cls.test_feat_path = os.path.join(cls.features_dir, "test_provider_features.parquet")

        cls.df_tr = pd.read_parquet(cls.train_feat_path)
        cls.df_te = pd.read_parquet(cls.test_feat_path)

    def test_required_feature_columns_exist(self):
        required_cols = [
            'total_claims', 'total_claim_amount', 'average_claim_amount', 'maximum_claim_amount',
            'claim_frequency', 'unique_beneficiaries', 'repeat_beneficiary_count', 'repeat_beneficiary_ratio',
            'inpatient_claim_count', 'outpatient_claim_count', 'inpatient_ratio', 'outpatient_ratio',
            'claims_per_month', 'claims_per_week', 'average_claim_vs_peer_average', 'claim_amount_vs_peer_average',
            'average_length_of_stay', 'unique_diagnosis_count', 'unique_procedure_count',
            'top_bene_claim_share', 'beneficiary_hhi_concentration', 'physician_to_claim_ratio'
        ]

        for col in required_cols:
            self.assertIn(col, self.df_tr.columns, f"Required feature '{col}' missing from train feature matrix.")
            self.assertIn(col, self.df_te.columns, f"Required feature '{col}' missing from test feature matrix.")

    def test_zero_nulls_in_engineered_features(self):
        null_counts = self.df_tr.isnull().sum()
        self.assertEqual(null_counts.max(), 0, f"No null values allowed in features. Max nulls: {null_counts.max()}")

    def test_eda_figures_generated(self):
        plots = [
            "01_target_and_claim_volume.png",
            "02_financial_distributions.png",
            "03_inpatient_vs_outpatient.png",
            "04_peer_deviation_analysis.png",
            "05_coordinated_fraud_patterns.png"
        ]

        for p in plots:
            p_path = os.path.join(self.figures_dir, p)
            self.assertTrue(os.path.exists(p_path), f"Plot {p} was not generated.")

    def test_eda_reports_exist(self):
        reports = [
            "feature_definitions.md",
            "feature_quality_report.md",
            "fraud_pattern_report.md"
        ]

        for r in reports:
            r_path = os.path.join(self.reports_dir, r)
            self.assertTrue(os.path.exists(r_path), f"Report {r} was not generated.")


if __name__ == "__main__":
    unittest.main()
