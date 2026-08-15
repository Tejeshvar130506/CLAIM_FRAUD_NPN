"""
Unit & Integration Tests for Provider Risk Scoring & Explainability Engine (Phase 6)
"""

import os
import unittest
from src.risk_scoring import get_provider_risk, calculate_risk_level


class TestProviderRiskScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls.models_dir = os.path.join(cls.base_dir, "models")
        cls.reports_dir = os.path.join(cls.base_dir, "reports")

    def test_calculate_risk_level_thresholds(self):
        self.assertEqual(calculate_risk_level(15), "LOW")
        self.assertEqual(calculate_risk_level(30), "LOW")
        self.assertEqual(calculate_risk_level(45), "MEDIUM")
        self.assertEqual(calculate_risk_level(60), "MEDIUM")
        self.assertEqual(calculate_risk_level(75), "HIGH")
        self.assertEqual(calculate_risk_level(85), "HIGH")
        self.assertEqual(calculate_risk_level(90), "CRITICAL")
        self.assertEqual(calculate_risk_level(100), "CRITICAL")

    def test_get_provider_risk_high_risk(self):
        result = get_provider_risk('PRV51069', base_dir=self.base_dir)
        
        self.assertEqual(result['provider_id'], 'PRV51069')
        self.assertGreaterEqual(result['fraud_probability'], 0.80)
        self.assertGreaterEqual(result['risk_score'], 80)
        self.assertIn(result['risk_level'], ['HIGH', 'CRITICAL'])
        self.assertIsInstance(result['top_potential_contributing_factors'], list)
        self.assertGreater(len(result['top_potential_contributing_factors']), 0)
        
        metrics = result['important_behavioral_metrics']
        self.assertIn('total_claims', metrics)
        self.assertIn('total_claim_amount', metrics)
        self.assertIn('average_claim_amount', metrics)
        self.assertIn('inpatient_ratio', metrics)
        self.assertIn('repeat_beneficiary_ratio', metrics)

    def test_get_provider_risk_low_risk(self):
        result = get_provider_risk('PRV51002', base_dir=self.base_dir)
        
        self.assertEqual(result['provider_id'], 'PRV51002')
        self.assertLessEqual(result['fraud_probability'], 0.30)
        self.assertLessEqual(result['risk_score'], 30)
        self.assertEqual(result['risk_level'], 'LOW')
        self.assertIsInstance(result['top_potential_contributing_factors'], list)

    def test_ebm_model_artifact_exists(self):
        ebm_path = os.path.join(self.models_dir, "ebm_model.pkl")
        self.assertTrue(os.path.exists(ebm_path), "ebm_model.pkl missing from models/.")

    def test_reports_exist(self):
        meth_path = os.path.join(self.reports_dir, "risk_scoring_methodology.md")
        exp_path = os.path.join(self.reports_dir, "explainability_report.md")

        self.assertTrue(os.path.exists(meth_path), "risk_scoring_methodology.md missing.")
        self.assertTrue(os.path.exists(exp_path), "explainability_report.md missing.")


if __name__ == "__main__":
    unittest.main()
