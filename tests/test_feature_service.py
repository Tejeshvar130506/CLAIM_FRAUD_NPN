"""
Unit & Integration Tests for Feature Engineering Service (Phase C)
"""

import os
import unittest
import joblib
import pandas as pd
import numpy as np

from src.services.feature_service import FeatureEngineeringService, MODEL_FEATURE_COLUMNS
from src.config import MODELS_DIR, FEATURES_DATA_DIR


class TestFeatureEngineeringService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = FeatureEngineeringService()
        cls.model_path = os.path.join(MODELS_DIR, "final_fraud_model.pkl")
        if os.path.exists(cls.model_path):
            cls.model = joblib.load(cls.model_path)
        else:
            cls.model = None

    def test_canonical_feature_columns_count(self):
        cols = self.service.get_canonical_feature_columns()
        self.assertEqual(len(cols), 52, "Model expects exactly 52 feature columns.")
        self.assertEqual(cols, MODEL_FEATURE_COLUMNS)

    def test_single_provider_form_vector_generation(self):
        input_data = {
            "total_claims": 120,
            "total_claim_amount": 350000.0,
            "inpatient_claim_count": 45,
            "repeat_beneficiary_ratio": 0.55,
            "same_attending_operating_ratio": 0.60,
            "average_claim_vs_peer_average": 2.4,
            "claim_amount_vs_peer_average": 3.1
        }
        df_feat = self.service.build_feature_vector_from_dict(input_data, provider_id="PRV_TEST_001")
        
        self.assertEqual(len(df_feat), 1)
        self.assertIn("Provider", df_feat.columns)
        
        # Check all 52 features are present
        for col in MODEL_FEATURE_COLUMNS:
            self.assertIn(col, df_feat.columns)

        # Test model inference on extracted vector if model is present
        if self.model is not None:
            X = df_feat[MODEL_FEATURE_COLUMNS]
            prob = float(self.model.predict_proba(X)[0, 1])
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

    def test_batch_claims_extraction(self):
        # Create minimal synthetic integrated claims
        claims_data = {
            "Provider": ["PRV101", "PRV101", "PRV102"],
            "ClaimID": ["C1", "C2", "C3"],
            "BeneID": ["B1", "B1", "B2"],
            "InscClaimAmtReimbursed": [5000.0, 3000.0, 1200.0],
            "DeductibleAmtPaid": [500.0, 300.0, 100.0],
            "IsInpatient": [1, 1, 0],
            "InpatientStayDuration": [4.0, 3.0, 0.0],
            "ClaimDuration": [5.0, 4.0, 1.0],
            "AttendingPhysician": ["PHY1", "PHY1", "PHY2"],
            "OperatingPhysician": ["PHY1", "PHY1", None],
            "OtherPhysician": [None, None, None],
            "ClmAdmitDiagnosisCode": ["D1", "D2", "D3"],
            "DiagnosisGroupCode": ["G1", "G1", None],
            "ClmDiagnosisCode_1": ["CD1", "CD2", "CD3"],
            "ClmProcedureCode_1": ["CP1", None, None],
            "PatientAgeAtClaim": [68.0, 68.0, 72.0],
            "IsDeceased": [0, 0, 0],
            "ChronicCond_Count": [2, 2, 4],
            "RenalDiseaseIndicator": [0, 0, 1],
            "Gender": [1, 1, 2],
            "State": [39, 39, 39],
            "County": [10, 10, 20],
            "ClaimStartDt": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"])
        }
        df_synthetic = pd.DataFrame(claims_data)
        feat_df = self.service.extract_from_integrated_claims(df_synthetic)
        
        self.assertEqual(len(feat_df), 2)
        self.assertIn("PRV101", feat_df["Provider"].values)
        self.assertIn("PRV102", feat_df["Provider"].values)

        for col in MODEL_FEATURE_COLUMNS:
            self.assertIn(col, feat_df.columns)


if __name__ == "__main__":
    unittest.main()
