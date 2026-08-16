"""
Unit & Integration Tests for Fraud Analysis Agent (Phase D)
"""

import os
import unittest
import tempfile
import pandas as pd

from src.agents.fraud_analysis_agent import FraudAnalysisAgent
from src.agents.contracts import EvidencePackage
from src.database.connection import init_db, db_transaction
from src.config import FEATURES_DATA_DIR


class TestFraudAnalysisAgent(unittest.TestCase):
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()
        init_db(self.db_path)
        self.agent = FraudAnalysisAgent(db_path=self.db_path)

    def tearDown(self):
        try:
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            for ext in ["-wal", "-shm"]:
                f = self.db_path + ext
                if os.path.exists(f):
                    os.remove(f)
        except Exception:
            pass

    def test_fraud_analysis_agent_on_feature_parquet(self):
        train_feat_path = os.path.join(FEATURES_DATA_DIR, "train_provider_features.parquet")
        if not os.path.exists(train_feat_path):
            self.skipTest(f"Features file {train_feat_path} not found.")

        df_train = pd.read_parquet(train_feat_path)
        sample_row = df_train.iloc[[0]]
        prov_id = str(sample_row['Provider'].values[0])

        evidence: EvidencePackage = self.agent.analyze_provider(
            provider_id=prov_id,
            df_feature_row=sample_row,
            actor_username="test_investigator"
        )

        self.assertEqual(evidence.provider_id, prov_id)
        self.assertGreaterEqual(evidence.risk_score, 0)
        self.assertLessEqual(evidence.risk_score, 100)
        self.assertIn(evidence.risk_level, {"LOW", "MEDIUM", "HIGH", "CRITICAL"})
        self.assertIn(evidence.investigation_priority, {"LOW", "NORMAL", "HIGH", "CRITICAL"})
        self.assertGreaterEqual(len(evidence.behavioral_metrics), 4)
        self.assertGreaterEqual(len(evidence.key_findings), 1)

        # Check SQLite persistence
        with db_transaction(self.db_path) as conn:
            runs = conn.execute("SELECT * FROM agent_runs WHERE agent_name='FRAUD_ANALYSIS_AGENT'").fetchall()
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["provider_id"], prov_id)

            logs = conn.execute("SELECT * FROM audit_logs WHERE action='FRAUD_ANALYSIS_AGENT_RUN'").fetchall()
            self.assertEqual(len(logs), 1)

    def test_fraud_analysis_agent_from_form_dict(self):
        form_data = {
            "total_claims": 80,
            "total_claim_amount": 180000.0,
            "inpatient_claim_count": 20,
            "repeat_beneficiary_ratio": 0.40,
            "same_attending_operating_ratio": 0.50,
            "average_claim_vs_peer_average": 2.2,
            "claim_amount_vs_peer_average": 2.8
        }

        evidence = self.agent.analyze_from_dict(
            input_dict=form_data,
            provider_id="PRV_FORM_01",
            actor_username="user1"
        )

        self.assertEqual(evidence.provider_id, "PRV_FORM_01")
        self.assertGreaterEqual(evidence.fraud_probability, 0.0)
        self.assertLessEqual(evidence.fraud_probability, 1.0)
        self.assertIsNotNone(evidence.peer_comparison_summary)
        self.assertIsNotNone(evidence.network_concentration_summary)


if __name__ == "__main__":
    unittest.main()
