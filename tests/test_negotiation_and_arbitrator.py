"""
Unit & Integration Tests for Negotiation Agent, Arbitrator, and Orchestrator (Phase E)
"""

import os
import unittest
import tempfile
import pandas as pd

from src.agents.negotiation_agent import NegotiationAgent
from src.agents.arbitrator_agent import Arbitrator
from src.agents.orchestrator import MultiAgentOrchestrator
from src.agents.contracts import (
    EvidencePackage, NegotiationResult, ArbitratorResult, AgentOrchestrationResult
)
from src.database.connection import init_db, db_transaction
from src.config import FEATURES_DATA_DIR


class TestNegotiationAndArbitrator(unittest.TestCase):
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()
        init_db(self.db_path)

        self.orchestrator = MultiAgentOrchestrator(db_path=self.db_path)
        self.negotiation_agent = NegotiationAgent(db_path=self.db_path)
        self.arbitrator = Arbitrator(db_path=self.db_path)

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

    def test_negotiation_agent_workflow(self):
        train_feat_path = os.path.join(FEATURES_DATA_DIR, "train_provider_features.parquet")
        if not os.path.exists(train_feat_path):
            self.skipTest("Feature file not found.")

        df_train = pd.read_parquet(train_feat_path)
        sample_row = df_train.iloc[[0]]
        prov_id = str(sample_row['Provider'].values[0])

        evidence = self.orchestrator.fraud_analysis_agent.analyze_provider(
            provider_id=prov_id,
            df_feature_row=sample_row
        )

        negotiation: NegotiationResult = self.negotiation_agent.negotiate(evidence)

        self.assertEqual(negotiation.provider_id, prov_id)
        self.assertGreaterEqual(len(negotiation.arguments_supporting_investigation), 1)
        self.assertGreaterEqual(len(negotiation.counter_challenges_and_mitigations), 1)
        self.assertIn(negotiation.proposed_action, {
            "LOW_CONCERN", "ROUTINE_MONITORING", "AUDIT_REVIEW", "HIGH_PRIORITY_INVESTIGATION", "INSUFFICIENT_EVIDENCE"
        })
        self.assertIn(negotiation.confidence_rating, {"LOW", "MODERATE", "HIGH"})

    def test_arbitrator_workflow(self):
        train_feat_path = os.path.join(FEATURES_DATA_DIR, "train_provider_features.parquet")
        if not os.path.exists(train_feat_path):
            self.skipTest("Feature file not found.")

        df_train = pd.read_parquet(train_feat_path)
        sample_row = df_train.iloc[[0]]
        prov_id = str(sample_row['Provider'].values[0])

        evidence = self.orchestrator.fraud_analysis_agent.analyze_provider(prov_id, sample_row)
        negotiation = self.negotiation_agent.negotiate(evidence)
        arbitration: ArbitratorResult = self.arbitrator.arbitrate(evidence, negotiation)

        self.assertEqual(arbitration.provider_id, prov_id)
        self.assertIn(arbitration.final_risk_level, {"LOW", "MEDIUM", "HIGH", "CRITICAL"})
        self.assertGreaterEqual(arbitration.final_risk_score, 0)
        self.assertLessEqual(arbitration.final_risk_score, 100)
        self.assertIn(arbitration.recommended_investigation_priority, {"LOW", "NORMAL", "HIGH", "CRITICAL"})
        self.assertIsInstance(arbitration.is_investigation_candidate, bool)

    def test_full_orchestrator_pipeline_and_case_creation(self):
        form_high_risk = {
            "total_claims": 200,
            "total_claim_amount": 750000.0,
            "inpatient_claim_count": 80,
            "repeat_beneficiary_ratio": 0.65,
            "same_attending_operating_ratio": 0.70,
            "average_claim_vs_peer_average": 3.4,
            "claim_amount_vs_peer_average": 4.2
        }

        result: AgentOrchestrationResult = self.orchestrator.run_pipeline_from_dict(
            input_dict=form_high_risk,
            provider_id="PRV_HIGH_RISK_TEST",
            actor_username="test_analyst"
        )

        self.assertEqual(result.status, "SUCCESS")
        self.assertIsNotNone(result.fraud_analysis)
        self.assertIsNotNone(result.negotiation)
        self.assertIsNotNone(result.arbitrator)

        # Check DB auto-provisioning
        with db_transaction(self.db_path) as conn:
            inv = conn.execute(
                "SELECT * FROM investigations WHERE provider_id='PRV_HIGH_RISK_TEST'"
            ).fetchone()
            
            if result.arbitrator.is_investigation_candidate:
                self.assertIsNotNone(inv, "Investigation candidate record must be auto-created in SQLite.")
                self.assertEqual(inv["status"], "NEW")
                self.assertTrue(result.investigation_case_created)
                
                # Check events table
                events = conn.execute(
                    "SELECT * FROM investigation_events WHERE investigation_id = ?",
                    (inv["id"],)
                ).fetchall()
                self.assertGreaterEqual(len(events), 1)


if __name__ == "__main__":
    unittest.main()
