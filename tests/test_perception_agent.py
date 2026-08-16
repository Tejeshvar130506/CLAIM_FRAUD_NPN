"""
Unit & Integration Tests for Perception Agent (Phase B)
"""

import os
import unittest
import pandas as pd
import tempfile

from src.agents.perception_agent import PerceptionAgent
from src.agents.contracts import PerceptionResult
from src.config import RAW_DATA_DIR, BASE_DIR
from src.database.connection import init_db, db_transaction


class TestPerceptionAgent(unittest.TestCase):
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()
        init_db(self.db_path)
        self.agent = PerceptionAgent(db_path=self.db_path)

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

    def test_perception_agent_on_raw_train_directory(self):
        train_dir = os.path.join(RAW_DATA_DIR, "train")
        if not os.path.exists(train_dir):
            self.skipTest(f"Directory {train_dir} not available.")

        result: PerceptionResult = self.agent.analyze_dataset_directory(
            train_dir, group_name="TRAIN", actor_username="analyst1"
        )

        self.assertEqual(result.status, "COMPLETED")
        self.assertEqual(result.dataset_group, "TRAIN")
        self.assertEqual(len(result.files_profiled), 4)
        self.assertEqual(result.quality_report.key_integrity_status, "VALIDATED")
        self.assertTrue(result.quality_report.provider_key_present)
        self.assertTrue(result.quality_report.beneficiary_key_present)
        self.assertTrue(result.quality_report.claim_key_present)
        self.assertGreaterEqual(result.quality_report.overall_quality_score, 80.0)

        # Verify DB audit & agent_runs persistence
        with db_transaction(self.db_path) as conn:
            agent_runs = conn.execute("SELECT * FROM agent_runs WHERE agent_name='PERCEPTION_AGENT'").fetchall()
            self.assertGreaterEqual(len(agent_runs), 1)
            
            audit_logs = conn.execute("SELECT * FROM audit_logs WHERE action='PERCEPTION_AGENT_RUN'").fetchall()
            self.assertGreaterEqual(len(audit_logs), 1)

    def test_perception_agent_on_in_memory_dataframes(self):
        sample_claims = pd.DataFrame({
            "Provider": ["PRV001", "PRV002", "PRV001"],
            "BeneID": ["B01", "B02", "B01"],
            "ClaimID": ["CLM01", "CLM02", "CLM03"],
            "InscClaimAmtReimbursed": [1000.0, 2500.0, 1500.0],
            "ClaimStartDt": ["2026-01-01", "2026-01-02", "2026-01-03"]
        })
        sample_bene = pd.DataFrame({
            "BeneID": ["B01", "B02"],
            "DOB": ["1950-01-01", "1960-05-15"],
            "Gender": [1, 2]
        })

        dfs = {
            "claims.csv": sample_claims,
            "beneficiary.csv": sample_bene
        }

        result: PerceptionResult = self.agent.analyze_uploaded_dataframes(
            dfs, group_name="CUSTOM_UPLOAD", actor_username="analyst_test"
        )

        self.assertEqual(result.status, "COMPLETED")
        self.assertEqual(result.total_providers_detected, 2)
        self.assertEqual(result.total_beneficiaries_detected, 2)
        self.assertEqual(result.total_claims_detected, 3)
        self.assertEqual(result.quality_report.overall_quality_score, 100.0)


if __name__ == "__main__":
    unittest.main()
