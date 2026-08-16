"""
Unit & Integration Tests for Human Investigation & Manager Workflow (Phase F)
"""

import os
import unittest
import tempfile

from src.services.investigation_service import InvestigationService
from src.services.manager_service import ManagerService
from src.database.connection import init_db, db_transaction
from src.auth.security import seed_default_users


class TestInvestigationAndManagerWorkflow(unittest.TestCase):
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()
        init_db(self.db_path)
        seed_default_users(self.db_path)

        self.inv_service = InvestigationService(db_path=self.db_path)
        self.mgr_service = ManagerService(db_path=self.db_path)

        # Seed sample provider
        with db_transaction(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO providers (provider_id, primary_state, total_claims, total_claim_amount, risk_score, risk_level, investigation_priority)
                VALUES ('PRV_SIU_01', '39', 150, 450000.0, 88, 'CRITICAL', 'CRITICAL')
                """
            )
            cursor = conn.execute(
                """
                INSERT INTO investigations (provider_id, case_number, priority, status, ai_risk_score, ai_risk_level, ai_fraud_probability)
                VALUES ('PRV_SIU_01', 'INV-PRV_SIU_01-1001', 'CRITICAL', 'NEW', 88, 'CRITICAL', 0.885)
                """
            )
            self.case_id = cursor.lastrowid

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

    def test_case_queue_and_notes(self):
        queue = self.inv_service.get_investigation_queue()
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["case_number"], "INV-PRV_SIU_01-1001")

        # Assign case
        self.inv_service.assign_case(self.case_id, "investigator", "manager")
        case = self.inv_service.get_investigation_by_id(self.case_id)
        self.assertEqual(case["assigned_to"], "investigator")
        self.assertEqual(case["status"], "ASSIGNED")

        # Add clinical note
        self.inv_service.add_case_note(self.case_id, "Reviewed Inpatient stay records; requested operative notes.", "investigator")
        events = self.inv_service.get_case_events(self.case_id)
        self.assertEqual(len(events), 2)  # Assigned + Note

    def test_path_a_sufficient_evidence_decision(self):
        success = self.inv_service.record_investigator_finding(
            investigation_id=self.case_id,
            finding_type="ELEVATED_RISK_VALIDATED",
            reasoning="Claims audits confirmed unbundling of surgical codes and repeat patient loops without documented necessity.",
            follow_up_action="Refer to special investigations audit for $120k recovery.",
            actor_username="investigator"
        )
        self.assertTrue(success)

        case = self.inv_service.get_investigation_by_id(self.case_id)
        self.assertEqual(case["status"], "RESOLVED_VALIDATED")
        self.assertIn("ELEVATED_RISK_VALIDATED", case["final_outcome"])

    def test_path_b_insufficient_evidence_escalation_and_manager_decision(self):
        # Step 1: Investigator escalates via Path B
        escalate_success = self.inv_service.escalate_to_management(
            investigation_id=self.case_id,
            escalation_reason="Statistical deviations exist, but provider specialization in tertiary surgical care creates substantial clinical ambiguity beyond standard claims analysis.",
            actor_username="investigator"
        )
        self.assertTrue(escalate_success)

        case_escalated = self.inv_service.get_investigation_by_id(self.case_id)
        self.assertEqual(case_escalated["status"], "ESCALATED")
        self.assertIsNotNone(case_escalated["escalation_reason"])

        # Step 2: Manager reviews escalated cases queue
        escalated_queue = self.mgr_service.get_escalated_cases()
        self.assertEqual(len(escalated_queue), 1)
        self.assertEqual(escalated_queue[0]["id"], self.case_id)

        # Step 3: Manager records decision
        mgr_decision_success = self.mgr_service.record_management_decision(
            investigation_id=self.case_id,
            decision_action="REFER_TO_PAYMENT_INTEGRITY_AUDIT",
            reasoning="Concur with investigator assessment. Refer case for on-site chart sampling and targeted audit.",
            actor_username="manager"
        )
        self.assertTrue(mgr_decision_success)

        case_final = self.inv_service.get_investigation_by_id(self.case_id)
        self.assertEqual(case_final["status"], "RESOLVED_VALIDATED")
        self.assertEqual(case_final["manager_decision"], "REFER_TO_PAYMENT_INTEGRITY_AUDIT")

        # Step 4: Verify Executive KPIs
        kpis = self.mgr_service.get_management_kpis()
        self.assertEqual(kpis["total_investigations"], 1)
        self.assertEqual(kpis["validated_fraud_risk_cases"], 1)
        self.assertEqual(kpis["total_risk_exposure_dollars"], 450000.0)


if __name__ == "__main__":
    unittest.main()
