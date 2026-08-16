"""
End-to-End System Integration Tests (Phase H)
---------------------------------------------
Validates complete multi-agent execution, human-in-the-loop triage,
manager escalation adjudication, model registry governance, and RBAC enforcement.
"""

import os
import unittest
import tempfile
import pandas as pd

from src.agents.perception_agent import PerceptionAgent
from src.agents.fraud_analysis_agent import FraudAnalysisAgent
from src.agents.negotiation_agent import NegotiationAgent
from src.agents.arbitrator_agent import Arbitrator
from src.agents.orchestrator import MultiAgentOrchestrator
from src.services.investigation_service import InvestigationService
from src.services.manager_service import ManagerService
from src.services.model_registry_service import ModelRegistryService
from src.services.audit_service import get_audit_logs
from src.auth.security import seed_default_users, verify_user_credentials
from src.auth.rbac import has_permission, Permission
from src.database.connection import init_db, db_transaction
from src.config import FEATURES_DATA_DIR


class TestEndToEndSystemIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()
        init_db(self.db_path)
        seed_default_users(self.db_path)

        self.orchestrator = MultiAgentOrchestrator(db_path=self.db_path)
        self.inv_service = InvestigationService(db_path=self.db_path)
        self.mgr_service = ManagerService(db_path=self.db_path)
        self.model_service = ModelRegistryService(db_path=self.db_path)

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

    def test_complete_multi_agent_to_investigation_lifecycle(self):
        # 1. Simulate single provider interactive assessment
        high_risk_input = {
            "total_claims": 250,
            "total_claim_amount": 920000.0,
            "inpatient_claim_count": 95,
            "repeat_beneficiary_ratio": 0.68,
            "same_attending_operating_ratio": 0.72,
            "average_claim_vs_peer_average": 3.8,
            "claim_amount_vs_peer_average": 4.5
        }

        orch_res = self.orchestrator.run_pipeline_from_dict(
            input_dict=high_risk_input,
            provider_id="PRV_E2E_999",
            actor_username="analyst_user"
        )

        self.assertEqual(orch_res.status, "SUCCESS")
        self.assertTrue(orch_res.arbitrator.is_investigation_candidate)
        self.assertTrue(orch_res.investigation_case_created)
        case_number = orch_res.investigation_case_number

        # 2. Verify case present in investigator queue
        queue = self.inv_service.get_investigation_queue(status="NEW")
        matched = [c for c in queue if c["case_number"] == case_number]
        self.assertEqual(len(matched), 1)
        case_id = matched[0]["id"]

        # 3. Assign case to investigator
        self.inv_service.assign_case(case_id, "investigator", "manager")
        case_assigned = self.inv_service.get_investigation_by_id(case_id)
        self.assertEqual(case_assigned["assigned_to"], "investigator")
        self.assertEqual(case_assigned["status"], "ASSIGNED")

        # 4. Investigator adds clinical note
        self.inv_service.add_case_note(
            case_id,
            "Reviewed billing logs: detected repeat patient concentration without corresponding diagnosis complexity.",
            "investigator"
        )

        # 5. Investigator escalates via Path B due to tertiary care specialization ambiguity
        self.inv_service.escalate_to_management(
            investigation_id=case_id,
            escalation_reason="Provider is sole regional oncology infusion clinic; requires managerial sign-off for clinical chart subpoena.",
            actor_username="investigator"
        )
        case_esc = self.inv_service.get_investigation_by_id(case_id)
        self.assertEqual(case_esc["status"], "ESCALATED")

        # 6. Manager reviews escalated queue
        esc_cases = self.mgr_service.get_escalated_cases()
        self.assertTrue(any(c["id"] == case_id for c in esc_cases))

        # 7. Manager adjudicates case
        self.mgr_service.record_management_decision(
            investigation_id=case_id,
            decision_action="REFER_TO_PAYMENT_INTEGRITY_AUDIT",
            reasoning="Approved for specialized chart audit and reimbursement escrow hold.",
            actor_username="manager"
        )
        case_resolved = self.inv_service.get_investigation_by_id(case_id)
        self.assertEqual(case_resolved["status"], "RESOLVED_VALIDATED")
        self.assertEqual(case_resolved["manager_decision"], "REFER_TO_PAYMENT_INTEGRITY_AUDIT")

        # 8. Audit trail completeness verification
        logs = get_audit_logs(entity_id=str(case_id), db_path=self.db_path)
        self.assertGreaterEqual(len(logs), 3)

    def test_rbac_security_matrix_enforcement(self):
        self.assertFalse(has_permission("USER", Permission.RECORD_INVESTIGATION_FINDING))
        self.assertFalse(has_permission("USER", Permission.RECORD_MANAGEMENT_DECISION))
        self.assertFalse(has_permission("USER", Permission.VIEW_AUDIT_LOGS))

        self.assertTrue(has_permission("INVESTIGATOR", Permission.RECORD_INVESTIGATION_FINDING))
        self.assertTrue(has_permission("INVESTIGATOR", Permission.ESCALATE_TO_MANAGEMENT))
        self.assertFalse(has_permission("INVESTIGATOR", Permission.RECORD_MANAGEMENT_DECISION))

        self.assertTrue(has_permission("MANAGER", Permission.RECORD_MANAGEMENT_DECISION))
        self.assertTrue(has_permission("MANAGER", Permission.ASSIGN_INVESTIGATIONS))

        self.assertTrue(has_permission("ADMIN", Permission.MANAGE_USERS))
        self.assertTrue(has_permission("ADMIN", Permission.MANAGE_MODEL_REGISTRY))

    def test_model_registry_lifecycle(self):
        mid = self.model_service.register_model(
            version_tag="xgb-v2.1.0-prod",
            model_type="XGBoost_Classifier",
            file_path="models/final_fraud_model.pkl",
            roc_auc=0.9692,
            pr_auc=0.8407,
            f1_score=0.7850,
            features_count=52,
            is_active=False,
            actor_username="admin"
        )
        self.assertIsInstance(mid, int)

        # Promote to active
        success = self.model_service.set_active_version(mid, actor_username="admin")
        self.assertTrue(success)

        active = self.model_service.get_active_model("XGBoost_Classifier")
        self.assertIsNotNone(active)
        self.assertEqual(active["version_tag"], "xgb-v2.1.0-prod")


if __name__ == "__main__":
    unittest.main()
