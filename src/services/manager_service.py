"""
Executive Fraud Operations & Manager Service
--------------------------------------------
Provides SIU Management leadership with:
- Escalated case review queue
- Comprehensive evidence, challenge, and investigator finding review
- Policy-configurable Management Decision execution
- Executive risk metrics and team workload analytics
"""

import logging
from typing import List, Dict, Any, Optional

from src.config import DATABASE_PATH
from src.database.connection import db_transaction
from src.services.audit_service import log_audit_event

logger = logging.getLogger(__name__)


class ManagerService:
    """
    Service for management case review, strategic decisions, and operational KPIs.
    """

    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path

    def get_escalated_cases(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Retrieves all cases currently flagged with status='ESCALATED'."""
        query = """
            SELECT 
                i.id, i.provider_id, i.case_number, i.priority, i.status,
                i.assigned_to, i.ai_risk_score, i.ai_risk_level, i.ai_fraud_probability,
                i.escalation_reason, i.manager_decision, i.manager_reasoning, i.final_outcome,
                i.created_at, i.updated_at,
                p.primary_state, p.total_claims, p.total_claim_amount, p.average_claim_amount,
                p.inpatient_ratio, p.repeat_beneficiary_ratio, p.average_claim_vs_peer_average
            FROM investigations i
            LEFT JOIN providers p ON i.provider_id = p.provider_id
            WHERE i.status = 'ESCALATED'
            ORDER BY i.ai_risk_score DESC, i.updated_at DESC
            LIMIT ? OFFSET ?
        """
        with db_transaction(self.db_path) as conn:
            cursor = conn.execute(query, (limit, offset))
            return [dict(r) for r in cursor.fetchall()]

    def record_management_decision(
        self,
        investigation_id: int,
        decision_action: str,
        reasoning: str,
        actor_username: str
    ) -> bool:
        """
        Records an executive Management Decision for an escalated case.
        """
        valid_actions = {
            "ACCEPT_INVESTIGATOR_ASSESSMENT": "RESOLVED_VALIDATED",
            "REQUEST_ADDITIONAL_CLINICAL_RECORDS": "IN_REVIEW",
            "REFER_TO_PAYMENT_INTEGRITY_AUDIT": "RESOLVED_VALIDATED",
            "REFER_TO_LAW_ENFORCEMENT_SIU": "RESOLVED_VALIDATED",
            "CLOSE_NO_FURTHER_ACTION": "RESOLVED_CLEARED"
        }

        new_status = valid_actions.get(decision_action, "RESOLVED_VALIDATED")

        if not reasoning or len(reasoning.strip()) < 10:
            raise ValueError("Management reasoning of at least 10 characters is required.")

        with db_transaction(self.db_path) as conn:
            conn.execute(
                """
                UPDATE investigations SET
                    status = ?,
                    manager_decision = ?,
                    manager_reasoning = ?,
                    final_outcome = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    new_status,
                    decision_action,
                    reasoning.strip(),
                    f"Management Decision: {decision_action}",
                    investigation_id
                )
            )

            conn.execute(
                """
                INSERT INTO investigation_events (
                    investigation_id, event_type, actor_username, actor_role,
                    decision, rationale, notes
                ) VALUES (?, 'MANAGEMENT_DECISION', ?, 'MANAGER', ?, ?, ?)
                """,
                (
                    investigation_id,
                    actor_username,
                    decision_action,
                    reasoning.strip(),
                    f"Management recorded outcome: {decision_action}."
                )
            )

        log_audit_event(
            username=actor_username,
            role="MANAGER",
            action="RECORD_MANAGEMENT_DECISION",
            entity_type="INVESTIGATION",
            entity_id=str(investigation_id),
            status="SUCCESS",
            details={
                "decision": decision_action,
                "new_status": new_status,
                "reasoning_preview": reasoning[:100]
            },
            db_path=self.db_path
        )
        return True

    def get_management_kpis(self) -> Dict[str, Any]:
        """Computes executive operational KPIs across cases, exposure, and investigator workloads."""
        with db_transaction(self.db_path) as conn:
            # Case counts
            total_cases = conn.execute("SELECT COUNT(*) FROM investigations").fetchone()[0]
            escalated_cnt = conn.execute("SELECT COUNT(*) FROM investigations WHERE status = 'ESCALATED'").fetchone()[0]
            in_review_cnt = conn.execute("SELECT COUNT(*) FROM investigations WHERE status IN ('NEW', 'ASSIGNED', 'IN_REVIEW')").fetchone()[0]
            resolved_validated = conn.execute("SELECT COUNT(*) FROM investigations WHERE status = 'RESOLVED_VALIDATED'").fetchone()[0]
            resolved_cleared = conn.execute("SELECT COUNT(*) FROM investigations WHERE status = 'RESOLVED_CLEARED'").fetchone()[0]

            # High Risk Billing Exposure ($)
            exposure_row = conn.execute(
                """
                SELECT SUM(p.total_claim_amount) as total_exposure
                FROM investigations i
                JOIN providers p ON i.provider_id = p.provider_id
                WHERE i.ai_risk_score >= 60
                """
            ).fetchone()
            total_exposure = float(exposure_row["total_exposure"] or 0.0)

            # High Risk Providers Count
            high_risk_provs = conn.execute(
                "SELECT COUNT(*) FROM providers WHERE risk_score >= 60"
            ).fetchone()[0]

            # Team Workload
            team_rows = conn.execute(
                """
                SELECT COALESCE(assigned_to, 'Unassigned') as member, COUNT(*) as case_count
                FROM investigations
                WHERE status NOT IN ('RESOLVED_VALIDATED', 'RESOLVED_CLEARED', 'CLOSED')
                GROUP BY assigned_to
                ORDER BY case_count DESC
                """
            ).fetchall()

            return {
                "total_investigations": total_cases,
                "escalated_cases": escalated_cnt,
                "active_queue_count": in_review_cnt,
                "validated_fraud_risk_cases": resolved_validated,
                "cleared_cases": resolved_cleared,
                "total_risk_exposure_dollars": round(total_exposure, 2),
                "total_high_risk_providers": high_risk_provs,
                "team_workload": [{"member": r["member"], "cases": r["case_count"]} for r in team_rows]
            }
