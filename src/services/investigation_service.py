"""
Investigation Case Management Service
--------------------------------------
Manages the human-in-the-loop clinical investigation lifecycle for SIU investigators:
- Investigation candidate triage and case assignment
- Clinical case notes and evidence review logs
- PATH A: Sufficient Evidence Findings and Case Resolution (Validated / Cleared)
- PATH B: Insufficient Evidence Escalation to Management with mandatory structured reasoning
- Complete timeline event tracking in investigation_events table
"""

import uuid
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.config import DATABASE_PATH
from src.database.connection import db_transaction
from src.services.audit_service import log_audit_event

logger = logging.getLogger(__name__)


class InvestigationService:
    """
    Service managing investigation candidate queues, investigator decisions, and escalations.
    """

    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path

    def get_investigation_queue(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        assigned_to: Optional[str] = None,
        search_query: Optional[str] = None,
        order_by: str = "NEWEST",
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Fetches filtered investigation cases with provider metrics joined.
        Supports ordering by NEWEST, OLDEST, or RISK_SCORE.
        """
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
            WHERE 1=1
        """
        params = []

        if status:
            query += " AND i.status = ?"
            params.append(status.upper().strip())
        if priority:
            query += " AND i.priority = ?"
            params.append(priority.upper().strip())
        if assigned_to:
            query += " AND i.assigned_to = ?"
            params.append(assigned_to.strip().lower())
        if search_query:
            query += " AND (i.provider_id LIKE ? OR i.case_number LIKE ?)"
            params.extend([f"%{search_query.strip().upper()}%", f"%{search_query.strip().upper()}%"])

        order_clause = " ORDER BY i.id DESC"
        if order_by == "OLDEST":
            order_clause = " ORDER BY i.id ASC"
        elif order_by == "RISK_SCORE":
            order_clause = " ORDER BY i.ai_risk_score DESC, i.id DESC"

        query += order_clause + " LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with db_transaction(self.db_path) as conn:
            cursor = conn.execute(query, params)
            return [dict(r) for r in cursor.fetchall()]

    def get_investigation_by_id(self, case_id: int) -> Optional[Dict[str, Any]]:
        """Fetches complete investigation details by case ID."""
        query = """
            SELECT 
                i.*, 
                p.primary_state, p.total_claims, p.total_claim_amount, p.average_claim_amount,
                p.inpatient_ratio, p.repeat_beneficiary_ratio, p.average_claim_vs_peer_average,
                p.claim_amount_vs_peer_average, p.same_attending_operating_ratio, p.claims_per_month
            FROM investigations i
            LEFT JOIN providers p ON i.provider_id = p.provider_id
            WHERE i.id = ?
        """
        with db_transaction(self.db_path) as conn:
            row = conn.execute(query, (case_id,)).fetchone()
            return dict(row) if row else None

    def get_investigation_by_provider(self, provider_id: str) -> Optional[Dict[str, Any]]:
        """Fetches investigation record by provider ID."""
        with db_transaction(self.db_path) as conn:
            row = conn.execute("SELECT * FROM investigations WHERE provider_id = ?", (provider_id,)).fetchone()
            return dict(row) if row else None

    def get_case_events(self, investigation_id: int) -> List[Dict[str, Any]]:
        """Retrieves chronological timeline of actions for a given investigation."""
        with db_transaction(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM investigation_events WHERE investigation_id = ? ORDER BY id ASC",
                (investigation_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def assign_case(self, investigation_id: int, assignee_username: str, actor_username: str) -> bool:
        """Assigns an investigation case to an investigator."""
        with db_transaction(self.db_path) as conn:
            conn.execute(
                """
                UPDATE investigations 
                SET assigned_to = ?, status = 'ASSIGNED', updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (assignee_username.strip().lower(), investigation_id)
            )

            conn.execute(
                """
                INSERT INTO investigation_events (investigation_id, event_type, actor_username, actor_role, notes)
                VALUES (?, 'ASSIGNED', ?, 'MANAGER', ?)
                """,
                (investigation_id, actor_username, f"Assigned case to investigator: {assignee_username}")
            )

        log_audit_event(
            username=actor_username,
            role="MANAGER",
            action="ASSIGN_CASE",
            entity_type="INVESTIGATION",
            entity_id=str(investigation_id),
            status="SUCCESS",
            details={"assigned_to": assignee_username},
            db_path=self.db_path
        )
        return True

    def add_case_note(self, investigation_id: int, note_text: str, actor_username: str, actor_role: str = "INVESTIGATOR") -> int:
        """Adds a timestamped clinical case note to the investigation timeline."""
        with db_transaction(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO investigation_events (investigation_id, event_type, actor_username, actor_role, notes)
                VALUES (?, 'NOTE_ADDED', ?, ?, ?)
                """,
                (investigation_id, actor_username, actor_role, note_text.strip())
            )
            event_id = cursor.lastrowid

            conn.execute(
                "UPDATE investigations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (investigation_id,)
            )

        log_audit_event(
            username=actor_username,
            role=actor_role,
            action="ADD_INVESTIGATION_NOTE",
            entity_type="INVESTIGATION",
            entity_id=str(investigation_id),
            status="SUCCESS",
            details={"note_preview": note_text[:80]},
            db_path=self.db_path
        )
        return event_id

    # ==========================================================================
    # PATH A — SUFFICIENT EVIDENCE: INVESTIGATOR OUTCOME
    # ==========================================================================

    def record_investigator_finding(
        self,
        investigation_id: int,
        finding_type: str,  # "ELEVATED_RISK_VALIDATED", "SUSPICION_CLEARED_LEGITIMATE", "MONITORING_CONTINUED"
        reasoning: str,
        follow_up_action: str,
        actor_username: str
    ) -> bool:
        """
        PATH A: Investigator has sufficient evidence to reach a finding.
        """
        valid_findings = {
            "ELEVATED_RISK_VALIDATED": "RESOLVED_VALIDATED",
            "SUSPICION_CLEARED_LEGITIMATE": "RESOLVED_CLEARED",
            "MONITORING_CONTINUED": "IN_REVIEW"
        }
        new_status = valid_findings.get(finding_type, "RESOLVED_VALIDATED")

        with db_transaction(self.db_path) as conn:
            conn.execute(
                """
                UPDATE investigations SET
                    status = ?,
                    final_outcome = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (new_status, f"{finding_type}: {follow_up_action}", investigation_id)
            )

            conn.execute(
                """
                INSERT INTO investigation_events (
                    investigation_id, event_type, actor_username, actor_role,
                    decision, rationale, notes
                ) VALUES (?, 'FINDING_RECORDED', ?, 'INVESTIGATOR', ?, ?, ?)
                """,
                (
                    investigation_id,
                    actor_username,
                    finding_type,
                    reasoning.strip(),
                    f"Follow-up action: {follow_up_action.strip()}"
                )
            )

        log_audit_event(
            username=actor_username,
            role="INVESTIGATOR",
            action="RECORD_INVESTIGATION_FINDING",
            entity_type="INVESTIGATION",
            entity_id=str(investigation_id),
            status="SUCCESS",
            details={
                "finding_type": finding_type,
                "status": new_status,
                "follow_up": follow_up_action
            },
            db_path=self.db_path
        )
        return True

    # ==========================================================================
    # PATH B — INSUFFICIENT EVIDENCE: ESCALATE TO MANAGEMENT
    # ==========================================================================

    def escalate_to_management(
        self,
        investigation_id: int,
        escalation_reason: str,
        actor_username: str
    ) -> bool:
        """
        PATH B: Available evidence is insufficient for investigator determination;
        escalates case to Management Review Queue with required reasoning.
        """
        if not escalation_reason or len(escalation_reason.strip()) < 10:
            raise ValueError("Structured escalation rationale of at least 10 characters is required.")

        with db_transaction(self.db_path) as conn:
            # Fetch provider ID for alert
            inv = conn.execute("SELECT provider_id, case_number FROM investigations WHERE id = ?", (investigation_id,)).fetchone()
            prov_id = inv["provider_id"] if inv else "UNKNOWN"
            case_num = inv["case_number"] if inv else str(investigation_id)

            conn.execute(
                """
                UPDATE investigations SET
                    status = 'ESCALATED',
                    escalation_reason = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (escalation_reason.strip(), investigation_id)
            )

            conn.execute(
                """
                INSERT INTO investigation_events (
                    investigation_id, event_type, actor_username, actor_role,
                    decision, rationale, notes
                ) VALUES (?, 'ESCALATED_TO_MANAGEMENT', ?, 'INVESTIGATOR', 'ESCALATED_FOR_MANAGEMENT_REVIEW', ?, ?)
                """,
                (
                    investigation_id,
                    actor_username,
                    escalation_reason.strip(),
                    "Investigator determined available claims evidence is insufficient to reach confident conclusion; escalated to management."
                )
            )

            # Insert alert for Management
            conn.execute(
                """
                INSERT INTO alerts (alert_type, severity, title, message, entity_id)
                VALUES ('CASE_ESCALATED_TO_MANAGEMENT', 'HIGH', ?, ?, ?)
                """,
                (
                    f"Case Escalated to Management: {prov_id}",
                    f"Investigator {actor_username} escalated case {case_num} due to insufficient evidence.",
                    str(investigation_id)
                )
            )

        log_audit_event(
            username=actor_username,
            role="INVESTIGATOR",
            action="ESCALATE_TO_MANAGEMENT",
            entity_type="INVESTIGATION",
            entity_id=str(investigation_id),
            status="SUCCESS",
            details={
                "provider_id": prov_id,
                "escalation_reason": escalation_reason[:100]
            },
            db_path=self.db_path
        )
        return True
