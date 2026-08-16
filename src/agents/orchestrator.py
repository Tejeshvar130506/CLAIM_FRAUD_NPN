"""
Multi-Agent Orchestrator Pipeline
---------------------------------
Coordinates the sequential execution of all agents:
Perception Agent -> Fraud Analysis Agent -> Negotiation Agent -> Arbitrator
Auto-provisions Investigation Candidates in SQLite when elevated risk is identified.
"""

import time
import uuid
import logging
import pandas as pd
from typing import Optional, Dict, Any, List

from src.agents.contracts import (
    PerceptionResult, EvidencePackage, NegotiationResult, ArbitratorResult,
    AgentOrchestrationResult
)
from src.agents.perception_agent import PerceptionAgent
from src.agents.fraud_analysis_agent import FraudAnalysisAgent
from src.agents.negotiation_agent import NegotiationAgent
from src.agents.arbitrator_agent import Arbitrator
from src.config import DATABASE_PATH
from src.database.connection import db_transaction
from src.services.audit_service import log_audit_event
from src.services.feature_service import FeatureEngineeringService

logger = logging.getLogger(__name__)


class MultiAgentOrchestrator:
    """
    End-to-End Orchestrator executing the complete multi-agent intelligence pipeline.
    """

    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self.perception_agent = PerceptionAgent(db_path=db_path)
        self.fraud_analysis_agent = FraudAnalysisAgent(db_path=db_path)
        self.negotiation_agent = NegotiationAgent(db_path=db_path)
        self.arbitrator = Arbitrator(db_path=db_path)
        self.feature_service = FeatureEngineeringService()

    def run_provider_pipeline(
        self,
        provider_id: str,
        df_feature_row: pd.DataFrame,
        perception_result: Optional[PerceptionResult] = None,
        actor_username: str = "system"
    ) -> AgentOrchestrationResult:
        """
        Executes Fraud Analysis -> Negotiation -> Arbitrator pipeline for a single provider.
        """
        start_time = time.time()
        run_id = f"ORCH-{uuid.uuid4().hex[:8].upper()}"
        logger.info(f"[{run_id}] Multi-Agent Orchestrator starting analysis for Provider: {provider_id}")

        # 1. Fraud Analysis Agent
        evidence: EvidencePackage = self.fraud_analysis_agent.analyze_provider(
            provider_id=provider_id,
            df_feature_row=df_feature_row,
            actor_username=actor_username
        )

        # 2. Negotiation Agent (Examine -> Argue -> Challenge -> Propose)
        negotiation: NegotiationResult = self.negotiation_agent.negotiate(
            evidence=evidence,
            actor_username=actor_username
        )

        # 3. Arbitrator
        arbitration: ArbitratorResult = self.arbitrator.arbitrate(
            evidence=evidence,
            negotiation=negotiation,
            perception=perception_result,
            actor_username=actor_username
        )

        # 4. Auto-Provision Investigation Candidate if qualified
        case_created = False
        case_number = None

        if arbitration.is_investigation_candidate:
            case_created, case_number = self._ensure_investigation_candidate(
                provider_id=provider_id,
                arbitration=arbitration,
                evidence=evidence,
                actor_username=actor_username
            )

        # 5. Update Provider Record in Database
        self._update_provider_db(provider_id, arbitration, evidence)

        exec_ms = int((time.time() - start_time) * 1000)

        result = AgentOrchestrationResult(
            run_id=run_id,
            provider_id=provider_id,
            status="SUCCESS",
            perception=perception_result,
            fraud_analysis=evidence,
            negotiation=negotiation,
            arbitrator=arbitration,
            investigation_case_created=case_created,
            investigation_case_number=case_number,
            total_execution_time_ms=exec_ms
        )

        log_audit_event(
            username=actor_username,
            role="SYSTEM",
            action="ORCHESTRATION_PIPELINE_COMPLETE",
            entity_type="PROVIDER",
            entity_id=provider_id,
            status="SUCCESS",
            details={
                "run_id": run_id,
                "risk_score": arbitration.final_risk_score,
                "risk_level": arbitration.final_risk_level,
                "is_candidate": arbitration.is_investigation_candidate,
                "case_number": case_number,
                "execution_ms": exec_ms
            },
            db_path=self.db_path
        )

        return result

    def run_pipeline_from_dict(
        self,
        input_dict: Dict[str, Any],
        provider_id: str = "CUSTOM_PROV",
        actor_username: str = "user"
    ) -> AgentOrchestrationResult:
        """
        Executes end-to-end orchestration from interactive form dictionary inputs.
        """
        df_row = self.feature_service.build_feature_vector_from_dict(input_dict, provider_id=provider_id)
        return self.run_provider_pipeline(
            provider_id=provider_id,
            df_feature_row=df_row,
            actor_username=actor_username
        )

    def _ensure_investigation_candidate(
        self,
        provider_id: str,
        arbitration: ArbitratorResult,
        evidence: EvidencePackage,
        actor_username: str
    ) -> (bool, str):
        """
        Creates an investigation candidate record in investigations table if not already present.
        """
        with db_transaction(self.db_path) as conn:
            # Ensure provider exists in providers table to satisfy foreign key constraint
            conn.execute(
                """
                INSERT INTO providers (
                    provider_id, primary_state, fraud_probability, risk_score, risk_level,
                    investigation_priority, status
                ) VALUES (?, ?, ?, ?, ?, ?, 'ACTIVE')
                ON CONFLICT(provider_id) DO UPDATE SET
                    fraud_probability = excluded.fraud_probability,
                    risk_score = excluded.risk_score,
                    risk_level = excluded.risk_level,
                    investigation_priority = excluded.investigation_priority,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    provider_id,
                    evidence.primary_state,
                    evidence.fraud_probability,
                    arbitration.final_risk_score,
                    arbitration.final_risk_level,
                    arbitration.recommended_investigation_priority
                )
            )

            existing = conn.execute(
                "SELECT case_number FROM investigations WHERE provider_id = ?",
                (provider_id,)
            ).fetchone()

            if existing:
                return False, existing["case_number"]

            case_number = f"INV-{provider_id}-{uuid.uuid4().hex[:4].upper()}"
            cursor = conn.execute(
                """
                INSERT INTO investigations (
                    provider_id, case_number, priority, status,
                    ai_risk_score, ai_risk_level, ai_fraud_probability
                ) VALUES (?, ?, ?, 'NEW', ?, ?, ?)
                """,
                (
                    provider_id,
                    case_number,
                    arbitration.recommended_investigation_priority,
                    arbitration.final_risk_score,
                    arbitration.final_risk_level,
                    evidence.fraud_probability
                )
            )
            inv_id = cursor.lastrowid

            # Record creation event in investigation_events
            conn.execute(
                """
                INSERT INTO investigation_events (
                    investigation_id, event_type, actor_username, actor_role,
                    notes, rationale, metadata_json
                ) VALUES (?, 'CREATED', ?, 'AI_ORCHESTRATOR', ?, ?, ?)
                """,
                (
                    inv_id,
                    actor_username,
                    f"Investigation candidate auto-created by AI Arbitrator. Risk Score: {arbitration.final_risk_score}/100.",
                    arbitration.ai_risk_assessment,
                    arbitration.model_dump_json()
                )
            )

            # Insert alert for SIU Investigators
            conn.execute(
                """
                INSERT INTO alerts (alert_type, severity, title, message, entity_id)
                VALUES ('NEW_INVESTIGATION_CANDIDATE', ?, ?, ?, ?)
                """,
                (
                    "CRITICAL" if arbitration.final_risk_score >= 85 else "HIGH",
                    f"New Investigation Candidate: {provider_id}",
                    f"AI Arbitrator flagged provider {provider_id} (Score: {arbitration.final_risk_score}/100, Priority: {arbitration.recommended_investigation_priority}).",
                    case_number
                )
            )

        logger.info(f"Auto-created Investigation Candidate: {case_number} for Provider: {provider_id}")
        return True, case_number

    def _update_provider_db(
        self,
        provider_id: str,
        arbitration: ArbitratorResult,
        evidence: EvidencePackage
    ) -> None:
        """Updates provider table risk score and priority."""
        try:
            with db_transaction(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE providers SET
                        fraud_probability = ?,
                        risk_score = ?,
                        risk_level = ?,
                        investigation_priority = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE provider_id = ?
                    """,
                    (
                        evidence.fraud_probability,
                        arbitration.final_risk_score,
                        arbitration.final_risk_level,
                        arbitration.recommended_investigation_priority,
                        provider_id
                    )
                )
        except Exception as e:
            logger.warning(f"Could not update provider table for {provider_id}: {e}")
