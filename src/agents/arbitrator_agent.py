"""
Arbitrator Agent Module
-----------------------
Fourth stage in the Multi-Agent Healthcare Provider Fraud Intelligence Platform.
Responsible for:
- Independent multi-agent evidence synthesis
- Cross-evaluating Perception data quality, Fraud Analysis ML metrics, and Negotiation arguments/challenges
- Resolving conflicting evidence and quantifying epistemic uncertainty
- Determining whether the provider qualifies as an "Investigation Candidate"
- Producing the final, defensible ArbitratorResult
"""

import time
import uuid
import logging
from typing import Optional, Dict, Any

from src.agents.contracts import (
    EvidencePackage, NegotiationResult, PerceptionResult, ArbitratorResult
)
from src.config import DATABASE_PATH
from src.database.connection import db_transaction
from src.services.audit_service import log_audit_event

logger = logging.getLogger(__name__)


class Arbitrator:
    """
    Independent Arbitrator Agent synthesizing multi-agent findings into a final AI Risk Assessment.
    """

    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path

    def arbitrate(
        self,
        evidence: EvidencePackage,
        negotiation: NegotiationResult,
        perception: Optional[PerceptionResult] = None,
        actor_username: str = "system"
    ) -> ArbitratorResult:
        """
        Synthesizes all agent outputs into final assessment.
        """
        start_time = time.time()
        run_id = f"ARB-{uuid.uuid4().hex[:8].upper()}"
        prov_id = evidence.provider_id
        logger.info(f"[{run_id}] Arbitrator independently evaluating Provider: {prov_id}")

        # 1. Quality & Perception Check
        quality_score = perception.quality_report.overall_quality_score if perception else 100.0
        data_uncertainty = "Low" if quality_score >= 85.0 else ("Moderate" if quality_score >= 60.0 else "High")

        # 2. Conflicting Evidence Evaluation
        conflicts = []
        high_args = [a for a in negotiation.arguments_supporting_investigation if a.strength in {"CRITICAL", "HIGH"}]
        plausible_chals = [c for c in negotiation.counter_challenges_and_mitigations if c.clinical_plausibility == "PLAUSIBLE"]

        if high_args and plausible_chals:
            conflicts.append(
                f"Statistical billing anomaly noted ({len(high_args)} strong indicators), but mitigated by "
                f"plausible clinical specialization hypotheses ({len(plausible_chals)} counter-explanations)."
            )
        elif not high_args and plausible_chals:
            conflicts.append("No critical statistical anomalies found; provider activity aligns with standard practice.")

        conflicting_text = " ".join(conflicts) if conflicts else "Evidence vectors align consistently across all evaluated dimensions."

        # 3. Alternative Explanations Evaluated
        chal_summaries = [f"{c.alternative_hypothesis} ({', '.join(c.mitigating_factors[:1])})" for c in negotiation.counter_challenges_and_mitigations]
        alt_text = "; ".join(chal_summaries) if chal_summaries else "No significant alternate clinical hypotheses required."

        # 4. Final Assessment & Priority Determination
        score = evidence.risk_score
        is_candidate = score >= 60 or negotiation.proposed_action in {"HIGH_PRIORITY_INVESTIGATION", "AUDIT_REVIEW"}

        if score >= 85:
            assessment = "Elevated fraud risk identified based on substantial peer reimbursement divergence and repeat patient concentration."
            priority = "CRITICAL"
            final_level = "CRITICAL"
        elif score >= 60:
            assessment = "Potentially suspicious provider behavior identified requiring human audit investigation."
            priority = "HIGH"
            final_level = "HIGH"
        elif score >= 30:
            assessment = "Moderate statistical deviation identified; provider recommended for routine monitoring."
            priority = "NORMAL"
            final_level = "MEDIUM"
        else:
            assessment = "Lower risk provider profile; metrics consistent with peer group baselines."
            priority = "LOW"
            final_level = "LOW"

        # 5. Uncertainty Assessment
        uncertainty = (
            f"Epistemic uncertainty rated as {data_uncertainty}. "
            f"Model-estimated probability is {evidence.fraud_probability*100:.1f}%. "
            f"Final determination requires human medical record audit verification."
        )

        concise_summary = (
            f"AI Risk Assessment: {assessment} [Risk Score: {score}/100, Level: {final_level}, "
            f"Investigation Candidate: {'YES' if is_candidate else 'NO'}]."
        )

        exec_ms = int((time.time() - start_time) * 1000)

        result = ArbitratorResult(
            provider_id=prov_id,
            ai_risk_assessment=assessment,
            final_risk_level=final_level,
            final_risk_score=score,
            recommended_investigation_priority=priority,
            is_investigation_candidate=is_candidate,
            evidence_synthesis=negotiation.examined_evidence_summary,
            conflicting_evidence_analysis=conflicting_text,
            alternative_explanations_evaluated=alt_text,
            uncertainty_assessment=uncertainty,
            concise_summary=concise_summary
        )

        self._record_agent_run(run_id, prov_id, result, exec_ms, actor_username)
        return result

    def _record_agent_run(
        self,
        run_id: str,
        provider_id: str,
        result: ArbitratorResult,
        exec_ms: int,
        username: str
    ) -> None:
        """Persists agent execution and audit log."""
        try:
            with db_transaction(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO agent_runs (run_id, provider_id, agent_name, status, input_summary, output_json, execution_time_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        provider_id,
                        "ARBITRATOR",
                        "COMPLETED",
                        f"Score: {result.final_risk_score}/100, Level: {result.final_risk_level}, Candidate: {result.is_investigation_candidate}",
                        result.model_dump_json(),
                        exec_ms
                    )
                )

            log_audit_event(
                username=username,
                role="SYSTEM",
                action="ARBITRATOR_RUN",
                entity_type="PROVIDER",
                entity_id=provider_id,
                status="SUCCESS",
                details={
                    "assessment": result.ai_risk_assessment,
                    "final_score": result.final_risk_score,
                    "final_level": result.final_risk_level,
                    "is_candidate": result.is_investigation_candidate,
                    "priority": result.recommended_investigation_priority
                },
                db_path=self.db_path
            )
        except Exception as e:
            logger.error(f"Failed to record Arbitrator execution: {e}")
