"""
Negotiation Agent Module
------------------------
Third stage in the Multi-Agent Healthcare Provider Fraud Intelligence Platform.
Executes a structured 4-step adversarial reasoning workflow:
1. EXAMINE: Ingests structured EvidencePackage (financials, utilization, physician overlaps, concentrations)
2. ARGUE: Constructs strongest evidence-based points supporting audit investigation
3. CHALLENGE: Formulates skeptical counter-arguments, identifying legitimate clinical/operational explanations
4. PROPOSE: Synthesizes a balanced, structured recommendation (e.g. HIGH_PRIORITY_INVESTIGATION, AUDIT_REVIEW)
"""

import os
import time
import uuid
import json
import logging
from typing import List, Dict, Any, Optional

from src.agents.contracts import (
    EvidencePackage, NegotiationArgument, NegotiationChallenge, NegotiationResult
)
from src.config import DATABASE_PATH, GEMINI_API_KEY, LLM_MODEL_NAME, ENABLE_LLM_REASONING
from src.database.connection import db_transaction
from src.services.audit_service import log_audit_event

logger = logging.getLogger(__name__)


class NegotiationAgent:
    """
    Adversarial Negotiation Agent balancing fraud risk indicators against legitimate clinical alternatives.
    """

    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path

    def negotiate(
        self,
        evidence: EvidencePackage,
        actor_username: str = "system"
    ) -> NegotiationResult:
        """
        Executes the 4-stage Examine -> Argue -> Challenge -> Propose reasoning cycle.
        """
        start_time = time.time()
        run_id = f"NEG-{uuid.uuid4().hex[:8].upper()}"
        prov_id = evidence.provider_id
        logger.info(f"[{run_id}] Negotiation Agent evaluating Evidence for Provider: {prov_id}")

        # 1. EXAMINE: Synthesize evidence highlights
        examined_summary = self._examine_evidence(evidence)

        # 2. ARGUE: Build pro-investigation evidence arguments
        arguments = self._build_investigation_arguments(evidence)

        # 3. CHALLENGE: Build skeptical counter-hypotheses and clinical mitigations
        challenges = self._build_counter_challenges(evidence)

        # 4. PROPOSE: Formulate final balanced recommendation
        proposed_action, confidence, synthesis = self._formulate_proposal(evidence, arguments, challenges)

        exec_ms = int((time.time() - start_time) * 1000)

        result = NegotiationResult(
            provider_id=prov_id,
            examined_evidence_summary=examined_summary,
            arguments_supporting_investigation=arguments,
            counter_challenges_and_mitigations=challenges,
            proposed_action=proposed_action,
            confidence_rating=confidence,
            plain_language_synthesis=synthesis
        )

        self._record_agent_run(run_id, prov_id, result, exec_ms, actor_username)
        return result

    def _examine_evidence(self, evidence: EvidencePackage) -> str:
        """Examines and summarizes key risk vectors."""
        parts = [
            f"Provider {evidence.provider_id} assessed with model-estimated fraud probability of {evidence.fraud_probability*100:.1f}% ",
            f"(Risk Score: {evidence.risk_score}/100, Level: {evidence.risk_level}, Priority: {evidence.investigation_priority}). ",
            evidence.peer_comparison_summary, " ",
            evidence.network_concentration_summary
        ]
        return "".join(parts)

    def _build_investigation_arguments(self, evidence: EvidencePackage) -> List[NegotiationArgument]:
        """Constructs evidence-based arguments supporting audit investigation."""
        args = []
        point_idx = 1

        for m in evidence.behavioral_metrics:
            if m.severity in {"CRITICAL", "HIGH"}:
                args.append(NegotiationArgument(
                    point_id=f"ARG-{point_idx}",
                    title=f"Significant Discrepancy in {m.display_name}",
                    evidence_basis=m.plain_language_explanation,
                    risk_indicator=f"Severity marked as {m.severity}",
                    strength="CRITICAL" if m.severity == "CRITICAL" else "HIGH"
                ))
                point_idx += 1

        # Check EBM score attributions
        high_ebm = [c for c in evidence.ebm_contributions if c.direction == "INCREASES_RISK" and c.score_contribution > 0.10]
        for c in high_ebm[:2]:
            args.append(NegotiationArgument(
                point_id=f"ARG-{point_idx}",
                title=f"EBM Risk Attribution: {c.display_name}",
                evidence_basis=c.plain_summary,
                risk_indicator=f"Additive risk score contribution of +{c.score_contribution:.2f}",
                strength="HIGH"
            ))
            point_idx += 1

        if not args:
            args.append(NegotiationArgument(
                point_id="ARG-1",
                title="Routine Practice Pattern Alignment",
                evidence_basis="No severe statistical deviations detected across primary billing indicators.",
                risk_indicator="Metrics within normal expected ranges",
                strength="LOW"
            ))

        return args

    def _build_counter_challenges(self, evidence: EvidencePackage) -> List[NegotiationChallenge]:
        """Formulates skeptical alternative hypotheses and clinical justifications."""
        challenges = []
        c_idx = 1

        # Challenge 1: Inpatient Ratio vs Tertiary Care / Specialization
        inp_metrics = [m for m in evidence.behavioral_metrics if m.metric_name == "inpatient_ratio" and m.severity in {"HIGH", "CRITICAL"}]
        if inp_metrics:
            challenges.append(NegotiationChallenge(
                challenge_id=f"CHAL-{c_idx}",
                alternative_hypothesis="Subspecialized Inpatient Facility / Surgical Center",
                mitigating_factors=[
                    "Provider may operate as a specialized surgical hospital, acute rehabilitation center, or long-term acute care hospital (LTACH).",
                    "Patient acuity case-mix index naturally elevates inpatient admission requirements."
                ],
                clinical_plausibility="PLAUSIBLE",
                data_limitation_notes="Dataset lacks explicit hospital designation license taxonomy codes."
            ))
            c_idx += 1

        # Challenge 2: High Repeat Patient Concentration vs Chronic Disease Management
        repeat_metrics = [m for m in evidence.behavioral_metrics if m.metric_name == "repeat_beneficiary_ratio" and m.severity in {"HIGH", "CRITICAL"}]
        if repeat_metrics:
            challenges.append(NegotiationChallenge(
                challenge_id=f"CHAL-{c_idx}",
                alternative_hypothesis="Specialized Chronic Illness or Oncology / Dialysis Clinic",
                mitigating_factors=[
                    "High repeat visits are clinically typical for outpatient hemodialysis, chemotherapy regimens, or wound care protocols.",
                    "Geographic isolation in rural counties can concentrate regional patient cohorts on a single qualified provider."
                ],
                clinical_plausibility="PLAUSIBLE",
                data_limitation_notes="Regional demographic density and competitor provider density not fully captured in claims file."
            ))
            c_idx += 1

        # Challenge 3: Attending & Operating Physician Overlap vs Solo Practitioner
        phys_metrics = [m for m in evidence.behavioral_metrics if m.metric_name == "same_attending_operating_ratio" and m.severity in {"HIGH", "CRITICAL"}]
        if phys_metrics:
            challenges.append(NegotiationChallenge(
                challenge_id=f"CHAL-{c_idx}",
                alternative_hypothesis="Solo Practice Surgeon / Independent Specialist",
                mitigating_factors=[
                    "Solo practitioner surgeons routinely act as both the attending admitting physician and the primary operating surgeon.",
                    "Small clinical group practices may not maintain separate resident or assistant surgical billing codes."
                ],
                clinical_plausibility="PLAUSIBLE",
                data_limitation_notes="Group practice taxonomy vs solo NPI identifier not explicitly separated."
            ))
            c_idx += 1

        if not challenges:
            challenges.append(NegotiationChallenge(
                challenge_id="CHAL-1",
                alternative_hypothesis="Standard Community Practice",
                mitigating_factors=["Provider activity aligns with standard outpatient community distribution."],
                clinical_plausibility="PLAUSIBLE",
                data_limitation_notes="No anomalous clinical vectors identified."
            ))

        return challenges

    def _formulate_proposal(
        self,
        evidence: EvidencePackage,
        arguments: List[NegotiationArgument],
        challenges: List[NegotiationChallenge]
    ) -> (str, str, str):
        """Formulates final structured proposal and plain language synthesis."""
        score = evidence.risk_score
        high_args_cnt = sum(1 for a in arguments if a.strength in {"CRITICAL", "HIGH"})
        
        if score >= 85 and high_args_cnt >= 2:
            action = "HIGH_PRIORITY_INVESTIGATION"
            confidence = "HIGH"
            synthesis = (
                f"Multi-vector anomalies across billing, utilization, and repeat patient networks strongly support "
                f"opening a high-priority audit investigation. While specialized practice represents a plausible alternative, "
                f"the magnitude of peer deviation warrants targeted medical record verification."
            )
        elif score >= 60:
            action = "AUDIT_REVIEW"
            confidence = "MODERATE"
            synthesis = (
                f"Elevated risk score ({score}/100) and statistical peer divergence justify formal SIU investigator review. "
                f"Counter-arguments regarding specialized care and local patient concentration should be cross-referenced during triage."
            )
        elif score >= 30:
            action = "ROUTINE_MONITORING"
            confidence = "MODERATE"
            synthesis = (
                f"Moderate indicators detected. Plausible clinical justifications exist; recommended action is routine monitoring "
                f"without urgent payment suspension."
            )
        else:
            action = "LOW_CONCERN"
            confidence = "HIGH"
            synthesis = (
                f"Provider metrics remain well within normal population benchmarks with minimal risk indicators. Low priority."
            )

        return action, confidence, synthesis

    def _record_agent_run(
        self,
        run_id: str,
        provider_id: str,
        result: NegotiationResult,
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
                        "NEGOTIATION_AGENT",
                        "COMPLETED",
                        f"Arguments: {len(result.arguments_supporting_investigation)}, Challenges: {len(result.counter_challenges_and_mitigations)}, Proposal: {result.proposed_action}",
                        result.model_dump_json(),
                        exec_ms
                    )
                )

            log_audit_event(
                username=username,
                role="SYSTEM",
                action="NEGOTIATION_AGENT_RUN",
                entity_type="PROVIDER",
                entity_id=provider_id,
                status="SUCCESS",
                details={
                    "proposed_action": result.proposed_action,
                    "confidence": result.confidence_rating,
                    "arguments_count": len(result.arguments_supporting_investigation),
                    "challenges_count": len(result.counter_challenges_and_mitigations)
                },
                db_path=self.db_path
            )
        except Exception as e:
            logger.error(f"Failed to record Negotiation Agent run: {e}")
