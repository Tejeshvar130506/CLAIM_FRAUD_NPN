"""
Typed Agent Contracts & State Schema Definitions
------------------------------------------------
Defines structured Pydantic data contracts for all multi-agent stages:
- Perception Agent
- Fraud Analysis Agent
- Negotiation Agent
- Arbitrator
- Complete Orchestration State & Evidence Package
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone


# ==============================================================================
# 1. PERCEPTION AGENT CONTRACTS
# ==============================================================================

class ColumnProfile(BaseModel):
    name: str
    dtype: str
    role: str
    missing_count: int
    missing_pct: float
    unique_count: int
    sample_values: List[str] = Field(default_factory=list)


class FilePerceptionSummary(BaseModel):
    filename: str
    group: str
    row_count: int
    col_count: int
    duplicate_rows: int
    size_mb: float
    columns: List[ColumnProfile] = Field(default_factory=list)


class DataQualityReport(BaseModel):
    overall_quality_score: float = 100.0  # 0 to 100 scale
    total_files_analyzed: int
    total_records: int
    key_integrity_status: str  # "VALIDATED", "WARNING", "FAILED"
    provider_key_present: bool
    beneficiary_key_present: bool
    claim_key_present: bool
    duplicate_keys_found: int = 0
    warnings: List[str] = Field(default_factory=list)
    anomalies_detected: List[str] = Field(default_factory=list)


class PerceptionResult(BaseModel):
    run_id: str
    status: str = "COMPLETED"  # "COMPLETED", "FAILED"
    dataset_group: str  # "TRAIN", "TEST", "UPLOADED_BATCH", "SINGLE_INPUT"
    files_profiled: List[FilePerceptionSummary] = Field(default_factory=list)
    total_providers_detected: int = 0
    total_claims_detected: int = 0
    total_beneficiaries_detected: int = 0
    quality_report: DataQualityReport
    preprocessed_path: Optional[str] = None
    execution_time_ms: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ==============================================================================
# 2. FRAUD ANALYSIS AGENT & EVIDENCE CONTRACTS
# ==============================================================================

class BehavioralMetric(BaseModel):
    metric_name: str
    display_name: str
    value: float
    peer_benchmark: Optional[float] = None
    peer_ratio: Optional[float] = None
    severity: str = "NORMAL"  # "NORMAL", "ELEVATED", "HIGH", "CRITICAL"
    plain_language_explanation: str


class EBMFeatureContribution(BaseModel):
    feature_name: str
    display_name: str
    actual_value: float
    score_contribution: float  # Additive GAM score effect
    direction: str  # "INCREASES_RISK", "DECREASES_RISK", "NEUTRAL"
    plain_summary: str


class EvidencePackage(BaseModel):
    provider_id: str
    primary_state: str
    fraud_probability: float  # 0.0 to 1.0 (Model-estimated probability)
    risk_score: int  # 0 to 100
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    investigation_priority: str  # "LOW", "NORMAL", "HIGH", "CRITICAL"
    behavioral_metrics: List[BehavioralMetric] = Field(default_factory=list)
    ebm_contributions: List[EBMFeatureContribution] = Field(default_factory=list)
    key_findings: List[str] = Field(default_factory=list)
    peer_comparison_summary: str
    network_concentration_summary: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ==============================================================================
# 3. NEGOTIATION AGENT CONTRACTS
# ==============================================================================

class NegotiationArgument(BaseModel):
    point_id: str
    title: str
    evidence_basis: str
    risk_indicator: str
    strength: str = "HIGH"  # "LOW", "MEDIUM", "HIGH", "CRITICAL"


class NegotiationChallenge(BaseModel):
    challenge_id: str
    alternative_hypothesis: str
    mitigating_factors: List[str] = Field(default_factory=list)
    clinical_plausibility: str  # "PLAUSIBLE", "PARTIALLY_PLAUSIBLE", "UNLIKELY"
    data_limitation_notes: Optional[str] = None


class NegotiationResult(BaseModel):
    provider_id: str
    examined_evidence_summary: str
    arguments_supporting_investigation: List[NegotiationArgument] = Field(default_factory=list)
    counter_challenges_and_mitigations: List[NegotiationChallenge] = Field(default_factory=list)
    proposed_action: str  # "LOW_CONCERN", "ROUTINE_MONITORING", "AUDIT_REVIEW", "HIGH_PRIORITY_INVESTIGATION", "INSUFFICIENT_EVIDENCE"
    confidence_rating: str = "MODERATE"  # "LOW", "MODERATE", "HIGH"
    plain_language_synthesis: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ==============================================================================
# 4. ARBITRATOR CONTRACTS
# ==============================================================================

class ArbitratorResult(BaseModel):
    provider_id: str
    ai_risk_assessment: str  # e.g., "Elevated fraud risk identified based on peer billing divergence and repeat patient loops."
    final_risk_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    final_risk_score: int  # 0 to 100
    recommended_investigation_priority: str  # "LOW", "NORMAL", "HIGH", "CRITICAL"
    is_investigation_candidate: bool
    evidence_synthesis: str
    conflicting_evidence_analysis: str
    alternative_explanations_evaluated: str
    uncertainty_assessment: str
    concise_summary: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ==============================================================================
# 5. ORCHESTRATION PIPELINE CONTRACT
# ==============================================================================

class AgentOrchestrationResult(BaseModel):
    run_id: str
    provider_id: str
    status: str = "SUCCESS"
    perception: Optional[PerceptionResult] = None
    fraud_analysis: Optional[EvidencePackage] = None
    negotiation: Optional[NegotiationResult] = None
    arbitrator: Optional[ArbitratorResult] = None
    investigation_case_created: bool = False
    investigation_case_number: Optional[str] = None
    total_execution_time_ms: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
