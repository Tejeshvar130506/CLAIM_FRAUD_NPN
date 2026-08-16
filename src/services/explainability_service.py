"""
Explainability & Evidence Generation Service
--------------------------------------------
Synthesizes ML model predictions (XGBoost), glass-box additive feature attributions (EBM),
and statistical peer benchmarks into a standardized, human-readable EvidencePackage.
"""

import os
import logging
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from src.config import MODELS_DIR, FEATURES_DATA_DIR, DATABASE_PATH
from src.agents.contracts import (
    BehavioralMetric, EBMFeatureContribution, EvidencePackage
)
from src.risk_scoring import calculate_risk_level
from src.services.feature_service import MODEL_FEATURE_COLUMNS

logger = logging.getLogger(__name__)

# Feature display mapping for human readability
FEATURE_DISPLAY_NAMES = {
    'total_claims': 'Total Claims Volume',
    'total_claim_amount': 'Total Reimbursement Amount ($)',
    'average_claim_amount': 'Average Claim Amount ($)',
    'maximum_claim_amount': 'Maximum Single Claim ($)',
    'inpatient_claim_count': 'Inpatient Admissions Count',
    'inpatient_ratio': 'Inpatient Admission Ratio',
    'average_length_of_stay': 'Average Inpatient Length of Stay (Days)',
    'same_attending_operating_ratio': 'Attending & Operating Physician Overlap Ratio',
    'unique_attending_physicians': 'Unique Attending Physicians',
    'unique_operating_physicians': 'Unique Operating Physicians',
    'repeat_beneficiary_ratio': 'Repeat Beneficiary Ratio',
    'beneficiary_hhi_concentration': 'Beneficiary Concentration Index (HHI)',
    'top_bene_claim_share': 'Top Beneficiary Claim Share',
    'claims_per_month': 'Monthly Claim Velocity',
    'average_claim_vs_peer_average': 'Average Claim vs State Peer Ratio',
    'claim_amount_vs_peer_average': 'Total Billing vs State Peer Ratio',
    'peer_claim_volume_zscore': 'Peer Volume Deviation Z-Score',
    'chronic_cond_score_mean': 'Average Patient Chronic Conditions'
}


class ExplainabilityService:
    """
    Service for computing EBM contributions, SHAP values, and building standardized Evidence Packages.
    """

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.xgb_model = None
        self.ebm_model = None
        self._load_models()

    def _load_models(self) -> None:
        """Loads trained XGBoost and EBM models."""
        xgb_path = os.path.join(self.models_dir, "final_fraud_model.pkl")
        if not os.path.exists(xgb_path):
            xgb_path = os.path.join(self.models_dir, "xgboost_fraud_model.pkl")

        ebm_path = os.path.join(self.models_dir, "ebm_model.pkl")

        if os.path.exists(xgb_path):
            self.xgb_model = joblib.load(xgb_path)
        if os.path.exists(ebm_path):
            self.ebm_model = joblib.load(ebm_path)

    def generate_evidence_package(
        self,
        provider_id: str,
        df_feature_row: pd.DataFrame
    ) -> EvidencePackage:
        """
        Generates a comprehensive EvidencePackage for a single provider.
        df_feature_row must be a 1-row DataFrame containing the 52 canonical features.
        """
        if self.xgb_model is None:
            self._load_models()

        X = df_feature_row[MODEL_FEATURE_COLUMNS].fillna(0.0)
        
        # 1. Model-estimated fraud probability
        if self.xgb_model is not None:
            fraud_prob = float(self.xgb_model.predict_proba(X)[0, 1])
        else:
            fraud_prob = 0.05

        risk_score = int(round(fraud_prob * 100.0))
        risk_level = calculate_risk_level(risk_score)

        # Determine Investigation Priority
        if risk_score >= 85:
            inv_priority = "CRITICAL"
        elif risk_score >= 60:
            inv_priority = "HIGH"
        elif risk_score >= 30:
            inv_priority = "NORMAL"
        else:
            inv_priority = "LOW"

        row_dict = df_feature_row.iloc[0].to_dict()
        primary_state = str(row_dict.get('primary_state', 'N/A'))

        # 2. Extract EBM Additive Feature Contributions
        ebm_contributions = self._extract_ebm_contributions(X)

        # 3. Build Behavioral Metrics Breakdown
        behavioral_metrics = self._build_behavioral_metrics(row_dict)

        # 4. Generate Plain-Language Key Findings
        key_findings = self._generate_key_findings(row_dict, fraud_prob, risk_score)

        # 5. Summaries
        peer_ratio_avg = float(row_dict.get('average_claim_vs_peer_average', 1.0))
        peer_ratio_tot = float(row_dict.get('claim_amount_vs_peer_average', 1.0))
        peer_summary = (
            f"Provider average reimbursement is {peer_ratio_avg:.1f}x state peer norm; "
            f"total billing volume is {peer_ratio_tot:.1f}x state peer benchmark."
        )

        repeat_ratio = float(row_dict.get('repeat_beneficiary_ratio', 0.0))
        hhi = float(row_dict.get('beneficiary_hhi_concentration', 0.0))
        net_summary = (
            f"Repeat patient ratio is {repeat_ratio*100.0:.1f}%; "
            f"beneficiary concentration index (HHI) is {hhi:.3f}."
        )

        return EvidencePackage(
            provider_id=provider_id,
            primary_state=primary_state,
            fraud_probability=round(fraud_prob, 4),
            risk_score=risk_score,
            risk_level=risk_level,
            investigation_priority=inv_priority,
            behavioral_metrics=behavioral_metrics,
            ebm_contributions=ebm_contributions,
            key_findings=key_findings,
            peer_comparison_summary=peer_summary,
            network_concentration_summary=net_summary
        )

    def _extract_ebm_contributions(self, X: pd.DataFrame, top_k: int = 8) -> List[EBMFeatureContribution]:
        """Extracts top positive and negative score contributions from EBM model."""
        if self.ebm_model is None:
            return []

        try:
            exp = self.ebm_model.explain_local(X)
            names = exp.data(0)['names']
            scores = exp.data(0)['scores']
            vals = exp.data(0)['values']

            items = []
            for name, score, val in zip(names, scores, vals):
                if name in MODEL_FEATURE_COLUMNS:
                    val_float = float(val) if pd.notnull(val) and isinstance(val, (int, float)) else 0.0
                    score_float = float(score)
                    items.append((name, score_float, val_float))

            # Sort by absolute score contribution magnitude
            items.sort(key=lambda x: abs(x[1]), reverse=True)

            contributions = []
            for name, score, val in items[:top_k]:
                disp_name = FEATURE_DISPLAY_NAMES.get(name, name.replace('_', ' ').title())
                if score > 0.05:
                    direction = "INCREASES_RISK"
                    summary = f"{disp_name} (+{score:.2f}) elevates model risk assessment."
                elif score < -0.05:
                    direction = "DECREASES_RISK"
                    summary = f"{disp_name} ({score:.2f}) reduces model risk assessment."
                else:
                    direction = "NEUTRAL"
                    summary = f"{disp_name} aligns within baseline population norms."

                contributions.append(EBMFeatureContribution(
                    feature_name=name,
                    display_name=disp_name,
                    actual_value=round(val, 2),
                    score_contribution=round(score, 3),
                    direction=direction,
                    plain_summary=summary
                ))
            return contributions
        except Exception as e:
            logger.warning(f"EBM local explanation failed: {e}")
            return []

    def _build_behavioral_metrics(self, row: Dict[str, Any]) -> List[BehavioralMetric]:
        """Constructs rich behavioral metrics with clinical context and severity flags."""
        metrics = []

        # 1. Financial: Average Claim vs Peer
        avg_peer_ratio = float(row.get('average_claim_vs_peer_average', 1.0))
        avg_amt = float(row.get('average_claim_amount', 0.0))
        sev = "CRITICAL" if avg_peer_ratio > 3.0 else ("HIGH" if avg_peer_ratio > 2.0 else ("ELEVATED" if avg_peer_ratio > 1.5 else "NORMAL"))
        metrics.append(BehavioralMetric(
            metric_name="average_claim_vs_peer_average",
            display_name="Average Claim Amount vs State Peer",
            value=round(avg_amt, 2),
            peer_ratio=round(avg_peer_ratio, 2),
            severity=sev,
            plain_language_explanation=f"Provider average reimbursement is {avg_peer_ratio:.1f}x the state peer group average."
        ))

        # 2. Inpatient Ratio
        inp_ratio = float(row.get('inpatient_ratio', 0.0))
        sev = "HIGH" if inp_ratio > 0.40 else ("ELEVATED" if inp_ratio > 0.25 else "NORMAL")
        metrics.append(BehavioralMetric(
            metric_name="inpatient_ratio",
            display_name="Inpatient Admission Ratio",
            value=round(inp_ratio * 100.0, 1),
            peer_ratio=round(inp_ratio / 0.15, 2),
            severity=sev,
            plain_language_explanation=f"{inp_ratio*100.0:.1f}% of submitted claims are for inpatient hospitalizations."
        ))

        # 3. Repeat Beneficiary Concentration
        repeat_ratio = float(row.get('repeat_beneficiary_ratio', 0.0))
        sev = "CRITICAL" if repeat_ratio > 0.50 else ("HIGH" if repeat_ratio > 0.35 else ("ELEVATED" if repeat_ratio > 0.25 else "NORMAL"))
        metrics.append(BehavioralMetric(
            metric_name="repeat_beneficiary_ratio",
            display_name="Repeat Beneficiary Claim Ratio",
            value=round(repeat_ratio * 100.0, 1),
            peer_ratio=round(repeat_ratio / 0.18, 2),
            severity=sev,
            plain_language_explanation=f"{repeat_ratio*100.0:.1f}% of claims originate from returning patients."
        ))

        # 4. Attending / Operating Physician Match Ratio
        same_phys = float(row.get('same_attending_operating_ratio', 0.0))
        sev = "HIGH" if same_phys > 0.60 else ("ELEVATED" if same_phys > 0.40 else "NORMAL")
        metrics.append(BehavioralMetric(
            metric_name="same_attending_operating_ratio",
            display_name="Attending & Operating Physician Overlap",
            value=round(same_phys * 100.0, 1),
            severity=sev,
            plain_language_explanation=f"{same_phys*100.0:.1f}% of procedure claims list the attending physician as operating physician."
        ))

        # 5. Monthly Claim Velocity
        cpm = float(row.get('claims_per_month', 0.0))
        sev = "CRITICAL" if cpm > 100 else ("HIGH" if cpm > 60 else ("ELEVATED" if cpm > 35 else "NORMAL"))
        metrics.append(BehavioralMetric(
            metric_name="claims_per_month",
            display_name="Monthly Claim Submission Velocity",
            value=round(cpm, 1),
            severity=sev,
            plain_language_explanation=f"Provider submits an average of {cpm:.1f} claims per active calendar month."
        ))

        return metrics

    def _generate_key_findings(self, row: Dict[str, Any], prob: float, risk_score: int) -> List[str]:
        """Generates evidence-backed key findings in decision-support language."""
        findings = []
        peer_avg_ratio = float(row.get('average_claim_vs_peer_average', 1.0))
        peer_tot_ratio = float(row.get('claim_amount_vs_peer_average', 1.0))
        inp_ratio = float(row.get('inpatient_ratio', 0.0))
        repeat_ratio = float(row.get('repeat_beneficiary_ratio', 0.0))
        same_phys = float(row.get('same_attending_operating_ratio', 0.0))
        cpm = float(row.get('claims_per_month', 0.0))

        if peer_avg_ratio > 1.8:
            findings.append(f"Significant reimbursement deviation: average claim amount is {peer_avg_ratio:.1f}x higher than peer providers.")
        elif peer_tot_ratio > 2.5:
            findings.append(f"Elevated total billing volume: aggregate reimbursements exceed peer norm by {peer_tot_ratio:.1f}x.")

        if repeat_ratio > 0.35:
            findings.append(f"Patient concentration anomaly: {repeat_ratio*100.0:.1f}% of total claims stem from repeat beneficiaries.")

        if inp_ratio > 0.35:
            findings.append(f"Disproportionate inpatient utilization: {inp_ratio*100.0:.1f}% inpatient share compared to typical outpatient focus.")

        if same_phys > 0.50:
            findings.append(f"Physician assignment pattern: {same_phys*100.0:.1f}% co-occurrence of attending and operating roles.")

        if cpm > 75.0:
            findings.append(f"High temporal billing frequency: {cpm:.1f} claims filed per active month.")

        if not findings:
            findings.append("Provider billing metrics fall within standard population peer distributions.")

        return findings
