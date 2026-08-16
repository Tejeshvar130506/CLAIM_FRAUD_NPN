"""
Fraud Analysis Agent Module
---------------------------
Second stage in the Multi-Agent Healthcare Provider Fraud Intelligence Platform.
Responsible for:
- Executing XGBoost model inference for fraud probability estimation
- Computing glass-box EBM additive feature contributions
- Synthesizing statistical peer benchmarks and behavioral metrics
- Producing standardized, audit-ready EvidencePackage contracts
- Persisting execution runs and audit logs in SQLite
"""

import time
import uuid
import logging
import pandas as pd
from typing import List, Dict, Any, Optional

from src.agents.contracts import EvidencePackage
from src.config import DATABASE_PATH, MODELS_DIR
from src.database.connection import db_transaction
from src.services.explainability_service import ExplainabilityService
from src.services.feature_service import FeatureEngineeringService, MODEL_FEATURE_COLUMNS
from src.services.audit_service import log_audit_event

logger = logging.getLogger(__name__)


class FraudAnalysisAgent:
    """
    Fraud Analysis Agent responsible for behavioral feature synthesis, risk modeling, and evidence packaging.
    """

    def __init__(self, db_path: str = DATABASE_PATH, models_dir: str = MODELS_DIR):
        self.db_path = db_path
        self.explainability_service = ExplainabilityService(models_dir=models_dir)
        self.feature_service = FeatureEngineeringService()

    def analyze_provider(
        self,
        provider_id: str,
        df_feature_row: pd.DataFrame,
        actor_username: str = "system"
    ) -> EvidencePackage:
        """
        Analyzes a single provider using a 1-row feature DataFrame and returns an EvidencePackage.
        """
        start_time = time.time()
        run_id = f"FA-{uuid.uuid4().hex[:8].upper()}"
        logger.info(f"[{run_id}] Fraud Analysis Agent evaluating Provider: {provider_id}")

        evidence_pkg = self.explainability_service.generate_evidence_package(
            provider_id=provider_id,
            df_feature_row=df_feature_row
        )

        exec_ms = int((time.time() - start_time) * 1000)
        self._record_agent_run(run_id, provider_id, evidence_pkg, exec_ms, actor_username)

        return evidence_pkg

    def analyze_from_dict(
        self,
        input_dict: Dict[str, Any],
        provider_id: str = "CUSTOM_PROV",
        actor_username: str = "user"
    ) -> EvidencePackage:
        """
        Analyzes a provider from interactive form dictionary inputs.
        """
        df_row = self.feature_service.build_feature_vector_from_dict(input_dict, provider_id=provider_id)
        return self.analyze_provider(provider_id=provider_id, df_feature_row=df_row, actor_username=actor_username)

    def batch_analyze_providers(
        self,
        df_features: pd.DataFrame,
        actor_username: str = "system"
    ) -> List[EvidencePackage]:
        """
        Batch analyzes a DataFrame of providers.
        """
        results = []
        logger.info(f"Fraud Analysis Agent batch evaluating {len(df_features):,} providers...")
        for idx, row in df_features.iterrows():
            prov_id = str(row.get('Provider', f'PROV_{idx}'))
            df_single = pd.DataFrame([row])
            pkg = self.explainability_service.generate_evidence_package(provider_id=prov_id, df_feature_row=df_single)
            results.append(pkg)

        return results

    def _record_agent_run(
        self,
        run_id: str,
        provider_id: str,
        evidence: EvidencePackage,
        exec_ms: int,
        username: str
    ) -> None:
        """Persists agent execution and audit logging."""
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
                        "FRAUD_ANALYSIS_AGENT",
                        "COMPLETED",
                        f"Evaluated Provider {provider_id} on 52 behavioral features",
                        evidence.model_dump_json(),
                        exec_ms
                    )
                )

            log_audit_event(
                username=username,
                role="SYSTEM",
                action="FRAUD_ANALYSIS_AGENT_RUN",
                entity_type="PROVIDER",
                entity_id=provider_id,
                status="SUCCESS",
                details={
                    "risk_score": evidence.risk_score,
                    "risk_level": evidence.risk_level,
                    "fraud_probability": evidence.fraud_probability,
                    "priority": evidence.investigation_priority
                },
                db_path=self.db_path
            )
        except Exception as e:
            logger.error(f"Failed to record Fraud Analysis Agent execution to database: {e}")
