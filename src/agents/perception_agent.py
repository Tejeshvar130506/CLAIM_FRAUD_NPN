"""
Perception Agent Module
-----------------------
First stage in the Multi-Agent Healthcare Provider Fraud Intelligence Platform.
Responsible for:
- Automated file & dataset discovery
- Schema detection & data type verification
- Missing value profiling & duplicate detection
- Entity identifier integrity verification (Provider, Beneficiary, Claim)
- Relational mapping & join safety verification
- Comprehensive Data Quality Scoring (0 - 100)
- Execution logging and auditability
"""

import os
import glob
import time
import uuid
import json
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

from src.agents.contracts import (
    ColumnProfile, FilePerceptionSummary, DataQualityReport, PerceptionResult
)
from src.config import DATABASE_PATH, RAW_DATA_DIR
from src.database.connection import db_transaction
from src.services.audit_service import log_audit_event

logger = logging.getLogger(__name__)


class PerceptionAgent:
    """
    Perception Agent for structural dataset understanding, validation, and data quality profiling.
    """

    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path

    def analyze_dataset_directory(
        self,
        directory_path: str,
        group_name: str = "TRAIN",
        actor_username: str = "system"
    ) -> PerceptionResult:
        """
        Profiles a directory of raw CSV or Parquet files.
        """
        start_time = time.time()
        run_id = f"PERC-{uuid.uuid4().hex[:8].upper()}"
        logger.info(f"[{run_id}] Perception Agent analyzing directory: {directory_path} (Group: {group_name})")

        files = sorted(glob.glob(os.path.join(directory_path, "*.csv")))
        if not files:
            files = sorted(glob.glob(os.path.join(directory_path, "*.parquet")))

        if not files:
            return self._build_empty_result(run_id, group_name, f"No CSV or Parquet files found in {directory_path}")

        file_summaries: List[FilePerceptionSummary] = []
        total_records = 0
        total_providers = 0
        total_claims = 0
        total_benes = 0
        warnings = []
        anomalies = []
        dedup_keys_found = 0

        has_provider_key = False
        has_bene_key = False
        has_claim_key = False

        for fpath in files:
            fname = os.path.basename(fpath)
            size_mb = os.path.getsize(fpath) / (1024 * 1024)

            try:
                if fpath.endswith(".csv"):
                    df = pd.read_csv(fpath, low_memory=False)
                else:
                    df = pd.read_parquet(fpath)
            except Exception as e:
                warnings.append(f"Failed to read file {fname}: {str(e)}")
                continue

            row_cnt, col_cnt = df.shape
            total_records += row_cnt
            dup_rows = int(df.duplicated().sum())
            if dup_rows > 0:
                dedup_keys_found += dup_rows
                warnings.append(f"File '{fname}' contains {dup_rows:,} exact duplicate rows.")

            # Check keys
            if "Provider" in df.columns:
                has_provider_key = True
                total_providers = max(total_providers, df["Provider"].nunique())
            if "BeneID" in df.columns:
                has_bene_key = True
                total_benes = max(total_benes, df["BeneID"].nunique())
            if "ClaimID" in df.columns:
                has_claim_key = True
                total_claims += df["ClaimID"].nunique()

            col_profiles: List[ColumnProfile] = []
            for col in df.columns:
                null_cnt = int(df[col].isnull().sum())
                null_pct = round((null_cnt / row_cnt) * 100.0, 2) if row_cnt > 0 else 0.0
                uniq_cnt = int(df[col].nunique(dropna=True))
                sample_vals = [str(x) for x in df[col].dropna().unique()[:3]]

                # Identify role
                if col in ["Provider", "BeneID", "ClaimID"]:
                    role = "Primary / Foreign Key"
                elif col == "PotentialFraud":
                    role = "Fraud Risk Label"
                elif "Amt" in col or "Reimbursement" in col or "Deductible" in col:
                    role = "Financial Metric"
                elif "Dt" in col or "DOB" in col or "DOD" in col:
                    role = "Temporal / Date"
                elif "Diagnosis" in col or "Procedure" in col:
                    role = "Clinical Code"
                else:
                    role = "Attribute Feature"

                col_profiles.append(ColumnProfile(
                    name=col,
                    dtype=str(df[col].dtype),
                    role=role,
                    missing_count=null_cnt,
                    missing_pct=null_pct,
                    unique_count=uniq_cnt,
                    sample_values=sample_vals
                ))

            file_summaries.append(FilePerceptionSummary(
                filename=fname,
                group=group_name,
                row_count=row_cnt,
                col_count=col_cnt,
                duplicate_rows=dup_rows,
                size_mb=round(size_mb, 2),
                columns=col_profiles
            ))

        # Compute Data Quality Score (0 to 100)
        quality_score = 100.0
        if not has_provider_key:
            quality_score -= 30.0
            warnings.append("Provider identifier column ('Provider') missing from dataset files.")
        if not has_bene_key:
            quality_score -= 15.0
            warnings.append("Beneficiary identifier ('BeneID') not detected.")
        if not has_claim_key:
            quality_score -= 15.0
            warnings.append("Claim identifier ('ClaimID') not detected.")
        if dedup_keys_found > 0:
            quality_score -= min(15.0, (dedup_keys_found / max(1, total_records)) * 100.0)

        quality_score = max(10.0, round(quality_score, 1))
        key_status = "VALIDATED" if (has_provider_key and has_bene_key and has_claim_key) else "WARNING"

        exec_ms = int((time.time() - start_time) * 1000)

        quality_report = DataQualityReport(
            overall_quality_score=quality_score,
            total_files_analyzed=len(file_summaries),
            total_records=total_records,
            key_integrity_status=key_status,
            provider_key_present=has_provider_key,
            beneficiary_key_present=has_bene_key,
            claim_key_present=has_claim_key,
            duplicate_keys_found=dedup_keys_found,
            warnings=warnings,
            anomalies_detected=anomalies
        )

        result = PerceptionResult(
            run_id=run_id,
            status="COMPLETED",
            dataset_group=group_name,
            files_profiled=file_summaries,
            total_providers_detected=total_providers,
            total_claims_detected=total_claims,
            total_beneficiaries_detected=total_benes,
            quality_report=quality_report,
            execution_time_ms=exec_ms
        )

        self._record_agent_run(result, actor_username)
        return result

    def analyze_uploaded_dataframes(
        self,
        dfs_dict: Dict[str, pd.DataFrame],
        group_name: str = "UPLOADED_BATCH",
        actor_username: str = "user"
    ) -> PerceptionResult:
        """
        Profiles in-memory dataframes (e.g., from Streamlit file upload widgets).
        """
        start_time = time.time()
        run_id = f"PERC-{uuid.uuid4().hex[:8].upper()}"

        file_summaries: List[FilePerceptionSummary] = []
        total_records = 0
        total_providers = 0
        total_claims = 0
        total_benes = 0
        warnings = []
        anomalies = []
        dedup_keys_found = 0

        has_provider_key = False
        has_bene_key = False
        has_claim_key = False

        for fname, df in dfs_dict.items():
            row_cnt, col_cnt = df.shape
            total_records += row_cnt
            dup_rows = int(df.duplicated().sum())
            if dup_rows > 0:
                dedup_keys_found += dup_rows
                warnings.append(f"Uploaded file '{fname}' has {dup_rows} duplicate rows.")

            if "Provider" in df.columns:
                has_provider_key = True
                total_providers = max(total_providers, df["Provider"].nunique())
            if "BeneID" in df.columns:
                has_bene_key = True
                total_benes = max(total_benes, df["BeneID"].nunique())
            if "ClaimID" in df.columns:
                has_claim_key = True
                total_claims += df["ClaimID"].nunique()

            col_profiles = []
            for col in df.columns:
                null_cnt = int(df[col].isnull().sum())
                null_pct = round((null_cnt / row_cnt) * 100.0, 2) if row_cnt > 0 else 0.0
                uniq_cnt = int(df[col].nunique(dropna=True))
                sample_vals = [str(x) for x in df[col].dropna().unique()[:3]]

                col_profiles.append(ColumnProfile(
                    name=col,
                    dtype=str(df[col].dtype),
                    role="Attribute",
                    missing_count=null_cnt,
                    missing_pct=null_pct,
                    unique_count=uniq_cnt,
                    sample_values=sample_vals
                ))

            file_summaries.append(FilePerceptionSummary(
                filename=fname,
                group=group_name,
                row_count=row_cnt,
                col_count=col_cnt,
                duplicate_rows=dup_rows,
                size_mb=round((df.memory_usage(deep=True).sum()) / (1024 * 1024), 2),
                columns=col_profiles
            ))

        quality_score = 100.0
        if not has_provider_key:
            quality_score -= 30.0
            warnings.append("Provider ID key column missing.")
        if dedup_keys_found > 0:
            quality_score -= 10.0

        quality_score = max(10.0, round(quality_score, 1))
        key_status = "VALIDATED" if has_provider_key else "WARNING"
        exec_ms = int((time.time() - start_time) * 1000)

        quality_report = DataQualityReport(
            overall_quality_score=quality_score,
            total_files_analyzed=len(file_summaries),
            total_records=total_records,
            key_integrity_status=key_status,
            provider_key_present=has_provider_key,
            beneficiary_key_present=has_bene_key,
            claim_key_present=has_claim_key,
            duplicate_keys_found=dedup_keys_found,
            warnings=warnings,
            anomalies_detected=anomalies
        )

        result = PerceptionResult(
            run_id=run_id,
            status="COMPLETED",
            dataset_group=group_name,
            files_profiled=file_summaries,
            total_providers_detected=total_providers,
            total_claims_detected=total_claims,
            total_beneficiaries_detected=total_benes,
            quality_report=quality_report,
            execution_time_ms=exec_ms
        )

        self._record_agent_run(result, actor_username)
        return result

    def _build_empty_result(self, run_id: str, group_name: str, warning_msg: str) -> PerceptionResult:
        return PerceptionResult(
            run_id=run_id,
            status="FAILED",
            dataset_group=group_name,
            files_profiled=[],
            total_providers_detected=0,
            total_claims_detected=0,
            total_beneficiaries_detected=0,
            quality_report=DataQualityReport(
                overall_quality_score=0.0,
                total_files_analyzed=0,
                total_records=0,
                key_integrity_status="FAILED",
                provider_key_present=False,
                beneficiary_key_present=False,
                claim_key_present=False,
                duplicate_keys_found=0,
                warnings=[warning_msg],
                anomalies_detected=[]
            ),
            execution_time_ms=0
        )

    def _record_agent_run(self, result: PerceptionResult, username: str) -> None:
        """Persists agent execution record and audit log."""
        try:
            with db_transaction(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO agent_runs (run_id, provider_id, agent_name, status, input_summary, output_json, execution_time_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result.run_id,
                        None,
                        "PERCEPTION_AGENT",
                        result.status,
                        f"Group: {result.dataset_group}, Files: {len(result.files_profiled)}, Records: {result.quality_report.total_records:,}",
                        result.model_dump_json(),
                        result.execution_time_ms
                    )
                )

            log_audit_event(
                username=username,
                role="USER",
                action="PERCEPTION_AGENT_RUN",
                entity_type="DATASET",
                entity_id=result.run_id,
                status="SUCCESS" if result.status == "COMPLETED" else "WARNING",
                details={
                    "files_count": len(result.files_profiled),
                    "total_records": result.quality_report.total_records,
                    "quality_score": result.quality_report.overall_quality_score
                },
                db_path=self.db_path
            )
        except Exception as e:
            logger.error(f"Failed to record Perception Agent run to database: {e}")
