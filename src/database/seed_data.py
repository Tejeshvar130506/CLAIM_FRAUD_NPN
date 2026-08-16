"""
Database Seeding & Pre-population Utility
-----------------------------------------
Initializes SQLite schema, seeds default role accounts (admin, manager, investigator, user),
and pre-populates the providers and model_versions tables from existing engineered features
and trained model artifacts.
"""

import os
import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import DATABASE_PATH, FEATURES_DATA_DIR, MODELS_DIR, REPORTS_DIR
from src.database.connection import db_transaction, init_db
from src.auth.security import seed_default_users
from src.risk_scoring import calculate_risk_level

logger = logging.getLogger(__name__)


def seed_database(db_path: str = DATABASE_PATH) -> None:
    """
    Complete initialization and seeding pipeline.
    """
    logger.info("--> Initializing SQLite database schema...")
    init_db(db_path)
    
    logger.info("--> Seeding default RBAC user accounts...")
    seed_default_users(db_path)
    
    logger.info("--> Seeding model versions in registry...")
    _seed_model_versions(db_path)
    
    logger.info("--> Seeding provider risk profiles...")
    _seed_providers(db_path)
    
    logger.info("--> Database initialization & seeding completed.")


def _seed_model_versions(db_path: str) -> None:
    """Seeds active model version metadata into model_versions table."""
    xgb_path = os.path.join(MODELS_DIR, "final_fraud_model.pkl")
    ebm_path = os.path.join(MODELS_DIR, "ebm_model.pkl")
    
    with db_transaction(db_path) as conn:
        # Check if already seeded
        cnt = conn.execute("SELECT COUNT(*) FROM model_versions").fetchone()[0]
        if cnt > 0:
            return
        
        if os.path.exists(xgb_path):
            conn.execute(
                """
                INSERT INTO model_versions 
                (version_tag, model_type, file_path, training_dataset, features_count, roc_auc, pr_auc, f1_score, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                ("v2.0-xgb-production", "XGBoost Classifier", xgb_path, "Train-1542865627584.csv", 41, 0.9692, 0.8407, 0.7297)
            )
            
        if os.path.exists(ebm_path):
            conn.execute(
                """
                INSERT INTO model_versions 
                (version_tag, model_type, file_path, training_dataset, features_count, roc_auc, pr_auc, f1_score, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                ("v2.0-ebm-glassbox", "Explainable Boosting Machine (EBM)", ebm_path, "Train-1542865627584.csv", 41, 0.9410, 0.7850, 0.6950)
            )


def _seed_providers(db_path: str) -> None:
    """Seeds provider behavioral records from features parquet files."""
    train_feat_path = os.path.join(FEATURES_DATA_DIR, "train_provider_features.parquet")
    test_feat_path = os.path.join(FEATURES_DATA_DIR, "test_provider_features.parquet")
    xgb_path = os.path.join(MODELS_DIR, "final_fraud_model.pkl")
    
    if not (os.path.exists(train_feat_path) and os.path.exists(xgb_path)):
        logger.warning("Feature or model files not found. Skipping provider seeding.")
        return

    with db_transaction(db_path) as conn:
        cnt = conn.execute("SELECT COUNT(*) FROM providers").fetchone()[0]
        if cnt > 0:
            logger.info(f"Providers table already contains {cnt:,} records.")
            return

    logger.info("Loading feature files and model for provider database pre-population...")
    df_train = pd.read_parquet(train_feat_path)
    df_test = pd.read_parquet(test_feat_path) if os.path.exists(test_feat_path) else pd.DataFrame()
    
    if 'PotentialFraud' in df_train.columns:
        df_train = df_train.drop(columns=['PotentialFraud'])
        
    df_combined = pd.concat([df_train, df_test], ignore_index=True).drop_duplicates(subset=['Provider'])
    
    model = joblib.load(xgb_path)
    feature_cols = [c for c in df_combined.columns if c != 'Provider']
    
    X = df_combined[feature_cols].fillna(0.0)
    probs = model.predict_proba(X)[:, 1]
    
    rows_to_insert = []
    for idx, row in df_combined.iterrows():
        prob = float(probs[idx])
        risk_score = int(round(prob * 100.0))
        risk_level = calculate_risk_level(risk_score)
        
        # Determine investigation priority
        if risk_score >= 85:
            priority = "CRITICAL"
        elif risk_score >= 60:
            priority = "HIGH"
        elif risk_score >= 30:
            priority = "NORMAL"
        else:
            priority = "LOW"
            
        rows_to_insert.append((
            str(row['Provider']),
            str(row.get('primary_state', '0')),
            int(row.get('total_claims', 0)),
            float(row.get('total_claim_amount', 0.0)),
            float(row.get('average_claim_amount', 0.0)),
            float(row.get('inpatient_ratio', 0.0)),
            float(row.get('repeat_beneficiary_ratio', 0.0)),
            float(row.get('average_claim_vs_peer_average', 1.0)),
            float(row.get('claim_amount_vs_peer_average', 1.0)),
            float(row.get('beneficiary_hhi_concentration', 0.0)),
            float(row.get('same_attending_operating_ratio', 0.0)),
            float(row.get('claims_per_month', 0.0)),
            round(prob, 4),
            risk_score,
            risk_level,
            priority,
            'ACTIVE'
        ))
        
    logger.info(f"Writing {len(rows_to_insert):,} provider records to SQLite...")
    with db_transaction(db_path) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO providers (
                provider_id, primary_state, total_claims, total_claim_amount, average_claim_amount,
                inpatient_ratio, repeat_beneficiary_ratio, average_claim_vs_peer_average, claim_amount_vs_peer_average,
                beneficiary_hhi_concentration, same_attending_operating_ratio, claims_per_month,
                fraud_probability, risk_score, risk_level, investigation_priority, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows_to_insert
        )
    logger.info(f"Successfully seeded {len(rows_to_insert):,} providers in database.")


if __name__ == "__main__":
    seed_database()
