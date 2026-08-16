"""
Model Registry & Governance Service
-----------------------------------
Manages machine learning model versioning, deployment statuses, performance benchmarks,
and governance metadata in the database.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.config import DATABASE_PATH, MODELS_DIR
from src.database.connection import db_transaction
from src.services.audit_service import log_audit_event

logger = logging.getLogger(__name__)


class ModelRegistryService:
    """
    Service for registering, evaluating, activating, and auditing ML models.
    """

    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path

    def list_models(self) -> List[Dict[str, Any]]:
        """Returns all registered model versions."""
        with db_transaction(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM model_versions ORDER BY id DESC")
            return [dict(r) for r in cursor.fetchall()]

    def get_active_model(self, model_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieves currently active version of a given model type or any active model."""
        with db_transaction(self.db_path) as conn:
            if model_type:
                row = conn.execute(
                    "SELECT * FROM model_versions WHERE model_type = ? AND is_active = 1 LIMIT 1",
                    (model_type,)
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM model_versions WHERE is_active = 1 LIMIT 1"
                ).fetchone()
            return dict(row) if row else None

    def register_model(
        self,
        version_tag: str,
        model_type: str,
        file_path: str,
        training_dataset: str = "Medicare Provider Inpatient/Outpatient Claims",
        roc_auc: float = 0.0,
        pr_auc: float = 0.0,
        f1_score: float = 0.0,
        features_count: int = 52,
        is_active: bool = False,
        actor_username: str = "system"
    ) -> int:
        """Registers a new model version in the registry."""
        with db_transaction(self.db_path) as conn:
            if is_active:
                conn.execute(
                    "UPDATE model_versions SET is_active = 0 WHERE model_type = ?",
                    (model_type,)
                )

            cursor = conn.execute(
                """
                INSERT INTO model_versions (
                    version_tag, model_type, file_path, training_dataset,
                    features_count, roc_auc, pr_auc, f1_score, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version_tag, model_type, file_path, training_dataset,
                    features_count, roc_auc, pr_auc, f1_score, 1 if is_active else 0
                )
            )
            model_id = cursor.lastrowid

        log_audit_event(
            username=actor_username,
            role="ADMIN",
            action="REGISTER_MODEL_VERSION",
            entity_type="MODEL",
            entity_id=version_tag,
            status="SUCCESS",
            details={
                "model_id": model_id,
                "model_type": model_type,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "f1_score": f1_score,
                "is_active": is_active
            },
            db_path=self.db_path
        )
        return model_id

    def set_active_version(self, model_id: int, actor_username: str = "admin") -> bool:
        """Promotes a registered model version to active status."""
        with db_transaction(self.db_path) as conn:
            target = conn.execute(
                "SELECT version_tag, model_type FROM model_versions WHERE id = ?",
                (model_id,)
            ).fetchone()
            if not target:
                return False

            m_tag = target["version_tag"]
            m_type = target["model_type"]

            conn.execute("UPDATE model_versions SET is_active = 0 WHERE model_type = ?", (m_type,))
            conn.execute("UPDATE model_versions SET is_active = 1 WHERE id = ?", (model_id,))

        log_audit_event(
            username=actor_username,
            role="ADMIN",
            action="PROMOTE_ACTIVE_MODEL",
            entity_type="MODEL",
            entity_id=m_tag,
            status="SUCCESS",
            details={"model_id": model_id, "model_type": m_type},
            db_path=self.db_path
        )
        return True
