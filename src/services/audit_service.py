"""
Audit Trail Logging Service
---------------------------
Provides immutable, persistent audit logging for every major system action:
logins, logouts, dataset uploads, analysis executions, agent workflows,
investigation findings, management escalations, decisions, and governance changes.
"""

import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from src.config import DATABASE_PATH
from src.database.connection import db_transaction

logger = logging.getLogger(__name__)


def sanitize_audit_details(details: Optional[Dict[str, Any]]) -> str:
    """
    Sanitizes dictionary and serializes to JSON. Redacts password, secret keys, or SSN-like tokens.
    """
    if not details:
        return json.dumps({})
    
    sanitized = {}
    sensitive_keys = {"password", "secret", "token", "key", "api_key", "password_hash"}
    
    for k, v in details.items():
        if any(s in k.lower() for s in sensitive_keys):
            sanitized[k] = "[REDACTED]"
        else:
            sanitized[k] = v
            
    return json.dumps(sanitized, default=str)


def log_audit_event(
    username: str,
    role: str,
    action: str,
    entity_type: str,
    entity_id: Optional[str] = None,
    status: str = "SUCCESS",
    details: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
    db_path: str = DATABASE_PATH
) -> int:
    """
    Persists an audit event to the SQLite database.
    """
    status_upper = status.upper()
    if status_upper not in {"SUCCESS", "FAILURE", "WARNING", "UNAUTHORIZED"}:
        status_upper = "SUCCESS"
        
    details_json = sanitize_audit_details(details)
    
    try:
        with db_transaction(db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO audit_logs (username, role, action, entity_type, entity_id, status, details_json, ip_address)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    username.strip().lower() if username else "system",
                    role.upper() if role else "SYSTEM",
                    action.upper().strip(),
                    entity_type.upper().strip(),
                    str(entity_id) if entity_id is not None else None,
                    status_upper,
                    details_json,
                    ip_address or "127.0.0.1"
                )
            )
            return cursor.lastrowid
    except Exception as e:
        logger.error(f"Failed to write audit log event: {e}")
        return -1


def get_audit_logs(
    username: Optional[str] = None,
    role: Optional[str] = None,
    action: Optional[str] = None,
    entity_type: Optional[str] = None,
    entity_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db_path: str = DATABASE_PATH
) -> List[Dict[str, Any]]:
    """
    Retrieves filtered audit log records from the database.
    """
    query = "SELECT id, timestamp, username, role, action, entity_type, entity_id, status, details_json, ip_address FROM audit_logs WHERE 1=1"
    params = []
    
    if username:
        query += " AND username = ?"
        params.append(username.strip().lower())
    if role:
        query += " AND role = ?"
        params.append(role.upper().strip())
    if action:
        query += " AND action LIKE ?"
        params.append(f"%{action.upper().strip()}%")
    if entity_type:
        query += " AND entity_type = ?"
        params.append(entity_type.upper().strip())
    if entity_id:
        query += " AND entity_id = ?"
        params.append(str(entity_id).strip())
    if status:
        query += " AND status = ?"
        params.append(status.upper().strip())
        
    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    with db_transaction(db_path) as conn:
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d["details"] = json.loads(d["details_json"]) if d["details_json"] else {}
            except Exception:
                d["details"] = {}
            result.append(d)
        return result


def get_audit_summary_metrics(db_path: str = DATABASE_PATH) -> Dict[str, Any]:
    """
    Computes summary KPI statistics for system administrators.
    """
    with db_transaction(db_path) as conn:
        total_events = conn.execute("SELECT COUNT(*) FROM audit_logs").fetchone()[0]
        success_events = conn.execute("SELECT COUNT(*) FROM audit_logs WHERE status = 'SUCCESS'").fetchone()[0]
        failure_events = conn.execute("SELECT COUNT(*) FROM audit_logs WHERE status IN ('FAILURE', 'UNAUTHORIZED')").fetchone()[0]
        unique_users = conn.execute("SELECT COUNT(DISTINCT username) FROM audit_logs").fetchone()[0]
        
        recent_actions = conn.execute(
            "SELECT action, COUNT(*) as cnt FROM audit_logs GROUP BY action ORDER BY cnt DESC LIMIT 5"
        ).fetchall()
        
        return {
            "total_events": total_events,
            "success_events": success_events,
            "failure_events": failure_events,
            "success_rate_pct": round((success_events / total_events) * 100.0, 1) if total_events > 0 else 100.0,
            "unique_active_users": unique_users,
            "top_actions": [{ "action": r["action"], "count": r["cnt"] } for r in recent_actions]
        }
