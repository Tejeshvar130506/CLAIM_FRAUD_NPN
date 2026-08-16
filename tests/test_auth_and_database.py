"""
Unit & Integration Tests for Authentication, RBAC, SQLite WAL, and Audit Trail (Phase A)
"""

import os
import tempfile
import unittest
import sqlite3

from src.database.connection import get_db_connection, db_transaction, init_db
from src.auth.security import (
    hash_password, verify_password, create_user, get_user_by_username,
    verify_user_credentials, list_users, update_user_role, toggle_user_active, seed_default_users
)
from src.auth.rbac import (
    Role, Permission, has_permission, check_permission, get_role_permissions, get_role_dashboard_name
)
from src.services.audit_service import (
    log_audit_event, get_audit_logs, get_audit_summary_metrics
)


class TestAuthAndDatabase(unittest.TestCase):
    def setUp(self):
        # Create a clean temporary database for isolation
        self.temp_db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db_file.name
        self.temp_db_file.close()
        init_db(self.db_path)

    def tearDown(self):
        try:
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            # Remove WAL files if present
            for ext in ["-wal", "-shm"]:
                wal_file = self.db_path + ext
                if os.path.exists(wal_file):
                    os.remove(wal_file)
        except Exception:
            pass

    def test_database_wal_mode_and_tables(self):
        conn = get_db_connection(self.db_path)
        cursor = conn.execute("PRAGMA journal_mode;")
        mode = cursor.fetchone()[0]
        self.assertEqual(mode.lower(), "wal", "SQLite should operate in WAL mode.")
        
        # Verify required tables exist
        tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
        tables = [r[0] for r in conn.execute(tables_query).fetchall()]
        conn.close()
        
        required_tables = [
            "users", "providers", "analysis_runs", "investigations",
            "investigation_events", "agent_runs", "audit_logs",
            "model_versions", "watchlist", "alerts"
        ]
        for t in required_tables:
            self.assertIn(t, tables, f"Table '{t}' must exist in database schema.")

    def test_transaction_commit_and_rollback(self):
        # Test successful commit
        with db_transaction(self.db_path) as conn:
            conn.execute("INSERT INTO alerts (alert_type, severity, title, message) VALUES ('TEST', 'INFO', 'T1', 'M1')")
        
        with db_transaction(self.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM alerts WHERE title='T1'").fetchone()[0]
            self.assertEqual(count, 1)

        # Test rollback on exception
        try:
            with db_transaction(self.db_path) as conn:
                conn.execute("INSERT INTO alerts (alert_type, severity, title, message) VALUES ('TEST', 'INFO', 'T2', 'M2')")
                raise RuntimeError("Simulated failure")
        except RuntimeError:
            pass

        with db_transaction(self.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM alerts WHERE title='T2'").fetchone()[0]
            self.assertEqual(count, 0, "Transaction must rollback on exception.")

    def test_password_hashing_and_verification(self):
        password = "SecurePassword2026!"
        pwd_hash = hash_password(password)
        
        self.assertTrue(pwd_hash.startswith("pbkdf2:sha256:100000$"))
        self.assertTrue(verify_password(password, pwd_hash))
        self.assertFalse(verify_password("WrongPassword", pwd_hash))
        self.assertFalse(verify_password("", pwd_hash))

    def test_user_creation_and_authentication(self):
        user = create_user("testuser", "Pass123!", "INVESTIGATOR", "Test User", "test@hospital.org", db_path=self.db_path)
        self.assertEqual(user["username"], "testuser")
        self.assertEqual(user["role"], "INVESTIGATOR")

        # Verify authentication
        auth_success = verify_user_credentials("testuser", "Pass123!", db_path=self.db_path)
        self.assertIsNotNone(auth_success)
        self.assertEqual(auth_success["username"], "testuser")
        self.assertNotIn("password_hash", auth_success)

        auth_fail = verify_user_credentials("testuser", "WrongPass", db_path=self.db_path)
        self.assertIsNone(auth_fail)

    def test_user_management_functions(self):
        create_user("u1", "P1", "USER", "User One", "u1@org.com", db_path=self.db_path)
        create_user("u2", "P2", "MANAGER", "User Two", "u2@org.com", db_path=self.db_path)

        users = list_users(db_path=self.db_path)
        self.assertEqual(len(users), 2)

        # Update role
        update_user_role("u1", "ADMIN", db_path=self.db_path)
        updated_u1 = get_user_by_username("u1", db_path=self.db_path)
        self.assertEqual(updated_u1["role"], "ADMIN")

        # Toggle active state
        toggle_user_active("u1", db_path=self.db_path)
        inactive_auth = verify_user_credentials("u1", "P1", db_path=self.db_path)
        self.assertIsNone(inactive_auth, "Deactivated user must not authenticate.")

    def test_rbac_permissions_matrix(self):
        # User permissions
        self.assertTrue(has_permission("USER", Permission.UPLOAD_DATASET))
        self.assertFalse(has_permission("USER", Permission.VIEW_INVESTIGATION_QUEUE))
        self.assertFalse(has_permission("USER", Permission.MANAGE_USERS))

        # Investigator permissions
        self.assertTrue(has_permission("INVESTIGATOR", Permission.VIEW_INVESTIGATION_QUEUE))
        self.assertTrue(has_permission("INVESTIGATOR", Permission.RECORD_INVESTIGATION_FINDING))
        self.assertTrue(has_permission("INVESTIGATOR", Permission.ESCALATE_TO_MANAGEMENT))
        self.assertFalse(has_permission("INVESTIGATOR", Permission.RECORD_MANAGEMENT_DECISION))
        self.assertFalse(has_permission("INVESTIGATOR", Permission.MANAGE_USERS))

        # Manager permissions
        self.assertTrue(has_permission("MANAGER", Permission.VIEW_EXECUTIVE_DASHBOARD))
        self.assertTrue(has_permission("MANAGER", Permission.VIEW_ESCALATED_CASES))
        self.assertTrue(has_permission("MANAGER", Permission.RECORD_MANAGEMENT_DECISION))
        self.assertFalse(has_permission("MANAGER", Permission.MANAGE_USERS))

        # Admin permissions
        self.assertTrue(has_permission("ADMIN", Permission.MANAGE_USERS))
        self.assertTrue(has_permission("ADMIN", Permission.VIEW_AUDIT_LOGS))
        self.assertTrue(has_permission("ADMIN", Permission.RECORD_MANAGEMENT_DECISION))
        self.assertTrue(has_permission("ADMIN", Permission.RECORD_INVESTIGATION_FINDING))

        # Check permission exception
        with self.assertRaises(PermissionError):
            check_permission("USER", Permission.RECORD_MANAGEMENT_DECISION)

    def test_audit_logging_and_retrieval(self):
        log_id = log_audit_event(
            username="investigator1",
            role="INVESTIGATOR",
            action="RECORD_FINDING",
            entity_type="INVESTIGATION",
            entity_id="INV-1001",
            status="SUCCESS",
            details={"provider_id": "PRV51069", "finding": "Evidence validated"},
            db_path=self.db_path
        )
        self.assertGreater(log_id, 0)

        logs = get_audit_logs(username="investigator1", db_path=self.db_path)
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["action"], "RECORD_FINDING")
        self.assertEqual(logs[0]["details"]["provider_id"], "PRV51069")

        metrics = get_audit_summary_metrics(db_path=self.db_path)
        self.assertEqual(metrics["total_events"], 1)
        self.assertEqual(metrics["success_events"], 1)
        self.assertEqual(metrics["success_rate_pct"], 100.0)

    def test_seed_default_users(self):
        seed_default_users(self.db_path)
        users = list_users(db_path=self.db_path)
        self.assertEqual(len(users), 4)
        
        usernames = {u["username"] for u in users}
        self.assertEqual(usernames, {"admin", "manager", "investigator", "user"})

        admin = verify_user_credentials("admin", "Admin@2026!", db_path=self.db_path)
        self.assertIsNotNone(admin)
        self.assertEqual(admin["role"], "ADMIN")


if __name__ == "__main__":
    unittest.main()
