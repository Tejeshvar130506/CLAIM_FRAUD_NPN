"""
Authentication & Cryptographic Security Module
-----------------------------------------------
Provides secure password hashing (PBKDF2-HMAC-SHA256 with cryptographic salt),
constant-time verification, and user management operations backed by SQLite.
"""

import hashlib
import hmac
import os
import secrets
from typing import Optional, Dict, Any, List
from datetime import datetime

from src.config import PASSWORD_SALT, DATABASE_PATH
from src.database.connection import db_transaction, init_db


def hash_password(password: str, salt: Optional[str] = None) -> str:
    """
    Hashes a password using PBKDF2-HMAC-SHA256 with 100,000 iterations.
    Format: 'pbkdf2:sha256:100000$<salt>$<hex_hash>'
    """
    if salt is None:
        salt = secrets.token_hex(16)
    
    combined_salt = f"{PASSWORD_SALT}:{salt}".encode("utf-8")
    pwd_bytes = password.encode("utf-8")
    
    key = hashlib.pbkdf2_hmac("sha256", pwd_bytes, combined_salt, 100000)
    return f"pbkdf2:sha256:100000${salt}${key.hex()}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifies a plain-text password against a PBKDF2 hash using constant-time comparison.
    """
    try:
        algorithm, sub_algo, iterations_str, salt, original_hash = hashed_password.split("$")
        iterations = int(iterations_str)
        combined_salt = f"{PASSWORD_SALT}:{salt}".encode("utf-8")
        pwd_bytes = plain_password.encode("utf-8")
        
        computed_key = hashlib.pbkdf2_hmac("sha256", pwd_bytes, combined_salt, iterations)
        return hmac.compare_digest(computed_key.hex(), original_hash)
    except Exception:
        # Fallback for simple legacy format if present
        try:
            parts = hashed_password.split("$")
            if len(parts) == 3:
                salt = parts[1]
                orig = parts[2]
                combined_salt = f"{PASSWORD_SALT}:{salt}".encode("utf-8")
                key = hashlib.pbkdf2_hmac("sha256", plain_password.encode("utf-8"), combined_salt, 100000)
                return hmac.compare_digest(key.hex(), orig)
        except Exception:
            pass
        return False


def create_user(
    username: str,
    password: str,
    role: str,
    full_name: str,
    email: str,
    db_path: str = DATABASE_PATH
) -> Dict[str, Any]:
    """Creates a new user in the database with secure hashed password."""
    role = role.upper()
    valid_roles = {"USER", "INVESTIGATOR", "MANAGER", "ADMIN"}
    if role not in valid_roles:
        raise ValueError(f"Invalid role '{role}'. Must be one of: {valid_roles}")

    pwd_hash = hash_password(password)
    
    with db_transaction(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO users (username, password_hash, role, full_name, email, is_active)
            VALUES (?, ?, ?, ?, ?, 1)
            """,
            (username.strip().lower(), pwd_hash, role, full_name.strip(), email.strip().lower())
        )
        user_id = cursor.lastrowid

    return {
        "id": user_id,
        "username": username.strip().lower(),
        "role": role,
        "full_name": full_name,
        "email": email
    }


def get_user_by_username(username: str, db_path: str = DATABASE_PATH) -> Optional[Dict[str, Any]]:
    """Fetches a user dictionary by username."""
    with db_transaction(db_path) as conn:
        cursor = conn.execute(
            "SELECT id, username, password_hash, role, full_name, email, is_active, created_at, last_login FROM users WHERE username = ?",
            (username.strip().lower(),)
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def verify_user_credentials(username: str, password: str, db_path: str = DATABASE_PATH) -> Optional[Dict[str, Any]]:
    """
    Verifies user credentials. If valid and user is active, updates last_login and returns user profile.
    """
    user = get_user_by_username(username, db_path=db_path)
    if not user:
        return None
    
    if not user.get("is_active", 1):
        return None
    
    if verify_password(password, user["password_hash"]):
        # Update last_login
        with db_transaction(db_path) as conn:
            conn.execute(
                "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?",
                (user["id"],)
            )
        # Exclude password hash from returned profile
        user_copy = user.copy()
        user_copy.pop("password_hash", None)
        return user_copy
    
    return None


def list_users(db_path: str = DATABASE_PATH) -> List[Dict[str, Any]]:
    """Returns a list of all user profiles (without password hashes)."""
    with db_transaction(db_path) as conn:
        cursor = conn.execute(
            "SELECT id, username, role, full_name, email, is_active, created_at, last_login FROM users ORDER BY id ASC"
        )
        return [dict(row) for row in cursor.fetchall()]


def update_user_role(username: str, new_role: str, db_path: str = DATABASE_PATH) -> bool:
    """Updates a user's role."""
    new_role = new_role.upper()
    if new_role not in {"USER", "INVESTIGATOR", "MANAGER", "ADMIN"}:
        raise ValueError(f"Invalid role: {new_role}")
    
    with db_transaction(db_path) as conn:
        cursor = conn.execute(
            "UPDATE users SET role = ? WHERE username = ?",
            (new_role, username.strip().lower())
        )
        return cursor.rowcount > 0


def toggle_user_active(username: str, db_path: str = DATABASE_PATH) -> bool:
    """Toggles user active state (activate/deactivate)."""
    with db_transaction(db_path) as conn:
        cursor = conn.execute(
            "UPDATE users SET is_active = CASE WHEN is_active = 1 THEN 0 ELSE 1 END WHERE username = ?",
            (username.strip().lower(),)
        )
        return cursor.rowcount > 0


def seed_default_users(db_path: str = DATABASE_PATH) -> None:
    """
    Seeds initial system accounts for each of the 4 roles if not already present.
    """
    init_db(db_path)
    
    default_accounts = [
        ("admin", "Admin@2026!", "ADMIN", "System Administrator", "admin@healthcare-audit.gov"),
        ("manager", "Manager@2026!", "MANAGER", "Special Investigations Unit Manager", "manager@healthcare-audit.gov"),
        ("investigator", "Investigator@2026!", "INVESTIGATOR", "Senior Clinical Fraud Investigator", "investigator@healthcare-audit.gov"),
        ("user", "User@2026!", "USER", "Claims Analyst / Data Contributor", "analyst@healthcare-audit.gov")
    ]
    
    for uname, pwd, role, full_name, email in default_accounts:
        existing = get_user_by_username(uname, db_path=db_path)
        if not existing:
            create_user(uname, pwd, role, full_name, email, db_path=db_path)
