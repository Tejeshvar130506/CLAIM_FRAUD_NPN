"""
Database Connection & Lifecycle Manager
---------------------------------------
Provides high-performance, concurrency-safe SQLite connection handling with:
- WAL (Write-Ahead Logging) mode
- Foreign key constraints enforcement
- Busy timeouts (10 seconds)
- Short transaction lifetimes via context managers
- Automatic schema initialization
"""

import sqlite3
import logging
from contextlib import contextmanager
from typing import Generator
from pathlib import Path

from src.config import DATABASE_PATH, DATABASE_TIMEOUT_SECONDS, BASE_DIR

logger = logging.getLogger(__name__)


def get_db_connection(db_path: str = DATABASE_PATH) -> sqlite3.Connection:
    """
    Creates and configures a SQLite connection with WAL mode and foreign keys enabled.
    """
    conn = sqlite3.connect(
        db_path,
        timeout=DATABASE_TIMEOUT_SECONDS,
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
    )
    conn.row_factory = sqlite3.Row
    
    # Configure SQLite PRAGMAs for concurrency and data integrity
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute(f"PRAGMA busy_timeout = {DATABASE_TIMEOUT_SECONDS * 1000};")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA cache_size = -64000;")  # 64MB cache
    
    return conn


@contextmanager
def db_transaction(db_path: str = DATABASE_PATH) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager for short-lived, safe database transactions.
    Automatically commits on success or rolls back on exception.
    """
    conn = get_db_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database transaction rolled back due to error: {e}")
        raise
    finally:
        conn.close()


def init_db(db_path: str = DATABASE_PATH, schema_file: str = None) -> None:
    """
    Initializes database schema from schema.sql.
    """
    if schema_file is None:
        schema_file = str(Path(__file__).parent / "schema.sql")

    logger.info(f"Initializing SQLite database at: {db_path}")
    with open(schema_file, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    with db_transaction(db_path) as conn:
        conn.executescript(schema_sql)
    logger.info("Database schema initialized successfully.")


def table_exists(table_name: str, db_path: str = DATABASE_PATH) -> bool:
    """Checks whether a specific table exists in the database."""
    with db_transaction(db_path) as conn:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        return cursor.fetchone() is not None
