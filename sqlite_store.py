"""
storage/sqlite_store.py — Full-text archive for the Orbital Memory System.

Stores the original, uncompressed text of each memory orb.
Each row is linked to its ChromaDB counterpart via the shared orb_id (UUID).

This module knows nothing about vectors, embeddings, or physics —
it is a pure key-value store: orb_id → full_text.

Schema:
    full_texts
        orb_id     TEXT PRIMARY KEY   — UUID matching ChromaDB entry
        full_text  TEXT NOT NULL      — original content
        created_at TEXT NOT NULL      — ISO 8601 UTC timestamp
"""

import sqlite3
from datetime import datetime

import config


def _get_connection(db_path: str = config.SQLITE_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with row_factory for named column access."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_db(db_path: str = config.SQLITE_DB_PATH) -> None:
    """
    Create the full_texts table if it does not already exist.
    Safe to call on every startup — uses CREATE TABLE IF NOT EXISTS.
    """
    conn = _get_connection(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS full_texts (
                orb_id     TEXT PRIMARY KEY,
                full_text  TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def save_full_text(orb_id: str, full_text: str, db_path: str = config.SQLITE_DB_PATH) -> None:
    """
    Persist the original full text for an orb.

    Args:
        orb_id:    UUID that links this row to its ChromaDB entry.
        full_text: The original, uncompressed memory content.
        db_path:   Path to the SQLite database file.
    """
    conn = _get_connection(db_path)
    try:
        conn.execute(
            "INSERT INTO full_texts (orb_id, full_text, created_at) VALUES (?, ?, ?)",
            (orb_id, full_text, datetime.utcnow().isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def get_full_text(orb_id: str, db_path: str = config.SQLITE_DB_PATH) -> str | None:
    """
    Retrieve the original full text for a given orb_id.

    Args:
        orb_id:  UUID of the memory orb.
        db_path: Path to the SQLite database file.

    Returns:
        The full text string, or None if orb_id is not found.
    """
    conn = _get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT full_text FROM full_texts WHERE orb_id = ?", (orb_id,)
        ).fetchone()
        return row["full_text"] if row else None
    finally:
        conn.close()
