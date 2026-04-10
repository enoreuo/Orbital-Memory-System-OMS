"""
storage.py — Persistence layer for the Orbital Memory System.

Handles all SQLite interactions. This module knows nothing about embeddings
or physics math — it only serializes/deserializes MemoryOrb objects to/from
the database.

Schema (orbs table):
    id             INTEGER PRIMARY KEY AUTOINCREMENT
    content        TEXT NOT NULL
    vector_blob    BLOB NOT NULL       — pickled numpy array
    orbital_radius REAL NOT NULL
    last_accessed  TEXT NOT NULL       — ISO 8601 UTC string
    metadata_blob  BLOB               — pickled dict (may be NULL)
"""

import pickle
import sqlite3
from datetime import datetime
from typing import List

import numpy as np

from engine import MemoryOrb

DB_PATH = "oms.db"


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Open and return a SQLite connection with row_factory set to sqlite3.Row.
    sqlite3.Row allows column access by name (row["content"]) instead of index.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_db(db_path: str = DB_PATH) -> None:
    """
    Create the 'orbs' table if it does not already exist.
    Safe to call on every startup — uses CREATE TABLE IF NOT EXISTS.
    """
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS orbs (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                content        TEXT    NOT NULL,
                vector_blob    BLOB    NOT NULL,
                orbital_radius REAL    NOT NULL,
                last_accessed  TEXT    NOT NULL,
                metadata_blob  BLOB
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def save_orb(orb: MemoryOrb, db_path: str = DB_PATH) -> int:
    """
    Insert a new MemoryOrb into the database.

    Serializes the numpy vector and metadata dict as BLOBs via pickle.
    Stores last_accessed as an ISO 8601 string for human readability.

    Args:
        orb:     The MemoryOrb to persist. Its id field will be set after insert.
        db_path: Path to the SQLite database file.

    Returns:
        The assigned row id (INTEGER PRIMARY KEY).
    """
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO orbs (content, vector_blob, orbital_radius, last_accessed, metadata_blob)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                orb.content,
                pickle.dumps(orb.vector),
                orb.orbital_radius,
                orb.last_accessed.isoformat(),
                pickle.dumps(orb.metadata) if orb.metadata else None,
            ),
        )
        conn.commit()
        orb.id = cursor.lastrowid
        return cursor.lastrowid
    finally:
        conn.close()


def load_all_orbs(db_path: str = DB_PATH) -> List[MemoryOrb]:
    """
    Load every MemoryOrb from the database.

    Deserializes vector_blob and metadata_blob back to numpy arrays and dicts.
    Parses last_accessed from ISO 8601 string back to datetime.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        List of MemoryOrb instances with id fields populated.
    """
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT * FROM orbs").fetchall()
    finally:
        conn.close()

    orbs = []
    for row in rows:
        vector: np.ndarray = pickle.loads(row["vector_blob"])
        metadata: dict = pickle.loads(row["metadata_blob"]) if row["metadata_blob"] else {}
        last_accessed: datetime = datetime.fromisoformat(row["last_accessed"])

        orb = MemoryOrb(
            content=row["content"],
            vector=vector,
            orbital_radius=row["orbital_radius"],
            last_accessed=last_accessed,
            metadata=metadata,
            id=row["id"],
        )
        orbs.append(orb)

    return orbs


def update_orb(orb: MemoryOrb, db_path: str = DB_PATH) -> None:
    """
    Update the orbital_radius and last_accessed fields for an existing orb.

    Called after momentum is applied during a search hit.
    Only these two fields change after creation — content and vector are immutable.

    Args:
        orb:     The MemoryOrb with updated orbital_radius and last_accessed.
                 Must have a non-None id field.
        db_path: Path to the SQLite database file.

    Raises:
        ValueError: If orb.id is None (orb has not been persisted yet).
    """
    if orb.id is None:
        raise ValueError("Cannot update an orb that has no database id (not yet saved).")

    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            UPDATE orbs
            SET orbital_radius = ?, last_accessed = ?
            WHERE id = ?
            """,
            (orb.orbital_radius, orb.last_accessed.isoformat(), orb.id),
        )
        conn.commit()
    finally:
        conn.close()
