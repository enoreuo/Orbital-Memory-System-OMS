"""
storage/chroma_store.py — Vector search layer for the Orbital Memory System.

Wraps ChromaDB to store and retrieve memory orb embeddings + orbital metadata.

Each orb entry in ChromaDB contains:
    id:        orb_id (UUID string)
    embedding: 384-dim summary_vector (cosine space)
    document:  LLM-generated summary text
    metadata: {
        orbital_radius: float   — current distance from center
        last_accessed:  str     — ISO 8601 UTC timestamp
        decay_rate:     float   — per-orb decay constant
        full_text_id:   str     — same UUID, pointer to SQLite row
    }

Note on cosine distance:
    ChromaDB returns cosine *distance* in [0, 2] (0 = identical).
    To get similarity: sim = 1.0 - distance
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Tuple, Dict

import numpy as np
import chromadb

import config

# ---------------------------------------------------------------------------
# Singleton collection
# ---------------------------------------------------------------------------

_client = None
_collection = None


def _get_collection():
    """
    Lazy-initialize the ChromaDB persistent client and collection.
    Reuses the singleton on subsequent calls.
    """
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        _collection = _client.get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------

def add_orb(
    orb_id: str,
    summary: str,
    summary_vector: np.ndarray,
    orbital_radius: float,
    last_accessed: datetime,
    full_text_id: str,
) -> None:
    """
    Add a new memory orb to ChromaDB.

    Args:
        orb_id:         UUID for this orb (shared with SQLite).
        summary:        LLM-generated semantic summary.
        summary_vector: Embedding of the summary (384-dim float32).
        orbital_radius: Starting radius (1.0 for new orbs).
        last_accessed:  Creation timestamp.
        full_text_id:   UUID pointing to SQLite full_texts row (same as orb_id).
    """
    collection = _get_collection()
    collection.add(
        ids=[orb_id],
        embeddings=[summary_vector.tolist()],
        documents=[summary],
        metadatas=[{
            "orbital_radius": orbital_radius,
            "last_accessed":  last_accessed.isoformat(),
            "decay_rate":     config.DECAY_RATE,
            "full_text_id":   full_text_id,
        }],
    )


def update_radius(orb_id: str, new_radius: float, last_accessed: datetime) -> None:
    """
    Update orbital_radius and last_accessed for a single orb (momentum update).

    ChromaDB requires a full metadata upsert — existing fields are preserved
    by fetching first, then updating only the two changed fields.

    Args:
        orb_id:       UUID of the orb to update.
        new_radius:   Post-momentum radius value.
        last_accessed: Timestamp of the retrieval event.
    """
    collection = _get_collection()
    result = collection.get(ids=[orb_id], include=["metadatas"])
    if not result["ids"]:
        return

    metadata = result["metadatas"][0]
    metadata["orbital_radius"] = new_radius
    metadata["last_accessed"]  = last_accessed.isoformat()

    collection.update(ids=[orb_id], metadatas=[metadata])


def batch_update_radii(updates: List[Tuple[str, float]]) -> None:
    """
    Batch-update orbital radii for multiple orbs (used by decay worker).

    Only updates orbital_radius — does NOT touch last_accessed.
    last_accessed is only changed by momentum (actual retrieval events).

    Args:
        updates: List of (orb_id, new_radius) tuples.
    """
    if not updates:
        return

    collection = _get_collection()
    ids = [u[0] for u in updates]

    result = collection.get(ids=ids, include=["metadatas"])
    if not result["ids"]:
        return

    # Build a lookup for the new radii
    radius_map = {orb_id: radius for orb_id, radius in updates}

    new_metadatas = []
    for orb_id, metadata in zip(result["ids"], result["metadatas"]):
        metadata["orbital_radius"] = radius_map[orb_id]
        new_metadatas.append(metadata)

    collection.update(ids=result["ids"], metadatas=new_metadatas)


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------

def search(query_vector: np.ndarray, n_results: int = 20) -> List[Dict]:
    """
    ANN search: find the top-N most similar orbs to the query vector.

    Returns candidates for gravity reranking in searcher.py.
    Does NOT apply physics — that happens in the searcher layer.

    Args:
        query_vector: 384-dim float32 query embedding.
        n_results:    Number of candidates to retrieve (before reranking).

    Returns:
        List of dicts with keys:
            orb_id    — UUID string
            summary   — LLM-generated summary text
            metadata  — dict with orbital_radius, last_accessed, etc.
            distance  — cosine distance (0=identical); convert to sim = 1 - distance
    """
    collection = _get_collection()
    total = collection.count()
    if total == 0:
        return []

    # Cap n_results to the actual collection size
    n = min(n_results, total)

    results = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    candidates = []
    for i, orb_id in enumerate(results["ids"][0]):
        candidates.append({
            "orb_id":   orb_id,
            "summary":  results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return candidates


def get_all_metadata() -> List[Tuple[str, Dict]]:
    """
    Fetch all orb IDs and their metadata (no vectors, no documents).
    Used exclusively by the decay worker for batch radius synchronization.

    Returns:
        List of (orb_id, metadata_dict) tuples.
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []

    result = collection.get(include=["metadatas"])
    return list(zip(result["ids"], result["metadatas"]))


def count() -> int:
    """Return the total number of orbs in the collection."""
    return _get_collection().count()
