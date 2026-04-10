"""
searcher.py — Orbital gravity search engine for the Orbital Memory System.

Search flow:
  1. Embed query text into a 384-dim vector
  2. ChromaDB ANN search → top-20 candidates (fast approximate search)
  3. Apply lazy decay to each candidate's stored radius
  4. Compute gravity score G_t = (sim × w_s) + ((1/radius) × w_r)
  5. Re-rank by gravity descending, take top-K
  6. Apply momentum to the top result → persist updated radius to ChromaDB
  7. Fetch full texts from SQLite for all results
  8. Return ranked list

Key design: lazy decay
  Decay is computed on-the-fly from (stored_radius + last_accessed) at query
  time. The decayed value is used for scoring but NOT written back to ChromaDB
  for every orb — only the momentum update to the top result is persisted.
  This prevents write-amplification while keeping scores accurate.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Tuple

import numpy as np

import config
from engine import MemoryOrb, apply_decay, apply_momentum, compute_gravity
from storage import chroma_store, sqlite_store

# ---------------------------------------------------------------------------
# Embedding model — lazy-loaded singleton
# ---------------------------------------------------------------------------

_model = None


def _get_model():
    """
    Load the SentenceTransformer on first call, reuse on subsequent calls.
    First run may download ~90 MB.
    """
    global _model
    if _model is None:
        print("Loading embedding model (first run may download ~90 MB)...")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
        print("Model ready.")
    return _model


def embed_text(text: str) -> np.ndarray:
    """
    Encode a string into a 384-dim float32 numpy vector.

    Args:
        text: The string to embed.

    Returns:
        A (384,) float32 numpy array.
    """
    return _get_model().encode(text, convert_to_numpy=True)


# ---------------------------------------------------------------------------
# Core search function
# ---------------------------------------------------------------------------

def query_memories(
    text: str,
    top_k: int = 5,
) -> List[Tuple[str, str, str, float]]:
    """
    Search the memory store by orbital gravity score.

    Args:
        text:   The query string to search for.
        top_k:  Maximum number of results to return.

    Returns:
        List of (orb_id, summary, full_text, gravity_score) tuples,
        sorted by gravity descending (highest = most relevant + accessible).
        Returns an empty list if no orbs are stored.
    """
    if chroma_store.count() == 0:
        return []

    # Step 1: Embed the query
    query_vector = embed_text(text)

    # Step 2: ANN search — get top-20 candidates from ChromaDB
    candidates = chroma_store.search(query_vector, n_results=20)
    if not candidates:
        return []

    # Step 3 + 4: Apply lazy decay and compute gravity for each candidate
    scored: List[Tuple[str, float, float]] = []  # (orb_id, gravity, effective_radius)

    for cand in candidates:
        meta           = cand["metadata"]
        stored_radius  = float(meta["orbital_radius"])
        last_accessed  = datetime.fromisoformat(meta["last_accessed"])
        decay_rate     = float(meta.get("decay_rate", config.DECAY_RATE))

        # Build a temporary MemoryOrb purely for the physics functions
        # (physics functions only read orbital_radius and last_accessed)
        temp_orb = MemoryOrb(
            orb_id         = cand["orb_id"],
            summary        = cand["summary"],
            summary_vector = np.zeros(1),   # unused by physics functions
            orbital_radius = stored_radius,
            last_accessed  = last_accessed,
        )

        # Lazy decay: compute effective radius without writing to DB
        effective_radius        = apply_decay(temp_orb, decay_rate=decay_rate)
        temp_orb.orbital_radius = effective_radius

        # ChromaDB returns cosine distance; convert to similarity
        sim = float(1.0 - cand["distance"])

        g = compute_gravity(
            temp_orb, sim,
            w_s=config.GRAVITY_W_S,
            w_r=config.GRAVITY_W_R,
        )
        scored.append((cand["orb_id"], g, effective_radius))

    # Step 5: Sort by gravity descending, take top_k
    scored.sort(key=lambda x: x[1], reverse=True)
    top_scored = scored[:top_k]

    # Step 6: Apply momentum to the top result and persist
    if top_scored:
        top_orb_id, _, top_effective_radius = top_scored[0]

        # Create temp orb at effective radius for momentum calculation
        temp_top = MemoryOrb(
            orb_id         = top_orb_id,
            summary        = "",
            summary_vector = np.zeros(1),
            orbital_radius = top_effective_radius,
        )
        new_radius = apply_momentum(temp_top, factor=config.MOMENTUM_FACTOR)
        chroma_store.update_radius(top_orb_id, new_radius, datetime.utcnow())

    # Step 7: Fetch full texts from SQLite
    results: List[Tuple[str, str, str, float]] = []
    for orb_id, gravity, _ in top_scored:
        # Retrieve summary from candidates list for display
        summary   = next(c["summary"] for c in candidates if c["orb_id"] == orb_id)
        full_text = sqlite_store.get_full_text(orb_id) or summary
        results.append((orb_id, summary, full_text, gravity))

    return results


# ---------------------------------------------------------------------------
# List all memories
# ---------------------------------------------------------------------------

def list_all() -> List[dict]:
    """
    Return all memory orbs sorted by orbital_radius ascending (nearest first).

    Returns:
        List of dicts with keys: orb_id, summary, orbital_radius, last_accessed.
    """
    all_meta = chroma_store.get_all_metadata()
    if not all_meta:
        return []

    orbs = []
    for orb_id, meta in all_meta:
        orbs.append({
            "orb_id":         orb_id,
            "summary":        "",   # not fetched for list — metadata only
            "orbital_radius": float(meta["orbital_radius"]),
            "last_accessed":  meta["last_accessed"],
        })

    # Sort by orbital_radius ascending (closest to center first)
    return sorted(orbs, key=lambda o: o["orbital_radius"])
