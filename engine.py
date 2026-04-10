"""
engine.py — Core physics of the Orbital Memory System.

This module is pure math: no I/O, no database, no embeddings.
It defines the MemoryOrb data structure and the three physics functions
that govern how memories behave over time.

Physics metaphor:
  - Memories orbit a "center of relevance".
  - Orbital radius represents distance from center (lower = more accessible).
  - Memories drift outward (decay) as time passes.
  - Memories snap inward (momentum) when retrieved.
  - Retrieval ranking uses a gravity score combining semantic similarity
    and orbital proximity.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np


@dataclass
class MemoryOrb:
    """
    A single unit of memory in the Orbital Memory System.

    In the hybrid architecture, the MemoryOrb represents the searchable,
    physics-governed layer. Full original text is stored separately in SQLite,
    referenced by full_text_pointer (same UUID as orb_id).

    Attributes:
        orb_id:           UUID string — shared key between ChromaDB and SQLite.
        summary:          LLM-generated semantic summary of the original text.
                          This is what gets embedded and searched against.
        summary_vector:   384-dim float32 embedding of the summary.
        orbital_radius:   Distance from the center of relevance.
                          Starts at 1.0; grows with decay, shrinks with access.
        last_accessed:    UTC timestamp of the last access or creation.
        metadata:         Optional dict for tags, source, or any extra data.
        full_text_pointer: Reference to the SQLite row (same UUID as orb_id).
    """
    orb_id: str
    summary: str
    summary_vector: np.ndarray
    orbital_radius: float = 1.0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)
    full_text_pointer: str = ""


def apply_decay(orb: MemoryOrb, decay_rate: float = 0.01) -> float:
    """
    Compute the new orbital radius after temporal decay.

    Memories fade from the center over time — the longer since last access,
    the further the orb drifts outward.

    Formula:
        R_new = R_old + ln(1 + Δt) × λ

    where:
        Δt   = hours elapsed since last_accessed
        λ    = decay_rate (controls how fast memories fade)

    Args:
        orb:        The MemoryOrb to decay.
        decay_rate: Lambda constant (λ). Default 0.01.

    Returns:
        New orbital_radius value. Does NOT mutate the orb.
    """
    now = datetime.utcnow()
    delta_t_hours = (now - orb.last_accessed).total_seconds() / 3600.0
    # ln(1 + 0) = 0, so a just-accessed orb barely changes radius
    return orb.orbital_radius + math.log(1.0 + delta_t_hours) * decay_rate


def apply_momentum(orb: MemoryOrb, factor: float = 0.9) -> float:
    """
    Compute the new orbital radius after a momentum boost from retrieval.

    When a memory is retrieved, it snaps closer to the center —
    reinforcing its accessibility for future queries.

    Formula:
        R_new = R_old × factor

    where:
        factor < 1.0 pulls the orb inward (toward center)

    Args:
        orb:    The MemoryOrb being retrieved.
        factor: Momentum factor (must be < 1.0 to pull inward). Default 0.9.

    Returns:
        New orbital_radius value. Does NOT mutate the orb.
    """
    return orb.orbital_radius * factor


def compute_gravity(
    orb: MemoryOrb,
    semantic_similarity: float,
    w_s: float = 0.7,
    w_r: float = 0.3,
) -> float:
    """
    Compute the total gravitational score for ranking search results.

    Combines two forces:
      1. Semantic pull  — how closely the orb's content matches the query.
      2. Orbital pull   — how close the orb is to the center (low radius = high pull).

    Formula:
        G_t = (semantic_similarity × w_s) + ((1 / radius) × w_r)

    Args:
        orb:                The MemoryOrb being scored.
        semantic_similarity: Cosine similarity between query vector and orb vector.
                             Typically in [0, 1] for text embeddings.
        w_s:                Weight for semantic similarity (default 0.7).
        w_r:                Weight for orbital proximity (default 0.3).

    Returns:
        Gravity score G_t. Higher = more relevant and more accessible.
    """
    # Epsilon guard: prevents ZeroDivisionError if radius ever collapses to 0
    epsilon = 1e-9
    radius_term = 1.0 / (orb.orbital_radius + epsilon)
    return (semantic_similarity * w_s) + (radius_term * w_r)
