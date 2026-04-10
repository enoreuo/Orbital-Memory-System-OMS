"""
tests/test_engine.py — Unit tests for the OMS physics engine.

Tests only engine.py — no database, no embeddings, no sentence-transformers.
All inputs are constructed manually so the suite runs in under a second.

Run with:
    python -m pytest tests/ -v
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from engine import MemoryOrb, apply_decay, apply_momentum, compute_gravity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_orb() -> MemoryOrb:
    """A brand-new MemoryOrb with radius=1.0, last_accessed=utcnow."""
    return MemoryOrb(
        orb_id="fixture-fresh-001",
        summary="test memory",
        summary_vector=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        orbital_radius=1.0,
        last_accessed=datetime.utcnow(),
    )


@pytest.fixture
def stale_orb() -> MemoryOrb:
    """An orb last accessed 24 hours ago, radius=1.0."""
    return MemoryOrb(
        orb_id="fixture-stale-001",
        summary="old memory",
        summary_vector=np.array([0.4, 0.5, 0.6], dtype=np.float32),
        orbital_radius=1.0,
        last_accessed=datetime.utcnow() - timedelta(hours=24),
    )


# ---------------------------------------------------------------------------
# TestDecay
# ---------------------------------------------------------------------------

class TestDecay:
    def test_radius_increases_with_time(self, stale_orb: MemoryOrb) -> None:
        """Core requirement: decay must increase radius when time has passed."""
        new_radius = apply_decay(stale_orb)
        assert new_radius > stale_orb.orbital_radius

    def test_no_decay_for_just_accessed(self, fresh_orb: MemoryOrb) -> None:
        """
        Orb accessed right now: Δt ≈ 0, so ln(1+0)×λ ≈ 0.
        New radius should be very close to the original.
        """
        new_radius = apply_decay(fresh_orb)
        assert new_radius == pytest.approx(fresh_orb.orbital_radius, abs=0.01)

    def test_decay_is_monotonically_increasing(self) -> None:
        """More time elapsed → larger radius increase (log is monotonically increasing)."""
        base = datetime.utcnow()
        orb_1h   = MemoryOrb("id1", "x", np.zeros(3), 1.0, base - timedelta(hours=1))
        orb_10h  = MemoryOrb("id2", "x", np.zeros(3), 1.0, base - timedelta(hours=10))
        orb_100h = MemoryOrb("id3", "x", np.zeros(3), 1.0, base - timedelta(hours=100))

        r1 = apply_decay(orb_1h)
        r10 = apply_decay(orb_10h)
        r100 = apply_decay(orb_100h)

        assert r1 < r10 < r100

    def test_apply_decay_does_not_mutate(self, stale_orb: MemoryOrb) -> None:
        """Physics functions must be pure — apply_decay must not modify the orb."""
        original_radius = stale_orb.orbital_radius
        original_time = stale_orb.last_accessed

        apply_decay(stale_orb)

        assert stale_orb.orbital_radius == original_radius
        assert stale_orb.last_accessed == original_time

    def test_custom_decay_rate(self, stale_orb: MemoryOrb) -> None:
        """Higher decay_rate produces a larger radius increase."""
        r_slow = apply_decay(stale_orb, decay_rate=0.001)
        r_fast = apply_decay(stale_orb, decay_rate=0.1)
        assert r_fast > r_slow


# ---------------------------------------------------------------------------
# TestMomentum
# ---------------------------------------------------------------------------

class TestMomentum:
    def test_radius_decreases_on_access(self, fresh_orb: MemoryOrb) -> None:
        """Core requirement: momentum must pull radius inward (shrink it)."""
        new_radius = apply_momentum(fresh_orb)
        assert new_radius < fresh_orb.orbital_radius

    def test_momentum_factor_respected(self, fresh_orb: MemoryOrb) -> None:
        """Custom factor is applied exactly as R_new = R_old × factor."""
        new_radius = apply_momentum(fresh_orb, factor=0.5)
        assert new_radius == pytest.approx(fresh_orb.orbital_radius * 0.5)

    def test_default_factor_is_0_9(self, fresh_orb: MemoryOrb) -> None:
        """Default factor of 0.9 reduces radius to 90% of original."""
        new_radius = apply_momentum(fresh_orb)
        assert new_radius == pytest.approx(fresh_orb.orbital_radius * 0.9)

    def test_apply_momentum_does_not_mutate(self, fresh_orb: MemoryOrb) -> None:
        """Physics functions must be pure — apply_momentum must not modify the orb."""
        original_radius = fresh_orb.orbital_radius
        apply_momentum(fresh_orb)
        assert fresh_orb.orbital_radius == original_radius


# ---------------------------------------------------------------------------
# TestGravity
# ---------------------------------------------------------------------------

class TestGravity:
    def test_gravity_formula_correctness(self) -> None:
        """
        Hand-calculated verification:
          G = (0.8 × 0.7) + ((1 / 2.0) × 0.3)
            = 0.56 + 0.15
            = 0.71
        """
        orb = MemoryOrb("id-g1", "x", np.zeros(3), orbital_radius=2.0)
        g = compute_gravity(orb, semantic_similarity=0.8, w_s=0.7, w_r=0.3)
        assert g == pytest.approx(0.71, abs=1e-6)

    def test_closer_orb_ranks_higher_than_distant(self) -> None:
        """Same semantic similarity — orb at radius=0.5 must beat orb at radius=5.0."""
        close = MemoryOrb("id-g2", "x", np.zeros(3), orbital_radius=0.5)
        far   = MemoryOrb("id-g3", "x", np.zeros(3), orbital_radius=5.0)
        sim   = 0.5

        assert compute_gravity(close, sim) > compute_gravity(far, sim)

    def test_higher_similarity_ranks_higher(self) -> None:
        """Same radius — orb with similarity=0.9 must beat orb with similarity=0.1."""
        orb = MemoryOrb("id-g4", "x", np.zeros(3), orbital_radius=1.0)
        assert compute_gravity(orb, 0.9) > compute_gravity(orb, 0.1)

    def test_zero_radius_guard(self) -> None:
        """Epsilon guard must prevent ZeroDivisionError when radius = 0.0."""
        orb = MemoryOrb("id-g5", "x", np.zeros(3), orbital_radius=0.0)
        g = compute_gravity(orb, semantic_similarity=0.5)  # must not raise
        assert g > 0

    def test_weights_sum_sensitivity(self) -> None:
        """
        With w_s=1.0, w_r=0.0 the score equals semantic_similarity exactly
        (orbital distance has no influence).
        """
        orb = MemoryOrb("id-g6", "x", np.zeros(3), orbital_radius=100.0)
        g = compute_gravity(orb, semantic_similarity=0.75, w_s=1.0, w_r=0.0)
        assert g == pytest.approx(0.75, abs=1e-6)
