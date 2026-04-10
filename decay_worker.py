"""
decay_worker.py — Standalone background decay synchronization process.

This is an opt-in, independent script — it does NOT run automatically.
Start it manually when you want stored orbital radii to stay current:

    python decay_worker.py               # sync every 30 minutes (default)
    python decay_worker.py --interval 60 # sync every 60 minutes
    python decay_worker.py --once        # sync once and exit (good for cron)

Why this exists:
    searcher.py uses *lazy decay* — it computes the correct effective radius
    at query time from stored_radius + last_accessed without writing back.
    This means stored radii in ChromaDB are only updated when a memory is
    retrieved (momentum). Without this worker, stored radii remain at their
    last-momentum value even as time passes.

    This worker syncs the computed decay values back to ChromaDB so that:
    - `python cli.py list` shows accurate current radii
    - External tools inspecting ChromaDB see realistic values
    - Very stale orbs (high radius) are visually identifiable

What this worker does NOT do:
    - It does NOT update last_accessed (that is only for retrieval events)
    - It does NOT delete orbs (archiving policy is the user's decision)
    - It does NOT affect search accuracy (lazy decay handles that)

Stop it cleanly with Ctrl+C or SIGTERM.
"""

import argparse
import math
import signal
import sys
import time
from datetime import datetime
from typing import List, Tuple

import config
from storage import chroma_store

# ---------------------------------------------------------------------------
# Decay computation (mirrors engine.apply_decay without importing MemoryOrb)
# ---------------------------------------------------------------------------

def _compute_decayed_radius(
    stored_radius: float,
    last_accessed_iso: str,
    decay_rate: float = config.DECAY_RATE,
) -> float:
    """
    Compute the current effective orbital radius using the decay formula.

    Formula: R_new = R_old + ln(1 + Δt) × λ
    where Δt = hours since last_accessed.

    This is the same formula as engine.apply_decay — duplicated here so
    decay_worker.py has zero dependency on the MemoryOrb dataclass.

    Args:
        stored_radius:     The radius value stored in ChromaDB.
        last_accessed_iso: ISO 8601 string of the last access timestamp.
        decay_rate:        Lambda constant (λ). Per-orb value from metadata.

    Returns:
        New orbital radius (always >= stored_radius).
    """
    last_accessed = datetime.fromisoformat(last_accessed_iso)
    delta_t_hours = (datetime.utcnow() - last_accessed).total_seconds() / 3600.0
    return stored_radius + math.log(1.0 + delta_t_hours) * decay_rate


# ---------------------------------------------------------------------------
# Core sync function
# ---------------------------------------------------------------------------

def sync_decay() -> int:
    """
    Fetch all orb metadata, recompute radii, and batch-update ChromaDB.

    Fetches only metadata (no vectors) — lightweight even with many orbs.
    Does NOT update last_accessed — that field is reserved for retrieval events.

    Returns:
        Number of orbs updated.
    """
    all_meta = chroma_store.get_all_metadata()
    if not all_meta:
        return 0

    updates: List[Tuple[str, float]] = []
    for orb_id, meta in all_meta:
        new_radius = _compute_decayed_radius(
            stored_radius    = float(meta["orbital_radius"]),
            last_accessed_iso = meta["last_accessed"],
            decay_rate       = float(meta.get("decay_rate", config.DECAY_RATE)),
        )
        updates.append((orb_id, new_radius))

    chroma_store.batch_update_radii(updates)
    return len(updates)


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------

_running = True


def _handle_signal(sig, frame) -> None:
    """Clean shutdown on SIGINT (Ctrl+C) or SIGTERM."""
    global _running
    print("\n[OMS Decay Worker] Shutdown signal received. Stopping...")
    _running = False
    sys.exit(0)


def run_loop(interval_seconds: int) -> None:
    """
    Run decay sync on a fixed interval until interrupted.

    Args:
        interval_seconds: Seconds to sleep between sync runs.
    """
    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    interval_min = interval_seconds // 60
    print(f"[OMS Decay Worker] Started. Syncing every {interval_min} minute(s).")
    print("Press Ctrl+C to stop.\n")

    while _running:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        try:
            count = sync_decay()
            print(f"[{timestamp}] Decay synced {count} orb(s).")
        except Exception as exc:
            print(f"[{timestamp}] Error during sync: {exc}")

        time.sleep(interval_seconds)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="decay_worker",
        description="OMS Orbital Decay Worker — syncs decayed radii back to ChromaDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python decay_worker.py                # sync every 30 min\n"
            "  python decay_worker.py --interval 60  # sync every 60 min\n"
            "  python decay_worker.py --once         # sync once and exit\n"
        ),
    )
    parser.add_argument(
        "--interval", type=int, default=30, metavar="MINUTES",
        help="Sync interval in minutes (default: 30). Ignored with --once.",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single decay sync and exit immediately.",
    )
    args = parser.parse_args()

    if args.once:
        count = sync_decay()
        print(f"Decay synced {count} orb(s). Done.")
    else:
        run_loop(interval_seconds=args.interval * 60)


if __name__ == "__main__":
    main()
