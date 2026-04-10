"""
config.py — Central configuration for the Orbital Memory System.

All tuneable values live here: model names, physics constants, storage paths.
To switch LLM provider: set OMS_LLM_PROVIDER=google in your .env file.
To switch models: edit the MODELS dict below.

Copy .env.example → .env and fill in your API keys before running.
"""

import os
from dotenv import load_dotenv

# Load .env file if present (safe to call even if .env doesn't exist)
load_dotenv()

# ---------------------------------------------------------------------------
# LLM Provider
# ---------------------------------------------------------------------------

# Active provider: "anthropic" or "google"
# Override via: OMS_LLM_PROVIDER=google in .env
PROVIDER: str = os.getenv("OMS_LLM_PROVIDER", "anthropic")

# Model templates — edit here to swap models without touching any other file
MODELS: dict = {
    "summarize": {
        # Fast + cheap models for ingestion (summary generation)
        "anthropic": "claude-haiku-4-5-20251001",
        "google":    "gemini-2.0-flash-lite",
    },
    "chat": {
        # Powerful models for interactive conversation
        "anthropic": "claude-sonnet-4-6",
        "google":    "gemini-2.5-flash",
    },
}

# ---------------------------------------------------------------------------
# API Keys (loaded from .env — never hardcode here)
# ---------------------------------------------------------------------------

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY: str    = os.getenv("GOOGLE_API_KEY", "")

# ---------------------------------------------------------------------------
# Orbital Physics Constants
# ---------------------------------------------------------------------------

# Decay rate (λ): controls how fast memories fade outward
# Higher value = faster fade. Default 0.01 is gentle (days-scale decay)
DECAY_RATE: float = 0.01

# Momentum factor: how much the radius shrinks when an orb is retrieved
# 0.9 = 10% inward pull per retrieval
MOMENTUM_FACTOR: float = 0.9

# Gravity formula weights (must sum to 1.0 for intuitive scores)
# w_s: weight of semantic similarity
# w_r: weight of orbital proximity (1/radius)
GRAVITY_W_S: float = 0.7
GRAVITY_W_R: float = 0.3

# ---------------------------------------------------------------------------
# Storage Paths
# ---------------------------------------------------------------------------

# ChromaDB persistent storage directory
CHROMA_PATH: str       = "./chroma_db"
CHROMA_COLLECTION: str = "oms_memories"

# SQLite full-text archive
SQLITE_DB_PATH: str = "oms_fulltext.db"

# ---------------------------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------------------------

# Sentence transformer model for embedding summaries
# all-MiniLM-L6-v2: 384 dimensions, ~90 MB download, fast inference
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
