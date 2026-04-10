"""
memory_agent.py — The LLM ↔ OMS bridge.

This module is the entry point for all memory operations that involve an LLM.
It owns two responsibilities:

  1. ingest(raw_text): Store a new memory
       raw_text → LLM summary → embed → ChromaDB + SQLite

  2. chat(user_message, history): Converse with memory context
       query OMS → inject memories → LLM → store exchange → return response

Callers (cli.py) do not need to know about ChromaDB, SQLite, or embeddings.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Dict, Optional

import config
import searcher
from llm_client import get_client
from storage import chroma_store, sqlite_store


def ingest(
    raw_text: str,
    metadata: Optional[Dict] = None,
    provider: Optional[str] = None,
) -> str:
    """
    Full ingestion pipeline: summarize → embed → store in both DBs.

    Steps:
      1. LLM generates a semantic summary of raw_text (2-3 sentences)
      2. SentenceTransformer embeds the summary into a 384-dim vector
      3. ChromaDB stores: orb_id, summary, vector, orbital metadata
      4. SQLite stores:   orb_id, full original text

    Args:
        raw_text: The original memory content to store.
        metadata: Optional extra data (tags, source, etc.) stored in ChromaDB.
        provider: LLM provider override ("anthropic" or "google").
                  Defaults to config.PROVIDER.

    Returns:
        The assigned orb_id (UUID string).
    """
    client = get_client(provider)

    # Step 1: Generate semantic summary via LLM
    print("  Summarizing with LLM...")
    summary = client.summarize(raw_text)

    # Step 2: Embed the summary (not the full text)
    summary_vector = searcher.embed_text(summary)

    # Step 3 + 4: Persist to both stores using the same UUID
    orb_id = str(uuid.uuid4())
    now    = datetime.utcnow()

    chroma_store.add_orb(
        orb_id         = orb_id,
        summary        = summary,
        summary_vector = summary_vector,
        orbital_radius = 1.0,
        last_accessed  = now,
        full_text_id   = orb_id,   # same UUID — pointer to SQLite row
    )
    sqlite_store.save_full_text(orb_id, raw_text)

    return orb_id


def chat(
    user_message: str,
    conversation_history: List[Dict],
    top_k: int = 3,
    provider: Optional[str] = None,
) -> str:
    """
    Memory-augmented chat: retrieve relevant memories, inject into LLM, store exchange.

    Steps:
      1. Query OMS for the top-K memories most relevant to user_message
      2. Format memories as a context string
      3. Call LLM with conversation history + memory context
      4. Store the full exchange (user + assistant) as a new memory
      5. Return the response

    Args:
        user_message:         The current user input.
        conversation_history: Prior turns as list of
                              {"role": "user"/"assistant", "content": "..."} dicts.
                              Should NOT include the current user_message yet.
        top_k:                Number of memory orbs to inject as context (default 3).
        provider:             LLM provider override. Defaults to config.PROVIDER.

    Returns:
        The LLM's response string.
    """
    client = get_client(provider)

    # Step 1: Retrieve relevant memories from OMS
    results = searcher.query_memories(user_message, top_k=top_k)

    # Step 2: Format memories into a readable context block
    memory_context = ""
    if results:
        lines = []
        for i, (orb_id, summary, full_text, gravity) in enumerate(results, start=1):
            # Show full_text truncated for context; summary for provenance
            preview = full_text[:400] if len(full_text) > 400 else full_text
            lines.append(f"[Memory {i} | G={gravity:.3f}]\n{preview}")
        memory_context = "\n\n".join(lines)

    # Step 3: Call LLM with memory-augmented context
    messages = conversation_history + [{"role": "user", "content": user_message}]
    response = client.chat(messages, memory_context=memory_context)

    # Step 4: Store the exchange as a new memory (without recursing into chat)
    exchange = f"User: {user_message}\nAssistant: {response}"
    ingest(exchange, provider=provider)

    return response
