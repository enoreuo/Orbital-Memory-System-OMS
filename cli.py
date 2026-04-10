"""
cli.py — Command-line interface for the Orbital Memory System v2.

Usage:
    python cli.py add "Your memory text here"
    python cli.py add "Your memory text here" --provider google
    python cli.py search "What you want to find"
    python cli.py search "query" --top-k 10
    python cli.py chat
    python cli.py chat --provider google
    python cli.py list

All heavy logic lives in memory_agent.py and searcher.py.
This module is a thin shell that parses arguments and formats output.
"""

import argparse
import sys

import config
import memory_agent
import searcher
from storage import sqlite_store, chroma_store


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_add(args: argparse.Namespace) -> None:
    """Store a new memory: summarize → embed → save to ChromaDB + SQLite."""
    provider = args.provider or config.PROVIDER
    print(f"[OMS] Storing memory (provider: {provider})...")
    orb_id = memory_agent.ingest(args.text, provider=provider)
    print(f"[OMS] Memory stored successfully.")
    print(f"  orb_id  : {orb_id}")
    print(f"  preview : \"{args.text[:80]}{'...' if len(args.text) > 80 else ''}\"")


def cmd_search(args: argparse.Namespace) -> None:
    """Search memories by orbital gravity score."""
    results = searcher.query_memories(args.query, top_k=args.top_k)

    if not results:
        print("No memories found. Add some with: python cli.py add \"your text\"")
        return

    print(f"\nSearch results for: \"{args.query}\"\n")
    for rank, (orb_id, summary, full_text, gravity) in enumerate(results, start=1):
        preview = full_text if len(full_text) <= 100 else full_text[:97] + "..."
        print(
            f"Rank {rank} [G={gravity:.4f}]  \"{preview}\"\n"
            f"         id={orb_id[:8]}...  summary: \"{summary[:60]}...\"\n"
        )


def cmd_chat(args: argparse.Namespace) -> None:
    """Interactive chat loop with memory-augmented responses."""
    provider = args.provider or config.PROVIDER
    print(f"[OMS Chat] Provider: {provider} | Model: {config.MODELS['chat'][provider]}")
    print("Type 'quit' or 'exit' to end the session.\n")

    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[OMS Chat] Session ended.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("[OMS Chat] Session ended.")
            break

        print("Assistant: ", end="", flush=True)
        response = memory_agent.chat(
            user_message         = user_input,
            conversation_history = history,
            top_k                = 3,
            provider             = provider,
        )
        print(response)
        print()

        # Update conversation history for multi-turn context
        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": response})


def cmd_list(args: argparse.Namespace) -> None:
    """List all memory orbs sorted by orbital radius (nearest to center first)."""
    orbs = searcher.list_all()

    if not orbs:
        print("No memories stored yet. Add one with: python cli.py add \"your text\"")
        return

    print(f"\nAll memories ({len(orbs)} total), sorted by orbital radius:\n")
    for orb in orbs:
        # Fetch full text for preview
        full_text = sqlite_store.get_full_text(orb["orb_id"]) or "(no full text)"
        preview   = full_text if len(full_text) <= 70 else full_text[:67] + "..."
        print(
            f"  [id={orb['orb_id'][:8]}...]  "
            f"radius={orb['orbital_radius']:.4f}  "
            f"\"{preview}\""
        )


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ArgumentParser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="oms",
        description="Orbital Memory System v2 — LLM memory with orbital physics.",
    )

    # Global --provider flag available on all subcommands
    provider_kwargs = dict(
        default=None,
        choices=["anthropic", "google"],
        metavar="PROVIDER",
        help="LLM provider: 'anthropic' (Claude) or 'google' (Gemini). "
             f"Default: {config.PROVIDER} (from .env)",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # --- add ---
    add_p = subparsers.add_parser("add", help="Store a new memory")
    add_p.add_argument("text", help="The memory content to store")
    add_p.add_argument("--provider", **provider_kwargs)
    add_p.set_defaults(func=cmd_add)

    # --- search ---
    search_p = subparsers.add_parser("search", help="Search memories by gravity")
    search_p.add_argument("query", help="The query text to search for")
    search_p.add_argument(
        "--top-k", type=int, default=5, dest="top_k",
        help="Number of results to return (default: 5)",
    )
    search_p.set_defaults(func=cmd_search)

    # --- chat ---
    chat_p = subparsers.add_parser("chat", help="Interactive chat with memory context")
    chat_p.add_argument("--provider", **provider_kwargs)
    chat_p.set_defaults(func=cmd_chat)

    # --- list ---
    list_p = subparsers.add_parser("list", help="List all memories by orbital radius")
    list_p.set_defaults(func=cmd_list)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Initialize storage, parse arguments, dispatch to subcommand."""
    # Initialize both stores on startup (idempotent — safe to call every run)
    sqlite_store.initialize_db()
    # ChromaDB initializes lazily on first access via _get_collection()

    parser = build_parser()
    args   = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
