"""
llm_client.py — Multi-provider LLM abstraction for the Orbital Memory System.

Provides a single LLMClient interface with two implementations:
  - AnthropicClient: uses Claude (haiku for summaries, sonnet for chat)
  - GoogleClient:    uses Gemini (flash for summaries, pro for chat)

Switch providers by setting OMS_LLM_PROVIDER in your .env file.
Switch models by editing config.MODELS — no other file needs to change.

Usage:
    client = get_client()               # uses config.PROVIDER
    client = get_client("google")       # override to Google
    summary = client.summarize(text)
    response = client.chat(messages, memory_context)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict

import config


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """Common interface for all LLM providers."""

    @abstractmethod
    def summarize(self, text: str) -> str:
        """
        Generate a concise semantic summary of the input text.
        This summary is what gets embedded and searched in ChromaDB.

        Args:
            text: Raw text to summarize.

        Returns:
            2-3 sentence summary capturing key semantic facts.
        """

    @abstractmethod
    def chat(self, messages: List[Dict], memory_context: str = "") -> str:
        """
        Generate a chat response, optionally grounded in memory context.

        Args:
            messages:       Conversation history as list of
                            {"role": "user"/"assistant", "content": "..."} dicts.
            memory_context: Pre-formatted string of relevant memories to inject
                            into the system prompt. Empty string = no memories.

        Returns:
            Assistant response string.
        """


# ---------------------------------------------------------------------------
# Anthropic (Claude) implementation
# ---------------------------------------------------------------------------

class AnthropicClient(LLMClient):
    """
    Claude-backed LLM client.
    - Summarization: claude-haiku-4-5-20251001 (fast + cheap)
    - Chat:          claude-sonnet-4-6 (high quality)
    """

    def __init__(self) -> None:
        import anthropic
        if not config.ANTHROPIC_API_KEY:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. "
                "Copy .env.example → .env and fill in your key."
            )
        self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self._summarize_model = config.MODELS["summarize"]["anthropic"]
        self._chat_model      = config.MODELS["chat"]["anthropic"]

    def summarize(self, text: str) -> str:
        response = self._client.messages.create(
            model=self._summarize_model,
            max_tokens=250,
            messages=[{
                "role": "user",
                "content": (
                    "Extract the key semantic facts from this text "
                    "in 2-3 concise sentences:\n\n" + text
                ),
            }],
        )
        return response.content[0].text.strip()

    def chat(self, messages: List[Dict], memory_context: str = "") -> str:
        system = (
            "You are a helpful assistant with access to a persistent memory system. "
            "Use the provided memories to give contextually aware responses."
        )
        if memory_context:
            system += f"\n\n--- Relevant Memories ---\n{memory_context}\n---"

        response = self._client.messages.create(
            model=self._chat_model,
            max_tokens=1024,
            system=system,
            messages=messages,
        )
        return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# Google (Gemini) implementation
# ---------------------------------------------------------------------------

class GoogleClient(LLMClient):
    """
    Gemini-backed LLM client.
    - Summarization: gemini-1.5-flash (fast + cheap)
    - Chat:          gemini-1.5-pro (high quality)
    """

    def __init__(self) -> None:
        import google.generativeai as genai
        if not config.GOOGLE_API_KEY:
            raise EnvironmentError(
                "GOOGLE_API_KEY is not set. "
                "Copy .env.example → .env and fill in your key."
            )
        genai.configure(api_key=config.GOOGLE_API_KEY)
        self._genai           = genai
        self._summarize_model = config.MODELS["summarize"]["google"]
        self._chat_model      = config.MODELS["chat"]["google"]

    def summarize(self, text: str) -> str:
        model = self._genai.GenerativeModel(self._summarize_model)
        response = model.generate_content(
            "Extract the key semantic facts from this text "
            "in 2-3 concise sentences:\n\n" + text
        )
        return response.text.strip()

    def chat(self, messages: List[Dict], memory_context: str = "") -> str:
        system_instruction = (
            "You are a helpful assistant with access to a persistent memory system. "
            "Use the provided memories to give contextually aware responses."
        )
        if memory_context:
            system_instruction += f"\n\n--- Relevant Memories ---\n{memory_context}\n---"

        model = self._genai.GenerativeModel(
            self._chat_model,
            system_instruction=system_instruction,
        )

        # Convert OpenAI-style messages to Gemini history format
        history = []
        for msg in messages[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [msg["content"]]})

        chat_session = model.start_chat(history=history)
        response = chat_session.send_message(messages[-1]["content"])
        return response.text.strip()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_client(provider: str | None = None) -> LLMClient:
    """
    Return the appropriate LLM client for the given provider.

    Args:
        provider: "anthropic" or "google". Defaults to config.PROVIDER
                  (set via OMS_LLM_PROVIDER in .env).

    Returns:
        An LLMClient instance ready for summarize() and chat() calls.

    Raises:
        ValueError: If provider name is not recognized.
    """
    p = provider or config.PROVIDER
    if p == "anthropic":
        return AnthropicClient()
    elif p == "google":
        return GoogleClient()
    else:
        raise ValueError(
            f"Unknown provider: '{p}'. "
            "Valid options: 'anthropic', 'google'. "
            "Set OMS_LLM_PROVIDER in your .env file."
        )
