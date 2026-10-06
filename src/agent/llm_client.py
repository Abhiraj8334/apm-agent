"""
llm_client.py

A single call_llm(prompt) -> str interface that the RCA agent depends on.
Today it's backed by Gemini; the agent's reasoning loop never imports
google.generativeai directly, so swapping in Ollama, OpenAI, or Claude
later means writing one new class here, not touching agent.py at all.
"""

from __future__ import annotations

import os
import random
import time
from abc import ABC, abstractmethod
from typing import Callable, TypeVar

T = TypeVar("T")

# HTTP status codes worth retrying: rate limit, server error, overloaded, timeout.
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def is_retryable(exc: Exception) -> bool:
    """True if the error looks transient (e.g. 503 'high demand'), False for permanent ones (bad key, bad request)."""
    return getattr(exc, "code", None) in RETRYABLE_STATUS_CODES


def retry_with_backoff(
    fn: Callable[[], T],
    max_attempts: int = 5,
    base_delay: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """
    Call fn(), retrying transient failures with exponential backoff plus jitter
    (2s, 4s, 8s, 16s...). Permanent errors are raised immediately, since retrying
    a bad API key would just waste time. After max_attempts the last error is raised.
    `sleep` is injectable so tests don't actually wait.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we re-raise anything non-retryable
            if not is_retryable(exc) or attempt == max_attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
            print(f"  [LLM busy or unavailable (status {getattr(exc, 'code', '?')}); retry {attempt}/{max_attempts - 1} in {delay:.0f}s]")
            sleep(delay)
    raise RuntimeError("unreachable")  # pragma: no cover


class LLMClient(ABC):
    @abstractmethod
    def call(self, prompt: str) -> str:
        """Send `prompt` to the model and return its raw text response."""
        raise NotImplementedError


class GeminiClient(LLMClient):
    """
    Gemini backend using Google's current SDK (`google-genai`). The older
    `google-generativeai` package is deprecated, and model names get retired
    regularly, so the model is configurable via the GEMINI_MODEL env var.
    The default is an alias that tracks Google's latest Flash model, which
    avoids hardcoding a name that goes stale.
    """

    DEFAULT_MODEL = "gemini-flash-latest"

    def __init__(self, model_name: str | None = None, api_key: str | None = None):
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "google-genai is required for GeminiClient. "
                "Install it with: pip install google-genai"
            ) from e

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "No Gemini API key found. Set GEMINI_API_KEY in your .env file "
                "(see .env.example) or pass api_key explicitly."
            )

        self._client = genai.Client(api_key=key)
        self._model_name = model_name or os.environ.get("GEMINI_MODEL") or self.DEFAULT_MODEL

    def call(self, prompt: str) -> str:
        response = retry_with_backoff(
            lambda: self._client.models.generate_content(model=self._model_name, contents=prompt)
        )
        # Newer Gemini models can return extra non-text parts (e.g. thought signatures).
        # Collect only the visible text parts so we don't trigger SDK warnings.
        try:
            parts = response.candidates[0].content.parts
            text = "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False))
            if text:
                return text
        except (AttributeError, IndexError, TypeError):
            pass
        return response.text or ""


class FakeLLMClient(LLMClient):
    """
    Deterministic stand-in for testing the agent loop without hitting a
    real API or requiring an API key. Returns responses from a fixed
    script, one per call, in order.
    """

    def __init__(self, scripted_responses: list[str]):
        self._responses = list(scripted_responses)
        self._call_count = 0

    def call(self, prompt: str) -> str:
        if self._call_count >= len(self._responses):
            raise RuntimeError("FakeLLMClient script exhausted — agent made more calls than expected")
        response = self._responses[self._call_count]
        self._call_count += 1
        return response
