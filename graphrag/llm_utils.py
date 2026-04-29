"""LLM helper – wrapper around Google Gemini API (google-genai SDK).

Provides chat() and embed() with rate limiting for Gemini free tier.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import Any

import numpy as np
from google import genai
from google.genai import types

import config

# ── Configure Gemini Client ─────────────────────────────────────────────
_client = genai.Client(api_key=config.GEMINI_API_KEY)

MAX_RETRIES = 5


# ── Rate Limiter ────────────────────────────────────────────────────────

class _RateLimiter:
    """Thread-safe sliding-window rate limiter for RPM and RPD."""

    def __init__(self, rpm: int, rpd: int):
        self._rpm = rpm
        self._rpd = rpd
        self._minute_window: deque[float] = deque()   # timestamps within last 60s
        self._day_window: deque[float] = deque()       # timestamps within last 24h
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Block until a request slot is available."""
        with self._lock:
            now = time.time()
            # Purge old entries
            while self._minute_window and now - self._minute_window[0] > 60:
                self._minute_window.popleft()
            while self._day_window and now - self._day_window[0] > 86400:
                self._day_window.popleft()

            # Wait for minute window
            if len(self._minute_window) >= self._rpm:
                sleep_until = self._minute_window[0] + 60
                wait_s = max(0, sleep_until - now)
                if wait_s > 0:
                    print(f"  ⏳ Rate limit: waiting {wait_s:.0f}s (RPM={self._rpm})")
                    time.sleep(wait_s)
                    now = time.time()
                    while self._minute_window and now - self._minute_window[0] > 60:
                        self._minute_window.popleft()

            # Check daily limit
            if len(self._day_window) >= self._rpd:
                sleep_until = self._day_window[0] + 86400
                remaining_h = max(0, sleep_until - time.time()) / 3600
                raise RuntimeError(
                    f"Daily API limit reached ({self._rpd} RPD). "
                    f"Resets in ~{remaining_h:.1f} hours."
                )

            # Record this request
            now = time.time()
            self._minute_window.append(now)
            self._day_window.append(now)

    @property
    def daily_remaining(self) -> int:
        now = time.time()
        while self._day_window and now - self._day_window[0] > 86400:
            self._day_window.popleft()
        return max(0, self._rpd - len(self._day_window))


_limiter = _RateLimiter(rpm=config.RATE_LIMIT_RPM, rpd=config.RATE_LIMIT_RPD)
_credits_depleted = False  # set True after first "depleted" error → skip all future calls


def is_available() -> bool:
    """Return True if LLM API is likely available (credits not depleted)."""
    return not _credits_depleted


# ── Chat Completion ─────────────────────────────────────────────────────

def chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = config.GENERATION_TEMPERATURE,
    max_tokens: int = config.MAX_GENERATION_TOKENS,
    **kwargs: Any,
) -> str:
    """Send messages to Gemini and return the response text."""
    global _credits_depleted
    if _credits_depleted:
        return ""
    model_name = model or config.GEMINI_CHAT_MODEL

    # Build Gemini contents from OpenAI-style messages
    system_text = ""
    contents = []
    for msg in messages:
        role = msg["role"]
        text = msg["content"]
        if role == "system":
            system_text = text
        elif role == "user":
            prompt = f"{system_text}\n\n{text}" if system_text and not contents else text
            contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
            if system_text:
                system_text = ""
        elif role == "assistant":
            contents.append(types.Content(role="model", parts=[types.Part(text=text)]))

    if not contents and system_text:
        contents.append(types.Content(role="user", parts=[types.Part(text=system_text)]))

    gen_config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )

    for attempt in range(MAX_RETRIES):
        try:
            _limiter.wait()  # respect RPM / RPD
            resp = _client.models.generate_content(
                model=model_name,
                contents=contents,
                config=gen_config,
            )
            if resp.text:
                return resp.text
            return ""
        except RuntimeError:
            raise  # daily limit — don't retry
        except Exception as e:
            err_str = str(e)
            # Credits depleted — no point retrying
            if "depleted" in err_str.lower() or "billing" in err_str.lower():
                _credits_depleted = True
                print(f"  [!] Gemini credits depleted -- falling back to no-LLM mode")
                return ""
            if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                wait = min(2 ** attempt * 15, 120)  # longer backoff for free tier
                print(f"  ⏳ Rate-limited (attempt {attempt+1}/{MAX_RETRIES}), waiting {wait}s …")
                time.sleep(wait)
            elif attempt == MAX_RETRIES - 1:
                print(f"  [!] Gemini error after {MAX_RETRIES} retries: {e}")
                return ""
            else:
                time.sleep(2)
    return ""


async def achat(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = config.GENERATION_TEMPERATURE,
    max_tokens: int = config.MAX_GENERATION_TOKENS,
    **kwargs: Any,
) -> str:
    """Async version – runs sync chat in executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, lambda: chat(messages, model, temperature, max_tokens, **kwargs)
    )


# ── Embeddings ──────────────────────────────────────────────────────────

def embed(texts: list[str], model: str | None = None) -> np.ndarray:
    """Return (N, dim) float32 array of embeddings using Gemini."""
    global _credits_depleted
    if _credits_depleted:
        return np.zeros((len(texts), config.EMBEDDING_DIMENSION), dtype=np.float32)
    model_name = model or config.GEMINI_EMBEDDING_MODEL
    all_vecs: list[list[float]] = []

    batch_size = 100  # Gemini batch limit
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for attempt in range(MAX_RETRIES):
            try:
                _limiter.wait()  # respect RPM / RPD
                result = _client.models.embed_content(
                    model=model_name,
                    contents=batch,
                )
                for emb in result.embeddings:
                    all_vecs.append(list(emb.values))
                break
            except RuntimeError:
                raise  # daily limit
            except Exception as e:
                err_str = str(e)
                if "depleted" in err_str.lower() or "billing" in err_str.lower():
                    _credits_depleted = True
                    print(f"  [!] Gemini credits depleted -- filling zeros")
                    all_vecs.extend([[0.0] * config.EMBEDDING_DIMENSION] * len(batch))
                    break
                if "429" in err_str or "quota" in err_str.lower():
                    wait = min(2 ** attempt * 15, 120)
                    print(f"  ⏳ Embedding rate-limited (attempt {attempt+1}), waiting {wait}s …")
                    time.sleep(wait)
                elif attempt == MAX_RETRIES - 1:
                    print(f"  [!] Embedding error: {e}, filling zeros")
                    all_vecs.extend([[0.0] * config.EMBEDDING_DIMENSION] * len(batch))
                else:
                    time.sleep(2)

    return np.array(all_vecs, dtype=np.float32)


async def aembed(texts: list[str], model: str | None = None) -> np.ndarray:
    """Async version – runs sync embed in executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: embed(texts, model))
