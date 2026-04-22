from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Protocol

log = logging.getLogger(__name__)


def cache_key(model: str, temperature: float, messages: list[dict]) -> str:
    payload = f"{model}\n{temperature}\n{json.dumps(messages, sort_keys=True, ensure_ascii=False)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Backend(Protocol):
    def chat(self, messages: list[dict], model: str, temperature: float) -> str: ...


class DummyBackend:
    def __init__(self, responder: Callable[[list[dict], str, float], str] | None = None):
        self._responder = responder or (lambda msgs, m, t: "DUMMY_RESPONSE")
        self.call_count = 0

    def chat(self, messages: list[dict], model: str, temperature: float) -> str:
        self.call_count += 1
        return self._responder(messages, model, temperature)


class OpenAIBackend:
    def __init__(self, base_url: str, api_key: str, timeout_s: float = 120.0):
        from openai import OpenAI
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)

    def chat(self, messages: list[dict], model: str, temperature: float) -> str:
        resp = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


def _validate_and_load_cache(path: Path) -> dict[str, str]:
    """Read JSONL cache; drop malformed trailing line; return {key: response}."""
    if not path.exists():
        return {}
    cache: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        raw = f.read()
    if not raw:
        return {}
    lines = raw.split("\n")
    trailing_empty = lines and lines[-1] == ""
    body_lines = lines[:-1] if trailing_empty else lines
    good_lines: list[str] = []
    dropped = False
    for i, line in enumerate(body_lines):
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            if i == len(body_lines) - 1:
                dropped = True
                break
            raise
        if "key" in obj and "response" in obj:
            cache[obj["key"]] = obj["response"]
            good_lines.append(line)
    if dropped:
        log.warning("Dropped malformed trailing line in %s", path)
        with path.open("w", encoding="utf-8") as f:
            if good_lines:
                f.write("\n".join(good_lines) + "\n")
            else:
                f.write("")
    return cache


class LLMClient:
    def __init__(
        self,
        backend: Backend,
        cache_path: Path,
        max_retries: int = 3,
        base_backoff_s: float = 1.0,
    ):
        self._backend = backend
        self._cache_path = Path(cache_path)
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = _validate_and_load_cache(self._cache_path)
        self._max_retries = max_retries
        self._base_backoff_s = base_backoff_s
        self.call_count = 0
        self.cache_hits = 0

    def chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.0,
    ) -> str:
        key = cache_key(model, temperature, messages)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]

        response = self._call_with_retry(messages, model, temperature)
        self._append_cache(key, model, temperature, messages, response)
        self._cache[key] = response
        self.call_count += 1
        return response

    def _call_with_retry(self, messages: list[dict], model: str, temperature: float) -> str:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return self._backend.chat(messages, model, temperature)
            except Exception as exc:
                last_exc = exc
                if attempt == self._max_retries - 1:
                    break
                wait = self._base_backoff_s * (4 ** attempt)
                log.warning("LLM call failed (attempt %d/%d): %s; retry in %.1fs",
                            attempt + 1, self._max_retries, exc, wait)
                time.sleep(wait)
        assert last_exc is not None
        raise last_exc

    def _append_cache(
        self,
        key: str,
        model: str,
        temperature: float,
        messages: list[dict],
        response: str,
    ) -> None:
        line = json.dumps(
            {
                "key": key,
                "model": model,
                "temperature": temperature,
                "messages": messages,
                "response": response,
                "created_at": time.time(),
            },
            ensure_ascii=False,
        )
        tmp = self._cache_path.with_suffix(self._cache_path.suffix + ".tmp")
        with self._cache_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        if tmp.exists():
            tmp.unlink(missing_ok=True)
