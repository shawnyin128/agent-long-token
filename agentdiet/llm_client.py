from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Protocol

log = logging.getLogger(__name__)


TOKENS_PER_CHAR = 0.25  # matches agentdiet.evaluate.count_tokens
ModelFamily = Literal["qwen3", "gpt-oss", "generic"]


@dataclass(frozen=True)
class ChatResult:
    response: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]


def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    return int(round(len(text) * TOKENS_PER_CHAR))


def _approx_prompt_tokens(messages: list[dict]) -> int:
    total = sum(len(m.get("content", "") or "") for m in messages)
    return _approx_tokens("x" * total)  # reuse rounding


def cache_key(
    model: str,
    temperature: float,
    messages: list[dict],
    *,
    thinking: bool = False,
    top_p: float = 1.0,
) -> str:
    """Deterministic cache key.

    The legacy 3-arg form (no thinking, no top_p) hashes the same payload
    as the new form with the documented defaults — existing on-disk
    cache entries stay valid. Non-default values are appended to the
    payload so they participate in the hash.
    """
    parts = [
        model,
        str(temperature),
        json.dumps(messages, sort_keys=True, ensure_ascii=False),
    ]
    if thinking:
        parts.append("thinking=True")
    if top_p != 1.0:
        parts.append(f"top_p={top_p}")
    payload = "\n".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Backend(Protocol):
    def chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        *,
        thinking: bool = False,
        top_p: float = 1.0,
    ) -> str: ...

    def chat_full(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        *,
        thinking: bool = False,
        top_p: float = 1.0,
    ) -> ChatResult: ...


class DummyBackend:
    def __init__(self, responder: Callable[..., str] | None = None):
        self._responder = responder or (lambda msgs, m, t: "DUMMY_RESPONSE")
        self.call_count = 0
        self.last_kwargs: dict[str, Any] = {}

    def chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        *,
        thinking: bool = False,
        top_p: float = 1.0,
    ) -> str:
        self.call_count += 1
        self.last_kwargs = {"thinking": thinking, "top_p": top_p}
        try:
            return self._responder(
                messages, model, temperature, thinking=thinking, top_p=top_p
            )
        except TypeError:
            return self._responder(messages, model, temperature)

    def chat_full(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        *,
        thinking: bool = False,
        top_p: float = 1.0,
    ) -> ChatResult:
        text = self.chat(
            messages, model, temperature, thinking=thinking, top_p=top_p
        )
        return ChatResult(response=text, prompt_tokens=None, completion_tokens=None)


class OpenAIBackend:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_s: float = 120.0,
        model_family: ModelFamily = "generic",
    ):
        from openai import OpenAI
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)
        self._model_family: ModelFamily = model_family
        self._warned_generic_thinking = False

    def chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        *,
        thinking: bool = False,
        top_p: float = 1.0,
    ) -> str:
        return self.chat_full(
            messages, model, temperature, thinking=thinking, top_p=top_p
        ).response

    def chat_full(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        *,
        thinking: bool = False,
        top_p: float = 1.0,
    ) -> ChatResult:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if temperature > 0 and top_p != 1.0:
            kwargs["top_p"] = top_p

        if self._model_family == "qwen3":
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": thinking}
            }
        elif self._model_family == "gpt-oss":
            kwargs["reasoning_effort"] = "high" if thinking else "low"
        else:  # generic
            if thinking and not self._warned_generic_thinking:
                warnings.warn(
                    "OpenAIBackend(model_family='generic') does not support "
                    "thinking; ignoring the flag",
                    stacklevel=2,
                )
                self._warned_generic_thinking = True

        resp = self._client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = (
            getattr(usage, "completion_tokens", None) if usage else None
        )
        return ChatResult(
            response=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


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
        # Lock guards _cache mutations + cache file appends + counters when
        # multiple threads issue chat_full concurrently. The OpenAI SDK
        # client itself is thread-safe; only our cache layer needs guarding.
        self._lock = threading.Lock()

    def chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.0,
        *,
        thinking: bool = False,
        top_p: float = 1.0,
    ) -> str:
        return self.chat_full(
            messages, model, temperature, thinking=thinking, top_p=top_p
        ).response

    def chat_full(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.0,
        *,
        thinking: bool = False,
        top_p: float = 1.0,
    ) -> ChatResult:
        key = cache_key(
            model, temperature, messages, thinking=thinking, top_p=top_p
        )
        with self._lock:
            if key in self._cache:
                self.cache_hits += 1
                text = self._cache[key]
                return ChatResult(
                    response=text,
                    prompt_tokens=_approx_prompt_tokens(messages),
                    completion_tokens=_approx_tokens(text),
                )

        # Backend call happens OUTSIDE the lock so multiple threads can
        # issue concurrent requests to vLLM in parallel.
        result = self._call_full_with_retry(
            messages, model, temperature, thinking=thinking, top_p=top_p
        )
        with self._lock:
            # Re-check the cache: another thread may have populated this key
            # while we were waiting on the backend.
            if key in self._cache:
                self.cache_hits += 1
                text = self._cache[key]
                # Discard our just-fetched result; return the cached one for
                # determinism. (At temp>0 this picks the first arrival;
                # at temp=0 they should be identical.)
                return ChatResult(
                    response=text,
                    prompt_tokens=_approx_prompt_tokens(messages),
                    completion_tokens=_approx_tokens(text),
                )
            self._append_cache(key, model, temperature, messages, result.response)
            self._cache[key] = result.response
            self.call_count += 1

        if result.prompt_tokens is None or result.completion_tokens is None:
            return ChatResult(
                response=result.response,
                prompt_tokens=result.prompt_tokens
                if result.prompt_tokens is not None
                else _approx_prompt_tokens(messages),
                completion_tokens=result.completion_tokens
                if result.completion_tokens is not None
                else _approx_tokens(result.response),
            )
        return result

    def _call_full_with_retry(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        *,
        thinking: bool,
        top_p: float,
    ) -> ChatResult:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return self._invoke_backend(
                    messages, model, temperature,
                    thinking=thinking, top_p=top_p,
                )
            except Exception as exc:
                last_exc = exc
                if attempt == self._max_retries - 1:
                    break
                wait = self._base_backoff_s * (4 ** attempt)
                log.warning(
                    "LLM call failed (attempt %d/%d): %s; retry in %.1fs",
                    attempt + 1, self._max_retries, exc, wait,
                )
                time.sleep(wait)
        assert last_exc is not None
        raise last_exc

    def _invoke_backend(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        *,
        thinking: bool,
        top_p: float,
    ) -> ChatResult:
        """Call backend.chat_full when present, else fall back to chat().
        Falls back further to chat(messages, model, temperature) for
        legacy backends that don't accept thinking/top_p kwargs."""
        chat_full = getattr(self._backend, "chat_full", None)
        if chat_full is not None:
            try:
                return chat_full(
                    messages, model, temperature,
                    thinking=thinking, top_p=top_p,
                )
            except TypeError:
                # backend.chat_full has a stricter signature
                return chat_full(messages, model, temperature)

        try:
            text = self._backend.chat(
                messages, model, temperature,
                thinking=thinking, top_p=top_p,
            )
        except TypeError:
            text = self._backend.chat(messages, model, temperature)
        return ChatResult(response=text, prompt_tokens=None, completion_tokens=None)

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
