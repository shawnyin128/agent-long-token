"""Cache-key + ChatResult + per-family translation tests for llm_client."""
from __future__ import annotations

import warnings
from typing import Any
from unittest.mock import MagicMock

import pytest

from agentdiet.llm_client import (
    ChatResult,
    DummyBackend,
    LLMClient,
    OpenAIBackend,
    cache_key,
)


MSG = [{"role": "user", "content": "hi"}]


def test_cache_key_default_args_match_legacy():
    """Legacy 3-arg call equals the new 5-arg call with thinking=False, top_p=1.0."""
    legacy = cache_key("m", 0.0, MSG)
    new = cache_key("m", 0.0, MSG, thinking=False, top_p=1.0)
    assert legacy == new


def test_cache_key_differs_by_thinking():
    a = cache_key("m", 0.0, MSG, thinking=False, top_p=1.0)
    b = cache_key("m", 0.0, MSG, thinking=True, top_p=1.0)
    assert a != b


def test_cache_key_differs_by_top_p():
    a = cache_key("m", 0.7, MSG, thinking=False, top_p=1.0)
    b = cache_key("m", 0.7, MSG, thinking=False, top_p=0.95)
    assert a != b


def test_chat_full_returns_chatresult_with_usage(tmp_path):
    """DummyBackend can report usage; chat_full should surface it."""

    class UsageBackend:
        call_count = 0

        def chat(self, messages, model, temperature, **kwargs):
            self.call_count += 1
            return "OK"

        def chat_full(self, messages, model, temperature, **kwargs):
            self.call_count += 1
            return ChatResult(response="OK", prompt_tokens=12, completion_tokens=3)

    client = LLMClient(UsageBackend(), cache_path=tmp_path / "cache.jsonl")
    result = client.chat_full(MSG, "m", 0.0)
    assert isinstance(result, ChatResult)
    assert result.response == "OK"
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 3


def test_chat_full_cache_hit_fills_approx_tokens(tmp_path):
    """Second call must be a cache hit and still return non-None tokens."""
    backend = DummyBackend(lambda m, mo, t: "hello world response")
    client = LLMClient(backend, cache_path=tmp_path / "cache.jsonl")
    r1 = client.chat_full(MSG, "m", 0.0)
    r2 = client.chat_full(MSG, "m", 0.0)
    assert r1.response == r2.response
    assert client.cache_hits == 1
    assert r2.prompt_tokens is not None
    assert r2.completion_tokens is not None
    # char-approx: TOKENS_PER_CHAR=0.25, "hello world response" len=20 -> 5
    assert r2.completion_tokens == 5


def test_chat_full_thinking_flag_is_part_of_cache_key(tmp_path):
    """thinking=True and thinking=False are distinct cache slices."""
    seq = iter(["A", "B"])
    backend = DummyBackend(lambda m, mo, t, **kw: next(seq))
    client = LLMClient(backend, cache_path=tmp_path / "cache.jsonl")
    a = client.chat_full(MSG, "m", 0.0, thinking=False)
    b = client.chat_full(MSG, "m", 0.0, thinking=True)
    assert a.response == "A"
    assert b.response == "B"
    # 2 distinct entries cached
    assert client.cache_hits == 0


def test_chat_full_top_p_is_part_of_cache_key(tmp_path):
    seq = iter(["A", "B"])
    backend = DummyBackend(lambda m, mo, t, **kw: next(seq))
    client = LLMClient(backend, cache_path=tmp_path / "cache.jsonl")
    a = client.chat_full(MSG, "m", 0.7, top_p=1.0)
    b = client.chat_full(MSG, "m", 0.7, top_p=0.95)
    assert a.response == "A"
    assert b.response == "B"


def test_legacy_chat_str_signature_still_works(tmp_path):
    """Existing callers using chat(...) -> str must keep working."""
    backend = DummyBackend(lambda m, mo, t: "STR")
    client = LLMClient(backend, cache_path=tmp_path / "cache.jsonl")
    out = client.chat(MSG, "m", 0.0)
    assert isinstance(out, str)
    assert out == "STR"


# --- OpenAIBackend translation table ----------------------------------------


def _patch_openai_completions(monkeypatch, captured_kwargs: dict):
    """Patch OpenAI client so OpenAIBackend constructs but the API call
    is captured into the dict passed in."""
    import openai

    class _FakeChatCompletions:
        def create(self, **kwargs):
            captured_kwargs.update(kwargs)
            usage = MagicMock()
            usage.prompt_tokens = 7
            usage.completion_tokens = 4
            choice = MagicMock()
            choice.message.content = "ok"
            resp = MagicMock()
            resp.choices = [choice]
            resp.usage = usage
            return resp

    class _FakeClient:
        def __init__(self, *a, **kw):
            self.chat = MagicMock()
            self.chat.completions = _FakeChatCompletions()

    monkeypatch.setattr(openai, "OpenAI", _FakeClient)


def test_openai_backend_qwen3_thinking_true_sets_chat_template_kwargs(monkeypatch):
    captured: dict[str, Any] = {}
    _patch_openai_completions(monkeypatch, captured)
    backend = OpenAIBackend(
        base_url="http://x", api_key="k", model_family="qwen3"
    )
    backend.chat(MSG, "m", 0.0, thinking=True)
    assert captured.get("extra_body") == {
        "chat_template_kwargs": {"enable_thinking": True}
    }


def test_openai_backend_qwen3_thinking_false_sets_disable(monkeypatch):
    captured: dict[str, Any] = {}
    _patch_openai_completions(monkeypatch, captured)
    backend = OpenAIBackend(
        base_url="http://x", api_key="k", model_family="qwen3"
    )
    backend.chat(MSG, "m", 0.0, thinking=False)
    assert captured.get("extra_body") == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_openai_backend_gpt_oss_thinking_true_sets_high(monkeypatch):
    captured: dict[str, Any] = {}
    _patch_openai_completions(monkeypatch, captured)
    backend = OpenAIBackend(
        base_url="http://x", api_key="k", model_family="gpt-oss"
    )
    backend.chat(MSG, "m", 0.0, thinking=True)
    assert captured.get("reasoning_effort") == "high"


def test_openai_backend_gpt_oss_thinking_false_sets_low(monkeypatch):
    captured: dict[str, Any] = {}
    _patch_openai_completions(monkeypatch, captured)
    backend = OpenAIBackend(
        base_url="http://x", api_key="k", model_family="gpt-oss"
    )
    backend.chat(MSG, "m", 0.0, thinking=False)
    assert captured.get("reasoning_effort") == "low"


def test_openai_backend_generic_warns_on_thinking_true(monkeypatch):
    captured: dict[str, Any] = {}
    _patch_openai_completions(monkeypatch, captured)
    backend = OpenAIBackend(
        base_url="http://x", api_key="k", model_family="generic"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        backend.chat(MSG, "m", 0.0, thinking=True)
    assert any("thinking" in str(w.message).lower() for w in caught)
    # generic must not pass either translation key
    assert "extra_body" not in captured or not captured.get("extra_body")
    assert "reasoning_effort" not in captured


def test_openai_backend_default_family_is_generic(monkeypatch):
    captured: dict[str, Any] = {}
    _patch_openai_completions(monkeypatch, captured)
    backend = OpenAIBackend(base_url="http://x", api_key="k")
    # back-compat: thinking=False default, no translation, no warning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        backend.chat(MSG, "m", 0.0)
    assert not any("thinking" in str(w.message).lower() for w in caught)


def test_openai_backend_chat_full_returns_real_usage(monkeypatch):
    captured: dict[str, Any] = {}
    _patch_openai_completions(monkeypatch, captured)
    backend = OpenAIBackend(base_url="http://x", api_key="k")
    r = backend.chat_full(MSG, "m", 0.0)
    assert isinstance(r, ChatResult)
    assert r.response == "ok"
    assert r.prompt_tokens == 7
    assert r.completion_tokens == 4


def test_openai_backend_top_p_only_when_temp_nonzero(monkeypatch):
    captured: dict[str, Any] = {}
    _patch_openai_completions(monkeypatch, captured)
    backend = OpenAIBackend(base_url="http://x", api_key="k")
    # temp=0 — top_p must NOT be in payload (no-op anyway)
    backend.chat(MSG, "m", 0.0, top_p=0.95)
    assert "top_p" not in captured

    captured.clear()
    backend.chat(MSG, "m", 0.7, top_p=0.95)
    assert captured.get("top_p") == 0.95


# --- DummyBackend stays usable ----------------------------------------------


def test_dummy_backend_accepts_thinking_kwarg():
    """Existing DummyBackend(responder) callers without **kw should keep working."""
    backend = DummyBackend(lambda m, mo, t: "X")
    out = backend.chat(MSG, "m", 0.0)  # legacy call
    assert out == "X"


def test_dummy_backend_thinking_passes_through_when_responder_accepts_it():
    seen: list[bool] = []

    def responder(messages, model, temperature, *, thinking=False, **kw):
        seen.append(thinking)
        return "OK"

    backend = DummyBackend(responder)
    backend.chat(MSG, "m", 0.0, thinking=True)
    backend.chat(MSG, "m", 0.0, thinking=False)
    assert seen == [True, False]
