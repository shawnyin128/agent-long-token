from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentdiet.llm_client import DummyBackend, LLMClient, cache_key


MSG = [{"role": "user", "content": "hi"}]


def test_cache_key_stable_across_ordering():
    a = cache_key("m", 0.0, [{"role": "user", "content": "x"}])
    b = cache_key("m", 0.0, [{"content": "x", "role": "user"}])
    assert a == b


def test_cache_key_differs_by_model():
    assert cache_key("m1", 0.0, MSG) != cache_key("m2", 0.0, MSG)


def test_cache_key_differs_by_temperature():
    assert cache_key("m", 0.0, MSG) != cache_key("m", 0.7, MSG)


def test_miss_then_hit(tmp_path):
    backend = DummyBackend(lambda m, mo, t: "R1")
    client = LLMClient(backend, cache_path=tmp_path / "cache.jsonl")
    r1 = client.chat(MSG, "m", 0.0)
    r2 = client.chat(MSG, "m", 0.0)
    assert r1 == r2 == "R1"
    assert backend.call_count == 1
    assert client.cache_hits == 1


def test_different_model_causes_miss(tmp_path):
    backend = DummyBackend(lambda m, mo, t: f"R-{mo}")
    client = LLMClient(backend, cache_path=tmp_path / "cache.jsonl")
    client.chat(MSG, "m1", 0.0)
    client.chat(MSG, "m2", 0.0)
    assert backend.call_count == 2


def test_cache_persists_across_instances(tmp_path):
    p = tmp_path / "cache.jsonl"
    b1 = DummyBackend(lambda m, mo, t: "R1")
    c1 = LLMClient(b1, cache_path=p)
    c1.chat(MSG, "m", 0.0)

    b2 = DummyBackend(lambda m, mo, t: "SHOULD_NOT_BE_CALLED")
    c2 = LLMClient(b2, cache_path=p)
    r = c2.chat(MSG, "m", 0.0)
    assert r == "R1"
    assert b2.call_count == 0


def test_truncated_trailing_line_dropped(tmp_path):
    p = tmp_path / "cache.jsonl"
    good = {"key": "K1", "model": "m", "temperature": 0.0, "messages": MSG, "response": "R1"}
    p.write_text(json.dumps(good) + "\n" + '{"key": "K2", "respon', encoding="utf-8")

    backend = DummyBackend(lambda m, mo, t: "RNEW")
    client = LLMClient(backend, cache_path=p)
    assert "K1" in client._cache
    assert "K2" not in client._cache


def test_retry_then_fail(tmp_path):
    calls = {"n": 0}

    def boom(msgs, model, temp):
        calls["n"] += 1
        raise RuntimeError("nope")

    backend = DummyBackend(boom)
    client = LLMClient(
        backend,
        cache_path=tmp_path / "cache.jsonl",
        max_retries=3,
        base_backoff_s=0.0,
    )
    with pytest.raises(RuntimeError):
        client.chat(MSG, "m", 0.0)
    assert calls["n"] == 3


def test_retry_then_succeed(tmp_path):
    calls = {"n": 0}

    def flaky(msgs, model, temp):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("transient")
        return "OK"

    backend = DummyBackend(flaky)
    client = LLMClient(
        backend,
        cache_path=tmp_path / "cache.jsonl",
        max_retries=3,
        base_backoff_s=0.0,
    )
    assert client.chat(MSG, "m", 0.0) == "OK"
    assert calls["n"] == 2
