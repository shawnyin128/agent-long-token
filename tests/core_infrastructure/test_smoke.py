from __future__ import annotations

import time

import pytest

from agentdiet.dataset import parse_answer
from agentdiet.llm_client import DummyBackend, LLMClient


pytestmark = [pytest.mark.smoke, pytest.mark.timeout(30)]


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    """Poison the real OpenAI client so any accidental instantiation fails."""
    import openai

    orig = openai.OpenAI

    class Guard:
        def __init__(self, *a, **k):
            raise RuntimeError("Smoke tests must not hit the network")

    monkeypatch.setattr(openai, "OpenAI", Guard)
    yield
    monkeypatch.setattr(openai, "OpenAI", orig)


def test_cache_roundtrip(tmp_path):
    calls = {"n": 0}

    def responder(msgs, model, temp):
        calls["n"] += 1
        return "dummy-reply"

    backend = DummyBackend(responder)
    client = LLMClient(backend, cache_path=tmp_path / "cache.jsonl")
    msg = [{"role": "user", "content": "hello"}]
    a = client.chat(msg, "m", 0.0)
    b = client.chat(msg, "m", 0.0)
    assert a == b == "dummy-reply"
    assert calls["n"] == 1


def test_dummy_chat_returns_nonempty_string(tmp_path):
    backend = DummyBackend(lambda m, mo, t: "hello world")
    client = LLMClient(backend, cache_path=tmp_path / "cache.jsonl")
    r = client.chat([{"role": "user", "content": "x"}], "m", 0.0)
    assert isinstance(r, str) and r


def test_parser_five_variants():
    assert parse_answer("#### 42") == "42"
    assert parse_answer("42.0") == "42"
    assert parse_answer("$42") == "42"
    assert parse_answer("he earned 42 dollars") == "42"
    assert parse_answer("the answer is 42.") == "42"


def test_total_walltime_under_budget(tmp_path):
    """End-to-end smoke is <5s even with 100 dummy calls."""
    backend = DummyBackend(lambda m, mo, t: "x")
    client = LLMClient(backend, cache_path=tmp_path / "cache.jsonl")
    t0 = time.time()
    for i in range(100):
        client.chat([{"role": "user", "content": str(i)}], "m", 0.0)
    assert time.time() - t0 < 5.0
