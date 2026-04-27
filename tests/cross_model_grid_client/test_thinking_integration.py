"""Integration: thinking toggle stores 2 cache entries; cache hit is
deterministic; round-trip preserves response identity."""
from __future__ import annotations

from agentdiet.llm_client import DummyBackend, LLMClient


MSG = [{"role": "user", "content": "hi"}]


def test_toggle_round_trip_stores_two_entries(tmp_path):
    """thinking=True and thinking=False populate distinct cache slots
    and reading them back returns the right response per toggle."""
    seq = iter(["thinking-on response", "thinking-off response"])
    backend = DummyBackend(lambda m, mo, t, **kw: next(seq))
    client = LLMClient(backend, cache_path=tmp_path / "cache.jsonl")

    on = client.chat_full(MSG, "m", 0.0, thinking=True)
    off = client.chat_full(MSG, "m", 0.0, thinking=False)

    # Distinct entries — both fresh
    assert client.cache_hits == 0
    assert on.response == "thinking-on response"
    assert off.response == "thinking-off response"

    # Now hit each — must return the right one
    on_hit = client.chat_full(MSG, "m", 0.0, thinking=True)
    off_hit = client.chat_full(MSG, "m", 0.0, thinking=False)
    assert on_hit.response == "thinking-on response"
    assert off_hit.response == "thinking-off response"
    assert client.cache_hits == 2


def test_toggle_persists_across_client_reload(tmp_path):
    """The on-disk cache must survive a fresh LLMClient and serve
    the right response per toggle."""
    seq = iter(["A", "B"])
    backend = DummyBackend(lambda m, mo, t, **kw: next(seq))
    cache_path = tmp_path / "cache.jsonl"
    client1 = LLMClient(backend, cache_path=cache_path)
    client1.chat_full(MSG, "m", 0.0, thinking=True)
    client1.chat_full(MSG, "m", 0.0, thinking=False)

    # Fresh client, same on-disk cache
    backend2 = DummyBackend(lambda m, mo, t, **kw: "SHOULD_NOT_BE_CALLED")
    client2 = LLMClient(backend2, cache_path=cache_path)
    on = client2.chat_full(MSG, "m", 0.0, thinking=True)
    off = client2.chat_full(MSG, "m", 0.0, thinking=False)
    assert on.response == "A"
    assert off.response == "B"
    assert backend2.call_count == 0


def test_legacy_cache_entries_remain_valid(tmp_path):
    """An entry written under the legacy 3-arg cache_key path must
    resolve when read with thinking=False, top_p=1.0 defaults."""
    import json

    from agentdiet.llm_client import cache_key

    cache_path = tmp_path / "cache.jsonl"
    legacy_key = cache_key("m", 0.0, MSG)  # 3-arg form
    with cache_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "key": legacy_key,
            "model": "m",
            "temperature": 0.0,
            "messages": MSG,
            "response": "legacy-cached-response",
        }) + "\n")

    backend = DummyBackend(lambda m, mo, t, **kw: "SHOULD_NOT_BE_CALLED")
    client = LLMClient(backend, cache_path=cache_path)
    out = client.chat_full(MSG, "m", 0.0)  # default thinking=False, top_p=1.0
    assert out.response == "legacy-cached-response"
    assert client.cache_hits == 1
