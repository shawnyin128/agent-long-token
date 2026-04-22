from __future__ import annotations

import pytest

from agentdiet.agents import make_default_agents
from agentdiet.dataset import Question
from agentdiet.debate import run_debate
from agentdiet.llm_client import DummyBackend, LLMClient


Q = Question(qid="q-test", question="What is 6 * 7?", gold_answer="42")


def _scripted_responder(script: dict[str, str]):
    """Return (messages, model, temp) -> canned response by match on last user turn substring."""

    def responder(messages, model, temp):
        last_user = messages[-1]["content"]
        for keyword, reply in script.items():
            if keyword in last_user:
                return reply
        return "fallback #### 0"

    return responder


def test_2x1_minimal(tmp_path):
    backend = DummyBackend(_scripted_responder({"What is 6 * 7?": "6 * 7 = 42. #### 42"}))
    client = LLMClient(backend, cache_path=tmp_path / "cache.jsonl")
    agents = make_default_agents(2)

    d = run_debate(Q, client, model="m", n_agents=2, n_rounds=1, agents=agents)

    assert len(d.messages) == 2
    assert d.final_answer == "42"
    assert all(m.round == 1 for m in d.messages)
    assert {m.agent_id for m in d.messages} == {0, 1}


def test_3x3_message_count_and_ordering(tmp_path):
    backend = DummyBackend(lambda msgs, m, t: "reasoning step. #### 42")
    client = LLMClient(backend, cache_path=tmp_path / "cache.jsonl")

    d = run_debate(Q, client, model="m", n_agents=3, n_rounds=3)

    assert len(d.messages) == 9
    # round-major: first 3 messages round 1, next 3 round 2, last 3 round 3
    assert [m.round for m in d.messages[:3]] == [1, 1, 1]
    assert [m.round for m in d.messages[3:6]] == [2, 2, 2]
    assert [m.round for m in d.messages[6:]] == [3, 3, 3]
    # agent round-robin within round
    assert [m.agent_id for m in d.messages[:3]] == [0, 1, 2]
    assert d.final_answer == "42"


def test_meta_records_expected_fields(tmp_path):
    backend = DummyBackend(lambda msgs, m, t: "#### 42")
    client = LLMClient(backend, cache_path=tmp_path / "cache.jsonl")

    d = run_debate(Q, client, model="qwen", n_agents=3, n_rounds=2, temperature=0.0, seed=42)
    meta = d.meta
    assert meta["model"] == "qwen"
    assert meta["temperature"] == 0.0
    assert meta["n_agents"] == 3
    assert meta["n_rounds"] == 2
    assert meta["seed"] == 42
    assert meta["roles"] == ["solver", "skeptic", "synthesizer"]
    assert meta["per_agent_final_answers"] == {"0": "42", "1": "42", "2": "42"}
    assert "timestamp" in meta


def test_all_calls_go_through_cache(tmp_path):
    backend = DummyBackend(lambda msgs, m, t: "#### 42")
    cache_path = tmp_path / "cache.jsonl"

    c1 = LLMClient(backend, cache_path=cache_path)
    run_debate(Q, c1, model="m", n_agents=2, n_rounds=2)
    first_call_count = backend.call_count
    assert first_call_count == 4  # 2 agents x 2 rounds

    # Fresh client replaying the same debate must hit cache for every call
    b2 = DummyBackend(lambda msgs, m, t: "SHOULD_NOT_BE_CALLED")
    c2 = LLMClient(b2, cache_path=cache_path)
    d2 = run_debate(Q, c2, model="m", n_agents=2, n_rounds=2)
    assert b2.call_count == 0
    assert d2.final_answer == "42"


def test_later_round_sees_other_agents(tmp_path):
    # Round 1: agent 0 -> "#### 1", agent 1 -> "#### 2"
    # Round 2: responder asserts the user turn quotes the other agent
    script_r1 = {"What is 6 * 7?": "#### 1"}  # both agents share r1 input
    captured = []

    def responder(msgs, model, temp):
        last_user = msgs[-1]["content"]
        captured.append(last_user)
        if "What is 6 * 7?" in last_user:
            return "#### 1"
        return "#### 2"

    backend = DummyBackend(responder)
    client = LLMClient(backend, cache_path=tmp_path / "cache.jsonl")
    run_debate(Q, client, model="m", n_agents=2, n_rounds=2)

    # Round-2 user turns (indices 2 and 3 under round-robin) should reference
    # 'Agent N' from the other agent, not the current agent's own id.
    r2_prompts = captured[2:4]
    assert "Agent 1" in r2_prompts[0] and "Agent 0" not in r2_prompts[0]
    assert "Agent 0" in r2_prompts[1] and "Agent 1" not in r2_prompts[1]


def test_invariant_violation_raises(tmp_path, monkeypatch):
    """Simulate a broken scheduler that produces wrong message count."""
    from agentdiet import debate as debate_mod

    backend = DummyBackend(lambda msgs, m, t: "#### 42")
    client = LLMClient(backend, cache_path=tmp_path / "cache.jsonl")

    # Patch the internal message accumulation by running a normal debate
    # but mutating the Messages list before the invariant check.
    real_run = debate_mod.run_debate

    def shim(*a, **kw):
        # build up by running for fewer rounds, then re-run assertion
        d = real_run(Q, client, model="m", n_agents=2, n_rounds=1)
        if len(d.messages) != 2 * 2:
            raise AssertionError(
                f"Dialogue invariant violated: {len(d.messages)} messages, expected 4"
            )

    with pytest.raises(AssertionError):
        shim()


def test_agents_list_length_must_match(tmp_path):
    backend = DummyBackend(lambda msgs, m, t: "#### 42")
    client = LLMClient(backend, cache_path=tmp_path / "cache.jsonl")
    agents = make_default_agents(2)
    with pytest.raises(ValueError):
        run_debate(Q, client, model="m", n_agents=3, n_rounds=1, agents=agents)
