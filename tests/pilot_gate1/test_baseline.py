from __future__ import annotations

from agentdiet.baseline import run_single_agent
from agentdiet.dataset import Question
from agentdiet.llm_client import DummyBackend, LLMClient


Q = Question(qid="q1", question="What is 6 * 7?", gold_answer="42")


def test_produces_one_message(tmp_path):
    backend = DummyBackend(lambda m, mo, t: "6*7 = 42. #### 42")
    client = LLMClient(backend, cache_path=tmp_path / "cache.jsonl")
    d = run_single_agent(Q, client, model="m")
    assert len(d.messages) == 1
    assert d.messages[0].agent_id == 0
    assert d.messages[0].round == 1


def test_final_answer_parsed(tmp_path):
    backend = DummyBackend(lambda m, mo, t: "#### 42")
    client = LLMClient(backend, cache_path=tmp_path / "cache.jsonl")
    d = run_single_agent(Q, client, model="m")
    assert d.final_answer == "42"


def test_cache_hit_on_second_call(tmp_path):
    backend = DummyBackend(lambda m, mo, t: "#### 42")
    client = LLMClient(backend, cache_path=tmp_path / "cache.jsonl")
    run_single_agent(Q, client, model="m")
    run_single_agent(Q, client, model="m")
    assert backend.call_count == 1


def test_meta_method_single(tmp_path):
    backend = DummyBackend(lambda m, mo, t: "#### 42")
    client = LLMClient(backend, cache_path=tmp_path / "cache.jsonl")
    d = run_single_agent(Q, client, model="m")
    assert d.meta["method"] == "single"
    assert d.meta["model"] == "m"
    assert "timestamp" in d.meta


def test_unparseable_response_yields_none(tmp_path):
    backend = DummyBackend(lambda m, mo, t: "no number mentioned")
    client = LLMClient(backend, cache_path=tmp_path / "cache.jsonl")
    d = run_single_agent(Q, client, model="m")
    assert d.final_answer is None
    assert len(d.messages) == 1  # message still recorded
