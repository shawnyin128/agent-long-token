from __future__ import annotations

from pathlib import Path

import pytest

from agentdiet.compress import Policy
from agentdiet.evaluate import evaluate_method_on_qid
from agentdiet.llm_client import DummyBackend, LLMClient
from agentdiet.types import Dialogue, Message


def _client(tmp_path, responder) -> LLMClient:
    return LLMClient(DummyBackend(responder), cache_path=tmp_path / "c.jsonl")


def _dialogue() -> Dialogue:
    return Dialogue(
        question_id="q", question="What is 3+4?", gold_answer="7",
        messages=[
            Message(agent_id=0, round=1, text="I compute 3+4=7. #### 7"),
            Message(agent_id=1, round=1, text="Agree. #### 7"),
            Message(agent_id=0, round=2, text="#### 7"),
            Message(agent_id=1, round=2, text="#### 7"),
        ],
        final_answer="7",
    )


def _claims(d: Dialogue) -> dict:
    return {"qid": d.question_id, "claims": [
        {"id": f"{d.question_id}_r1_a0_c0", "text": "e",
         "agent_id": 0, "round": 1, "type": "evidence",
         "source_message_span": [2, 15]},
    ], "per_message_status": [], "extraction_failed": False}


def test_happy_path_b1(tmp_path):
    d = _dialogue()
    client = _client(tmp_path, lambda m, mo, t: "#### 7")
    r = evaluate_method_on_qid(
        dialogue=d, claims_doc=_claims(d), signal_scores=None,
        method="b1", policy=Policy(mode="b1"),
        llm_client=client, model="m",
    )
    assert r.method == "b1"
    assert r.correct is True
    assert r.compressed_tokens > 0


def test_wrong_answer_returns_correct_false(tmp_path):
    d = _dialogue()
    client = _client(tmp_path, lambda m, mo, t: "#### 3")
    r = evaluate_method_on_qid(
        dialogue=d, claims_doc=_claims(d), signal_scores=None,
        method="b1", policy=Policy(mode="b1"),
        llm_client=client, model="m",
    )
    assert r.correct is False
    assert r.final_answer == "3"


def test_ours_without_claims_doc_raises(tmp_path):
    d = _dialogue()
    client = _client(tmp_path, lambda m, mo, t: "#### 7")
    with pytest.raises(ValueError):
        evaluate_method_on_qid(
            dialogue=d, claims_doc=None, signal_scores=None,
            method="ours", policy=Policy(mode="ours", drop_types=["agreement"]),
            llm_client=client, model="m",
        )


def test_b2_produces_fewer_tokens_than_b1(tmp_path):
    d = _dialogue()
    client = _client(tmp_path, lambda m, mo, t: "#### 7")
    r_b1 = evaluate_method_on_qid(
        dialogue=d, claims_doc=_claims(d), signal_scores=None,
        method="b1", policy=Policy(mode="b1"),
        llm_client=client, model="m",
    )
    r_b2 = evaluate_method_on_qid(
        dialogue=d, claims_doc=_claims(d), signal_scores=None,
        method="b2", policy=Policy(mode="b2"),
        llm_client=client, model="m",
    )
    assert r_b2.compressed_tokens < r_b1.compressed_tokens


def test_cache_hit_on_repeat(tmp_path):
    d = _dialogue()
    client = _client(tmp_path, lambda m, mo, t: "#### 7")
    evaluate_method_on_qid(
        dialogue=d, claims_doc=_claims(d), signal_scores=None,
        method="b1", policy=Policy(mode="b1"),
        llm_client=client, model="m",
    )
    first = client.call_count
    evaluate_method_on_qid(
        dialogue=d, claims_doc=_claims(d), signal_scores=None,
        method="b1", policy=Policy(mode="b1"),
        llm_client=client, model="m",
    )
    assert client.call_count == first


def test_unparseable_response_yields_final_none_and_correct_false(tmp_path):
    d = _dialogue()
    client = _client(tmp_path, lambda m, mo, t: "I have no final answer.")
    r = evaluate_method_on_qid(
        dialogue=d, claims_doc=_claims(d), signal_scores=None,
        method="b1", policy=Policy(mode="b1"),
        llm_client=client, model="m",
    )
    assert r.final_answer is None
    assert r.correct is False
