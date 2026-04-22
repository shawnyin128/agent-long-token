from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentdiet.extract_claims import (
    extract_claims_from_message,
    extract_claims_for_dialogue,
    _parse_claims_payload,
)
from agentdiet.llm_client import DummyBackend, LLMClient
from agentdiet.types import Dialogue, Message


MSG_TEXT = "Let me compute 3 + 4 = 7. So the answer is 7. #### 7"


def _client(tmp_path: Path, responder):
    cache = tmp_path / "cache.jsonl"
    return LLMClient(DummyBackend(responder), cache_path=cache)


def test_happy_path_valid_json(tmp_path):
    payload = [
        {"type": "evidence", "text": "3+4=7", "quote": "3 + 4 = 7"},
        {"type": "proposal", "text": "answer is 7", "quote": "#### 7"},
    ]
    client = _client(tmp_path, lambda msgs, m, t: json.dumps(payload))
    m = Message(agent_id=1, round=2, text=MSG_TEXT)
    claims, failed = extract_claims_from_message(
        message=m,
        question="What is 3+4?",
        qid="q42",
        llm_client=client,
        model="test-model",
        failures_dir=tmp_path / "fail",
    )
    assert failed is False
    assert len(claims) == 2
    assert claims[0].type == "evidence"
    assert claims[0].source_message_span == (MSG_TEXT.index("3 + 4 = 7"), MSG_TEXT.index("3 + 4 = 7") + len("3 + 4 = 7"))
    assert claims[1].id == "q42_r2_a1_c1"
    assert client._backend.call_count == 1  # no retry needed


def test_strict_retry_on_malformed_then_success(tmp_path):
    responses = [
        "not json at all",
        json.dumps([{"type": "proposal", "text": "x", "quote": "#### 7"}]),
    ]
    calls = {"n": 0}

    def resp(msgs, m, t):
        i = calls["n"]
        calls["n"] += 1
        return responses[i]

    client = _client(tmp_path, resp)
    m = Message(agent_id=0, round=1, text=MSG_TEXT)
    claims, failed = extract_claims_from_message(
        message=m, question="q", qid="qx", llm_client=client,
        model="t", failures_dir=tmp_path / "fail",
    )
    assert failed is False
    assert len(claims) == 1
    assert calls["n"] == 2


def test_both_attempts_fail_logs_failure_and_returns_empty(tmp_path):
    client = _client(tmp_path, lambda msgs, m, t: "still not json")
    m = Message(agent_id=2, round=3, text=MSG_TEXT)
    claims, failed = extract_claims_from_message(
        message=m, question="q", qid="qZ", llm_client=client,
        model="t", failures_dir=tmp_path / "fail",
    )
    assert failed is True
    assert claims == []
    failure_file = tmp_path / "fail" / "claim_extraction" / "qZ_r3_a2.json"
    assert failure_file.exists()
    payload = json.loads(failure_file.read_text())
    assert payload["qid"] == "qZ"
    assert payload["round"] == 3
    assert payload["agent_id"] == 2
    assert "raw_response" in payload
    assert "first:" in payload["reason"] and "retry:" in payload["reason"]


def test_quote_not_found_drops_that_claim_only(tmp_path):
    payload = [
        {"type": "evidence", "text": "ok", "quote": "3 + 4 = 7"},
        {"type": "proposal", "text": "made up", "quote": "THIS_IS_NOT_IN_THE_MESSAGE"},
        {"type": "proposal", "text": "final", "quote": "#### 7"},
    ]
    client = _client(tmp_path, lambda msgs, m, t: json.dumps(payload))
    m = Message(agent_id=0, round=1, text=MSG_TEXT)
    claims, failed = extract_claims_from_message(
        message=m, question="q", qid="qA", llm_client=client,
        model="t", failures_dir=tmp_path / "fail",
    )
    assert failed is False
    assert len(claims) == 2
    assert all("NOT_IN" not in c.text for c in claims)


def test_invalid_type_enum_triggers_retry(tmp_path):
    responses = [
        json.dumps([{"type": "bogus", "text": "x", "quote": "#### 7"}]),
        json.dumps([{"type": "proposal", "text": "x", "quote": "#### 7"}]),
    ]
    calls = {"n": 0}

    def resp(msgs, m, t):
        i = calls["n"]
        calls["n"] += 1
        return responses[i]

    client = _client(tmp_path, resp)
    m = Message(agent_id=0, round=1, text=MSG_TEXT)
    claims, failed = extract_claims_from_message(
        message=m, question="q", qid="qE", llm_client=client,
        model="t", failures_dir=tmp_path / "fail",
    )
    assert failed is False
    assert len(claims) == 1
    assert calls["n"] == 2


def test_json_with_markdown_fence_is_parsed(tmp_path):
    raw = "```json\n" + json.dumps([{"type": "proposal", "text": "x", "quote": "#### 7"}]) + "\n```"
    client = _client(tmp_path, lambda msgs, m, t: raw)
    m = Message(agent_id=0, round=1, text=MSG_TEXT)
    claims, failed = extract_claims_from_message(
        message=m, question="q", qid="qF", llm_client=client,
        model="t", failures_dir=tmp_path / "fail",
    )
    assert failed is False
    assert len(claims) == 1


def test_non_array_top_level_triggers_retry(tmp_path):
    responses = [
        json.dumps({"type": "proposal", "text": "x", "quote": "#### 7"}),  # object, not array
        json.dumps([{"type": "proposal", "text": "x", "quote": "#### 7"}]),
    ]
    calls = {"n": 0}

    def resp(msgs, m, t):
        out = responses[calls["n"]]
        calls["n"] += 1
        return out

    client = _client(tmp_path, resp)
    m = Message(agent_id=0, round=1, text=MSG_TEXT)
    claims, failed = extract_claims_from_message(
        message=m, question="q", qid="qG", llm_client=client,
        model="t", failures_dir=tmp_path / "fail",
    )
    assert failed is False
    assert len(claims) == 1


def test_extract_for_dialogue_iterates_every_message(tmp_path):
    good = json.dumps([{"type": "proposal", "text": "x", "quote": "#### 7"}])
    client = _client(tmp_path, lambda msgs, m, t: good)
    messages = [
        Message(agent_id=i, round=r, text=MSG_TEXT)
        for r in (1, 2) for i in (0, 1, 2)
    ]
    d = Dialogue(
        question_id="q7", question="Q", gold_answer="7",
        messages=messages, final_answer="7",
    )
    result = extract_claims_for_dialogue(
        dialogue=d, llm_client=client, model="t",
        failures_dir=tmp_path / "fail",
    )
    assert result["qid"] == "q7"
    assert len(result["per_message_status"]) == len(messages)
    assert all(ps["n_claims"] == 1 for ps in result["per_message_status"])
    assert result["extraction_failed"] is False
    # One call per unique message (caching not helpful since text identical but still 1 call
    # per unique prompt). Actually all 6 messages have identical text → cache hits after the first.
    assert client._backend.call_count <= 6  # lax bound


def test_extract_for_dialogue_records_partial_failure(tmp_path):
    def resp(msgs, m, t):
        # Fail only when extracting from agent 1, round 1 (unique marker).
        user = msgs[-1]["content"]
        if "from agent 1, round 1" in user:
            return "not json"
        return json.dumps([{"type": "proposal", "text": "x", "quote": "#### 7"}])

    client = _client(tmp_path, resp)
    messages = [
        Message(agent_id=0, round=1, text=MSG_TEXT),
        Message(agent_id=1, round=1, text=MSG_TEXT),
    ]
    d = Dialogue(
        question_id="q8", question="Q", gold_answer="7",
        messages=messages, final_answer="7",
    )
    result = extract_claims_for_dialogue(
        dialogue=d, llm_client=client, model="t",
        failures_dir=tmp_path / "fail",
    )
    assert result["extraction_failed"] is True
    ps = {(s["agent_id"], s["round"]): s for s in result["per_message_status"]}
    assert ps[(0, 1)]["extraction_failed"] is False
    assert ps[(1, 1)]["extraction_failed"] is True
    assert (tmp_path / "fail" / "claim_extraction" / "q8_r1_a1.json").exists()


def test_parse_claims_payload_empty_array_returns_empty_no_error():
    claims, err = _parse_claims_payload(
        "[]", MSG_TEXT, "qid", 0, 1, 0
    )
    assert claims == []
    assert err is None


def test_claim_ids_are_sequential_within_a_message(tmp_path):
    payload = [
        {"type": "proposal", "text": "a", "quote": "3 + 4 = 7"},
        {"type": "proposal", "text": "b", "quote": "#### 7"},
        {"type": "evidence", "text": "c", "quote": "the answer is 7"},
    ]
    client = _client(tmp_path, lambda msgs, m, t: json.dumps(payload))
    m = Message(agent_id=0, round=1, text=MSG_TEXT)
    claims, _ = extract_claims_from_message(
        message=m, question="q", qid="qi", llm_client=client,
        model="t", failures_dir=tmp_path / "fail",
    )
    ids = [c.id for c in claims]
    assert ids == ["qi_r1_a0_c0", "qi_r1_a0_c1", "qi_r1_a0_c2"]
