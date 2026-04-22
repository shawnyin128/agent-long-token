"""Adversarial (Round-1 red-team) tests for claim-extraction.

Probes the specific concerns Generator surfaced in flags_for_eval:

  * multi-byte character span correctness
  * resume correctness for "partial" outcomes
  * spot-check RNG uses cfg.seed, not a global random.seed
  * every parse-failure branch in _parse_claims_payload is exercised
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from agentdiet.cli.extract import run_extraction
from agentdiet.cli.spot_check import sample_and_write
from agentdiet.config import Config
from agentdiet.extract_claims import (
    extract_claims_from_message,
    _parse_claims_payload,
)
from agentdiet.llm_client import DummyBackend, LLMClient
from agentdiet.types import Dialogue, Message


def _cfg(tmp_path: Path, *, seed: int = 42) -> Config:
    return Config(
        artifacts_dir=tmp_path,
        hf_cache_dir=tmp_path / "hf",
        model="test-model",
        n_agents=2, n_rounds=1, seed=seed,
    )


def _client(tmp_path: Path, responder):
    return LLMClient(DummyBackend(responder), cache_path=tmp_path / "c.jsonl")


# ---------------------------------------------------------------------------
# 1. Multi-byte characters
# ---------------------------------------------------------------------------

def test_quote_with_nonascii_is_indexed_by_codepoint(tmp_path):
    """Python str.find returns code-point offsets; verify span matches."""
    msg_text = "Answer: α + β = γ, so final answer is 42. #### 42"
    payload = [
        {"type": "evidence", "text": "greek ident", "quote": "α + β = γ"},
        {"type": "proposal", "text": "final", "quote": "#### 42"},
    ]
    client = _client(tmp_path, lambda msgs, m, t: json.dumps(payload))
    m = Message(agent_id=0, round=1, text=msg_text)
    claims, failed = extract_claims_from_message(
        message=m, question="q", qid="qα", llm_client=client,
        model="t", failures_dir=tmp_path / "fail",
    )
    assert failed is False
    assert len(claims) == 2
    # span[0], span[1] should slice back to the quote character-for-character.
    for c in claims:
        a, b = c.source_message_span
        extracted = msg_text[a:b]
        # Find the quote corresponding to this claim text so we know what to check.
        expected_quote = "α + β = γ" if c.type == "evidence" else "#### 42"
        assert extracted == expected_quote, \
            f"span {c.source_message_span} returned {extracted!r}, want {expected_quote!r}"


def test_quote_with_emoji_span_roundtrip(tmp_path):
    msg_text = "I think 🚀 = rocket. answer 7. #### 7"
    payload = [{"type": "evidence", "text": "emoji claim", "quote": "🚀 = rocket"}]
    client = _client(tmp_path, lambda msgs, m, t: json.dumps(payload))
    m = Message(agent_id=0, round=1, text=msg_text)
    claims, failed = extract_claims_from_message(
        message=m, question="q", qid="qE", llm_client=client,
        model="t", failures_dir=tmp_path / "fail",
    )
    assert failed is False
    assert len(claims) == 1
    a, b = claims[0].source_message_span
    assert msg_text[a:b] == "🚀 = rocket"


# ---------------------------------------------------------------------------
# 2. Resume correctness for "partial" outcomes
# ---------------------------------------------------------------------------

def test_resume_does_not_re_extract_partial_dialogues(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()

    d = Dialogue(
        question_id="qp", question="Q", gold_answer="7",
        messages=[
            Message(agent_id=0, round=1, text="ok 3 + 4 = 7. #### 7"),
            Message(agent_id=1, round=1, text="agree 3 + 4 = 7. #### 7"),
        ],
        final_answer="7",
    )
    (cfg.dialogues_dir).mkdir(parents=True, exist_ok=True)
    (cfg.dialogues_dir / "qp.json").write_text(d.model_dump_json())

    # First run: agent 1 message always fails → outcome "partial".
    def flaky(msgs, m, t):
        user = msgs[-1]["content"]
        if "from agent 1, round 1" in user:
            return "not json"
        return json.dumps([{"type": "evidence", "text": "x", "quote": "3 + 4 = 7"}])

    b1 = DummyBackend(flaky)
    manifest1 = run_extraction(cfg, backend=b1)
    assert manifest1["counts"]["partial"] == 1
    first_calls = b1.call_count

    # Second run: even if the responder would now succeed, run_extraction
    # must NOT re-call the LLM — the claim artifact already exists.
    b2 = DummyBackend(lambda msgs, m, t: json.dumps([{"type": "evidence", "text": "x", "quote": "3 + 4 = 7"}]))
    manifest2 = run_extraction(cfg, backend=b2)
    assert manifest2["counts"]["cached"] == 1
    assert manifest2["counts"]["ok"] == 0 and manifest2["counts"]["partial"] == 0
    assert b2.call_count == 0


# ---------------------------------------------------------------------------
# 3. Spot-check RNG isolation
# ---------------------------------------------------------------------------

def _seed_five_dialogues(cfg: Config) -> None:
    cfg.ensure_dirs()
    for qid in ("q1", "q2", "q3", "q4", "q5", "q6"):
        d = Dialogue(
            question_id=qid, question="Q", gold_answer="7",
            messages=[Message(agent_id=0, round=1, text="#### 7")],
            final_answer="7",
        )
        (cfg.dialogues_dir / f"{qid}.json").write_text(d.model_dump_json())
        claim = {
            "id": f"{qid}_r1_a0_c0", "text": f"answer for {qid}",
            "agent_id": 0, "round": 1, "type": "proposal",
            "source_message_span": [0, 7],
        }
        (cfg.claims_dir / f"{qid}.json").write_text(json.dumps({
            "qid": qid, "claims": [claim], "per_message_status": [],
            "extraction_failed": False,
        }))


def test_spot_check_does_not_mutate_global_random_state(tmp_path):
    cfg = _cfg(tmp_path, seed=42)
    _seed_five_dialogues(cfg)

    random.seed(12345)
    before = random.random()
    random.seed(12345)          # reset to the same point
    sample_and_write(cfg, k=3)  # should NOT advance the global RNG
    after = random.random()
    assert before == after, "sample_and_write mutated global random state"


def test_spot_check_reflects_cfg_seed_change(tmp_path):
    cfg_a = _cfg(tmp_path / "a", seed=42)
    cfg_b = _cfg(tmp_path / "b", seed=43)
    _seed_five_dialogues(cfg_a)
    _seed_five_dialogues(cfg_b)

    sample_and_write(cfg_a, k=3)
    sample_and_write(cfg_b, k=3)

    csv_a = (cfg_a.artifacts_dir / "spot_check.csv").read_text()
    csv_b = (cfg_b.artifacts_dir / "spot_check.csv").read_text()
    # Different seeds should pick different subsets (with high probability
    # given 6 candidates and k=3). If identical, seeding is broken.
    assert csv_a != csv_b


# ---------------------------------------------------------------------------
# 4. Every parse-failure branch in _parse_claims_payload
# ---------------------------------------------------------------------------

MSG = "I compute 3 + 4 = 7. #### 7"


def test_parse_non_dict_item_dropped():
    payload = [["not", "a", "dict"], {"type": "proposal", "text": "x", "quote": "#### 7"}]
    claims, err = _parse_claims_payload(json.dumps(payload), MSG, "q", 0, 1, 0)
    assert err is None
    assert len(claims) == 1
    assert claims[0].type == "proposal"


def test_parse_missing_quote_key_dropped():
    payload = [{"type": "proposal", "text": "x"}, {"type": "evidence", "text": "y", "quote": "3 + 4 = 7"}]
    claims, err = _parse_claims_payload(json.dumps(payload), MSG, "q", 0, 1, 0)
    assert err is None
    assert len(claims) == 1
    assert claims[0].type == "evidence"


def test_parse_missing_text_key_dropped():
    payload = [{"type": "proposal", "quote": "#### 7"}, {"type": "evidence", "text": "y", "quote": "3 + 4 = 7"}]
    claims, err = _parse_claims_payload(json.dumps(payload), MSG, "q", 0, 1, 0)
    assert err is None
    assert len(claims) == 1
    assert claims[0].type == "evidence"


def test_parse_empty_quote_dropped():
    payload = [{"type": "proposal", "text": "x", "quote": ""}, {"type": "evidence", "text": "y", "quote": "3 + 4 = 7"}]
    claims, err = _parse_claims_payload(json.dumps(payload), MSG, "q", 0, 1, 0)
    assert err is None
    assert len(claims) == 1


def test_parse_all_items_invalid_returns_error():
    payload = [{"type": "bogus", "text": "x", "quote": "3 + 4 = 7"}]
    claims, err = _parse_claims_payload(json.dumps(payload), MSG, "q", 0, 1, 0)
    assert claims == []
    assert err is not None
    assert "bad type" in err


# ---------------------------------------------------------------------------
# 5. Dialogue-level exception path in run_extraction
# ---------------------------------------------------------------------------

def test_dialogue_level_exception_marked_failed(tmp_path):
    """If extract_claims_for_dialogue itself raises, outcome is 'failed'
    and a dialogue-level failure artifact is written."""
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    d = Dialogue(
        question_id="qx", question="Q", gold_answer="7",
        messages=[Message(agent_id=0, round=1, text="ok")],
        final_answer="7",
    )
    (cfg.dialogues_dir / "qx.json").write_text(d.model_dump_json())

    class BadClient:
        def chat(self, messages, model, temperature):
            raise RuntimeError("backend meltdown")

    # LLMClient wraps the backend; RuntimeError inside chat will bubble
    # up through _call_with_retry after max_retries exhaustion.
    from agentdiet.llm_client import LLMClient
    client = LLMClient(BadClient(), cache_path=tmp_path / "c.jsonl", max_retries=1, base_backoff_s=0.0)
    # Monkeypatch _make_client to return our failing client.
    import agentdiet.cli.extract as mod
    orig = mod._make_client
    mod._make_client = lambda cfg, backend=None: client
    try:
        manifest = run_extraction(cfg)
    finally:
        mod._make_client = orig

    assert manifest["counts"]["failed"] == 1
    fail_files = list((cfg.failures_dir / "claim_extraction").glob("qx__*.json"))
    assert len(fail_files) == 1
