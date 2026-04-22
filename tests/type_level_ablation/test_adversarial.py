"""Red-team tests for type-level-ablation.

Covers subtle interactions that per-step tests do not:
  * cache-hit when drop_type has no claims (history should be unchanged)
  * budget tracking across call and cache boundaries
  * masked-message preserves (agent_id, round) indices even when every
    claim in a message is dropped
  * Gate-2 verdict boundary cases
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentdiet.analysis.ablate import (
    reconstruct_masked_history, replay_final_round, run_ablation,
    select_subset,
)
from agentdiet.cli.ablate import (
    GATE2_NOISE_THRESHOLD, GATE2_PASS_THRESHOLD, _render_gate2,
)
from agentdiet.config import Config
from agentdiet.llm_client import DummyBackend, LLMClient
from agentdiet.types import Dialogue, Message


def _cfg(tmp_path: Path) -> Config:
    return Config(
        artifacts_dir=tmp_path, hf_cache_dir=tmp_path / "hf",
        model="test-model", n_agents=2, n_rounds=2, seed=42,
    )


def _client(tmp_path: Path, responder) -> LLMClient:
    return LLMClient(DummyBackend(responder), cache_path=tmp_path / "c.jsonl")


def _dialogue() -> tuple[Dialogue, dict]:
    d = Dialogue(
        question_id="q", question="Q", gold_answer="7",
        messages=[
            Message(agent_id=0, round=1, text="first turn"),
            Message(agent_id=1, round=1, text="second turn"),
            Message(agent_id=0, round=2, text="#### 7"),
            Message(agent_id=1, round=2, text="#### 7"),
        ],
        final_answer="7",
    )
    cd = {
        "qid": "q",
        "claims": [
            {"id": "q_r1_a0_c0", "text": "e",
             "agent_id": 0, "round": 1, "type": "evidence",
             "source_message_span": [0, 5]},
        ],
        "per_message_status": [], "extraction_failed": False,
    }
    return d, cd


# ---------------------------------------------------------------------------
# 1. Cache-hit when drop_type has zero claims
# ---------------------------------------------------------------------------

def test_drop_type_with_zero_claims_reuses_cache_on_repeat(tmp_path):
    d, cd = _dialogue()
    client = _client(tmp_path, lambda m, mo, t: "#### 7")
    # Warm the cache.
    replay_final_round(
        dialogue=d, claims_doc=cd, drop_type="question",
        llm_client=client, model="m",
    )
    first = client.call_count
    # Repeating the same (drop_type, dialogue) must cache-hit fully.
    replay_final_round(
        dialogue=d, claims_doc=cd, drop_type="question",
        llm_client=client, model="m",
    )
    assert client.call_count == first


# ---------------------------------------------------------------------------
# 2. Masked-message preserves (agent_id, round) indices even when emptied
# ---------------------------------------------------------------------------

def test_fully_masked_message_preserves_indices():
    d = Dialogue(
        question_id="q", question="Q", gold_answer="7",
        messages=[
            Message(agent_id=0, round=1, text="claim text only"),
            Message(agent_id=1, round=1, text="other content"),
        ],
    )
    cd = {
        "claims": [
            {"id": "q_r1_a0_c0", "text": "all of it",
             "agent_id": 0, "round": 1, "type": "other",
             "source_message_span": [0, 15]},
        ],
    }
    new = reconstruct_masked_history(d, cd, drop_type="other", up_to_round=1)
    keys = [(m.agent_id, m.round) for m in new]
    assert keys == [(0, 1), (1, 1)]
    assert new[0].text == ""
    assert new[1].text == "other content"


# ---------------------------------------------------------------------------
# 3. Budget boundary — exactly at cap
# ---------------------------------------------------------------------------

def test_budget_exactly_at_cap_still_runs_first_pair(tmp_path):
    d, cd = _dialogue()
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    (cfg.dialogues_dir / "q.json").write_text(d.model_dump_json())
    (cfg.claims_dir / "q.json").write_text(json.dumps(cd))

    counter = {"n": 0}

    def resp(m, mo, t):
        counter["n"] += 1
        return f"#### {counter['n']}"

    client = _client(tmp_path, resp)
    # Cap = 2 (exactly one (qid, type) pair's worth of calls).
    rows = run_ablation(cfg=cfg, qids=["q"], llm_client=client,
                        max_new_llm_calls=2)
    # First pair succeeds, rest get budget-skipped.
    completed = [r for r in rows if not r.get("skipped")]
    skipped = [r for r in rows if r.get("skipped")]
    assert len(completed) == 1
    assert all("budget" in r.get("skip_reason", "") for r in skipped)


# ---------------------------------------------------------------------------
# 4. Gate-2 verdict on exact threshold boundaries
# ---------------------------------------------------------------------------

def test_gate2_exact_pass_threshold_is_pass():
    summary = {"per_type": [
        {"type": t, "n_used": 10, "n_skipped": 0, "acc_with": 1.0,
         "acc_without": 1.0 - GATE2_PASS_THRESHOLD, "delta": GATE2_PASS_THRESHOLD}
        for t in ["proposal"]
    ] + [
        {"type": t, "n_used": 10, "n_skipped": 0, "acc_with": 1.0,
         "acc_without": 1.0, "delta": 0.0}
        for t in ["evidence", "correction", "agreement", "question", "other"]
    ]}
    _, exit_code = _render_gate2(summary)
    assert exit_code == 0  # PASS at exact threshold


def test_gate2_exact_noise_threshold_is_null():
    summary = {"per_type": [
        {"type": t, "n_used": 10, "n_skipped": 0, "acc_with": 1.0,
         "acc_without": 1.0 - GATE2_NOISE_THRESHOLD, "delta": GATE2_NOISE_THRESHOLD}
        for t in ["proposal", "evidence", "correction",
                  "agreement", "question", "other"]
    ]}
    _, exit_code = _render_gate2(summary)
    assert exit_code == 10  # all within noise


# ---------------------------------------------------------------------------
# 5. Subset selection is empty when no pilot artifact present
# ---------------------------------------------------------------------------

def test_select_subset_empty_without_pilot_artifacts(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    d = Dialogue(
        question_id="q", question="Q", gold_answer="7",
        messages=[Message(agent_id=0, round=1, text="#### 7")],
        final_answer="7",
    )
    (cfg.dialogues_dir / "q.json").write_text(d.model_dump_json())
    (cfg.claims_dir / "q.json").write_text(json.dumps({
        "qid": "q", "claims": [], "per_message_status": [],
        "extraction_failed": False,
    }))
    # No pilot/single/ file — subset must be empty.
    assert select_subset(cfg, target_size=5) == []


# ---------------------------------------------------------------------------
# 6. Different drop_types produce different cache keys (prompt differs)
# ---------------------------------------------------------------------------

def test_different_drop_types_produce_fresh_calls(tmp_path):
    d = Dialogue(
        question_id="q", question="Q", gold_answer="7",
        messages=[
            Message(agent_id=0, round=1, text="evidence part; agreement part"),
            Message(agent_id=1, round=1, text="evidence part; agreement part"),
            Message(agent_id=0, round=2, text="#### 7"),
            Message(agent_id=1, round=2, text="#### 7"),
        ],
        final_answer="7",
    )
    cd = {
        "claims": [
            {"id": "q_r1_a0_ce", "text": "e", "agent_id": 0, "round": 1,
             "type": "evidence", "source_message_span": [0, 13]},
            {"id": "q_r1_a0_ca", "text": "a", "agent_id": 0, "round": 1,
             "type": "agreement", "source_message_span": [15, 29]},
        ],
    }
    client = _client(tmp_path, lambda m, mo, t: "#### 7")
    replay_final_round(dialogue=d, claims_doc=cd, drop_type="evidence",
                       llm_client=client, model="m")
    after_e = client.call_count
    replay_final_round(dialogue=d, claims_doc=cd, drop_type="agreement",
                       llm_client=client, model="m")
    after_a = client.call_count
    # agreement drop changes messages differently → new cache entries.
    assert after_a > after_e
