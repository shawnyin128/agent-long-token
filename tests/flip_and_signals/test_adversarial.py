"""Red-team tests for flip-and-signals.

Probes concerns not covered by per-step tests:

  * empty / unicode inputs to HashingFakeEmbedder
  * flip detection across 3+ rounds with a no-flip middle
  * real-embedder ImportError → fallback path in analyze CLI
  * signal_scores.parquet round-trip preserves bool/float dtypes
  * FlipEvent validation on missing gold answer
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from agentdiet.analysis.flip import locate_flips
from agentdiet.analysis.signals import (
    HashingFakeEmbedder,
    SentenceTransformerEmbedder,
    compute_signals,
)
from agentdiet.cli.analyze import run_analysis, _select_embedder
from agentdiet.config import Config
from agentdiet.types import Dialogue, FlipEvent, Message


# ---------------------------------------------------------------------------
# HashingFakeEmbedder edge cases
# ---------------------------------------------------------------------------

def test_hashing_fake_embedder_handles_empty_input():
    emb = HashingFakeEmbedder(dim=8)
    out = emb.encode([])
    assert out.shape == (0, 8)


def test_hashing_fake_embedder_handles_empty_string():
    emb = HashingFakeEmbedder(dim=8)
    out = emb.encode([""])
    assert out.shape == (1, 8)
    assert np.isclose(np.linalg.norm(out[0]), 1.0, atol=1e-6)


def test_hashing_fake_embedder_handles_unicode():
    emb = HashingFakeEmbedder(dim=16)
    out = emb.encode(["α β γ", "δ ε ζ"])
    assert out.shape == (2, 16)


# ---------------------------------------------------------------------------
# Flip detection in 3+ rounds
# ---------------------------------------------------------------------------

def _d(per_round_ans: list[list[str]], gold: str = "7") -> Dialogue:
    msgs = []
    for r_idx, rnd in enumerate(per_round_ans, 1):
        for a_idx, ans in enumerate(rnd):
            msgs.append(Message(
                agent_id=a_idx, round=r_idx,
                text=f"answer {ans} #### {ans}",
            ))
    return Dialogue(
        question_id="q", question="Q", gold_answer=gold,
        messages=msgs, final_answer=None,
    )


def _c(d: Dialogue) -> dict:
    claims = []
    for m in d.messages:
        claims.append({
            "id": f"q_r{m.round}_a{m.agent_id}_c0",
            "text": m.text, "agent_id": m.agent_id, "round": m.round,
            "type": "proposal", "source_message_span": [0, 3],
        })
    return {"qid": "q", "claims": claims, "per_message_status": [],
            "extraction_failed": False}


def test_flip_then_hold_no_duplicate_event():
    # R1 wrong, R2 right, R3 right → one flip at R2, none at R3.
    d = _d([["3", "3", "3"], ["7", "7", "7"], ["7", "7", "7"]])
    events = locate_flips(d, _c(d))
    assert len(events) == 1
    assert events[0].round == 2


def test_no_flip_on_right_to_right_transition():
    d = _d([["7", "7", "7"], ["7", "7", "7"]])
    assert locate_flips(d, _c(d)) == []


def test_single_round_dialogue_has_no_flips():
    d = _d([["3", "3", "3"]])
    assert locate_flips(d, _c(d)) == []


# ---------------------------------------------------------------------------
# Real-embedder import fallback
# ---------------------------------------------------------------------------

def test_select_embedder_fake_flag_short_circuits():
    emb = _select_embedder(use_fake=True)
    assert isinstance(emb, HashingFakeEmbedder)


def test_select_embedder_falls_back_when_import_fails(monkeypatch, capsys):
    # Simulate sentence-transformers being unavailable.
    def _boom(self):
        raise ImportError("sentence-transformers not installed for test")

    monkeypatch.setattr(SentenceTransformerEmbedder, "_ensure", _boom)
    emb = _select_embedder(use_fake=False)
    assert isinstance(emb, HashingFakeEmbedder)
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "HashingFakeEmbedder" in captured.err


# ---------------------------------------------------------------------------
# Parquet round-trip preserves dtypes
# ---------------------------------------------------------------------------

def _cfg(tmp_path: Path) -> Config:
    return Config(
        artifacts_dir=tmp_path, hf_cache_dir=tmp_path / "hf",
        model="test-model", n_agents=3, n_rounds=2,
    )


def _seed_pair(cfg: Config, qid: str) -> None:
    d = _d([["3", "3", "3"], ["7", "7", "7"]])
    d = Dialogue(
        question_id=qid, question=d.question, gold_answer=d.gold_answer,
        messages=d.messages, final_answer=d.final_answer,
    )
    cfg.dialogues_dir.mkdir(parents=True, exist_ok=True)
    (cfg.dialogues_dir / f"{qid}.json").write_text(d.model_dump_json())

    claims_doc = _c(d)
    claims_doc["qid"] = qid
    for c in claims_doc["claims"]:
        c["id"] = c["id"].replace("q_", f"{qid}_")
    cfg.claims_dir.mkdir(parents=True, exist_ok=True)
    (cfg.claims_dir / f"{qid}.json").write_text(json.dumps(claims_doc))


def test_parquet_dtypes_preserved(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _seed_pair(cfg, "qa")
    run_analysis(cfg, use_fake_embedder=True)

    table = pq.read_table(cfg.analysis_dir / "signal_scores.parquet")
    schema = {name: table.schema.field(name).type for name in table.column_names}
    # booleans must remain booleans, floats must remain floats.
    assert str(schema["flip_coincidence"]) == "bool"
    assert str(schema["referenced_later"]) == "bool"
    assert str(schema["novelty"]).startswith("double") or str(schema["novelty"]).startswith("float")
    assert str(schema["position"]).startswith("int")


def test_flip_events_jsonl_is_line_delimited(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _seed_pair(cfg, "qa")
    _seed_pair(cfg, "qb")
    run_analysis(cfg, use_fake_embedder=True)

    raw = (cfg.analysis_dir / "flip_events.jsonl").read_text()
    lines = [l for l in raw.splitlines() if l]
    # Each line must be parseable JSON on its own.
    for l in lines:
        obj = json.loads(l)
        assert "qid" in obj
        assert "round" in obj
        assert "triggering_claim_id" in obj


# ---------------------------------------------------------------------------
# Signal invariants under stress
# ---------------------------------------------------------------------------

def test_novelty_bounded_in_0_1():
    claims = [
        {"id": f"c{i}", "text": f"text variant {i}", "agent_id": 0,
         "round": 1, "type": "proposal", "source_message_span": [0, 1]}
        for i in range(20)
    ]
    rows = compute_signals(claims, flip_events=[], embedder=HashingFakeEmbedder())
    for r in rows:
        assert 0.0 <= r["novelty"] <= 1.0


def test_flip_coincidence_across_multiple_flip_events():
    claims = [
        {"id": "r1_a0", "text": "x", "agent_id": 0, "round": 1,
         "type": "proposal", "source_message_span": [0, 1]},
        {"id": "r2_a0", "text": "y", "agent_id": 0, "round": 2,
         "type": "proposal", "source_message_span": [0, 1]},
        {"id": "r3_a0", "text": "z", "agent_id": 0, "round": 3,
         "type": "proposal", "source_message_span": [0, 1]},
    ]
    flips = [
        FlipEvent(question_id="q", round=2, triggering_claim_id="r2_a0",
                  pre_flip_answers={}, post_flip_answers={}),
        FlipEvent(question_id="q", round=3, triggering_claim_id="r3_a0",
                  pre_flip_answers={}, post_flip_answers={}),
    ]
    rows = compute_signals(claims, flip_events=flips, embedder=HashingFakeEmbedder())
    by_id = {r["claim_id"]: r for r in rows}
    assert by_id["r1_a0"]["flip_coincidence"] is False
    assert by_id["r2_a0"]["flip_coincidence"] is True
    assert by_id["r3_a0"]["flip_coincidence"] is True
