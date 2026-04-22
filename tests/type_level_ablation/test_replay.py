from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentdiet.analysis.ablate import (
    replay_final_round,
    run_ablation,
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
    return LLMClient(DummyBackend(responder), cache_path=tmp_path / "cache.jsonl")


def _dialogue_with_claims() -> tuple[Dialogue, dict]:
    d = Dialogue(
        question_id="q", question="What is 3+4?", gold_answer="7",
        messages=[
            Message(agent_id=0, round=1, text="computing 3+4 = 7. #### 7"),
            Message(agent_id=1, round=1, text="I agree with agent 0. #### 7"),
            Message(agent_id=0, round=2, text="Confirmed: #### 7"),
            Message(agent_id=1, round=2, text="Same answer: #### 7"),
        ],
        final_answer="7",
    )
    claims = [
        {"id": "q_r1_a0_c0", "text": "evidence",
         "agent_id": 0, "round": 1, "type": "evidence",
         "source_message_span": [10, 18]},
        {"id": "q_r1_a1_c0", "text": "agreement",
         "agent_id": 1, "round": 1, "type": "agreement",
         "source_message_span": [0, 22]},
    ]
    cd = {"qid": "q", "claims": claims, "per_message_status": [],
          "extraction_failed": False}
    return d, cd


def test_replay_returns_row_schema(tmp_path):
    d, cd = _dialogue_with_claims()
    cfg = _cfg(tmp_path)
    client = _client(tmp_path, lambda m, mo, t: "#### 7")
    result = replay_final_round(
        dialogue=d, claims_doc=cd, drop_type="agreement",
        llm_client=client, model=cfg.model, temperature=0.0,
    )
    assert result["qid"] == "q"
    assert result["drop_type"] == "agreement"
    assert result["post_final"] == "7"
    assert result["pre_final"] == "7"
    assert result["gold"] == "7"
    assert result["correct_with"] is True
    assert result["correct_without"] is True


def test_replay_detects_flip_to_wrong(tmp_path):
    d, cd = _dialogue_with_claims()
    cfg = _cfg(tmp_path)
    # With agreement removed, both agents produce "3" → new majority "3" ≠ gold.
    client = _client(tmp_path, lambda m, mo, t: "#### 3")
    result = replay_final_round(
        dialogue=d, claims_doc=cd, drop_type="agreement",
        llm_client=client, model=cfg.model, temperature=0.0,
    )
    assert result["post_final"] == "3"
    assert result["correct_without"] is False


def test_replay_cached_call_is_free_on_repeat(tmp_path):
    d, cd = _dialogue_with_claims()
    cfg = _cfg(tmp_path)
    client = _client(tmp_path, lambda m, mo, t: "#### 7")
    replay_final_round(
        dialogue=d, claims_doc=cd, drop_type="agreement",
        llm_client=client, model=cfg.model,
    )
    first_count = client.call_count
    replay_final_round(
        dialogue=d, claims_doc=cd, drop_type="agreement",
        llm_client=client, model=cfg.model,
    )
    assert client.call_count == first_count  # cache hit


def test_replay_one_call_per_agent(tmp_path):
    d, cd = _dialogue_with_claims()
    cfg = _cfg(tmp_path)
    client = _client(tmp_path, lambda m, mo, t: "#### 7")
    replay_final_round(
        dialogue=d, claims_doc=cd, drop_type="agreement",
        llm_client=client, model=cfg.model,
    )
    # 2 agents in this dialogue (cfg.n_agents=2 in test, but dialogue has 2 agents).
    assert client.call_count == 2


def test_run_ablation_iterates_six_types_per_qid(tmp_path):
    d, cd = _dialogue_with_claims()
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    (cfg.dialogues_dir / "q.json").write_text(d.model_dump_json())
    (cfg.claims_dir / "q.json").write_text(json.dumps(cd))

    client = _client(tmp_path, lambda m, mo, t: "#### 7")
    rows = run_ablation(
        cfg=cfg, qids=["q"], llm_client=client, max_new_llm_calls=1000,
    )
    assert len(rows) == 6  # 1 qid × 6 types
    drop_types = {r["drop_type"] for r in rows}
    assert drop_types == {"proposal", "evidence", "correction",
                          "agreement", "question", "other"}


def test_run_ablation_hard_cap_stops_mid_run(tmp_path):
    d, cd = _dialogue_with_claims()
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    for qid in ("q1", "q2", "q3"):
        dd = Dialogue(**{**d.model_dump(), "question_id": qid})
        (cfg.dialogues_dir / f"{qid}.json").write_text(dd.model_dump_json())
        cdd = {**cd, "qid": qid,
               "claims": [{**c, "id": c["id"].replace("q_", f"{qid}_")}
                          for c in cd["claims"]]}
        (cfg.claims_dir / f"{qid}.json").write_text(json.dumps(cdd))

    # Each unique (qid, type) triggers 2 new calls (2 agents).
    # Cap at 4 → only 2 (qid, type) pairs complete, rest skipped.
    calls_made = {"n": 0}

    def resp(m, mo, t):
        # Vary the response by qid so cache doesn't save us.
        calls_made["n"] += 1
        return f"#### {calls_made['n']}"

    client = _client(tmp_path, resp)
    rows = run_ablation(
        cfg=cfg, qids=["q1", "q2", "q3"], llm_client=client,
        max_new_llm_calls=4,
    )
    skipped = [r for r in rows if r.get("skipped")]
    completed = [r for r in rows if not r.get("skipped")]
    assert len(completed) == 2
    assert len(skipped) == len(rows) - 2
    assert any("budget" in r.get("skip_reason", "") for r in skipped)


def test_run_ablation_skips_claims_missing_gracefully(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    # Write dialogue but no claims artifact.
    d = Dialogue(
        question_id="qx", question="Q", gold_answer="7",
        messages=[Message(agent_id=0, round=1, text="#### 7")],
    )
    (cfg.dialogues_dir / "qx.json").write_text(d.model_dump_json())
    client = _client(tmp_path, lambda m, mo, t: "#### 7")
    rows = run_ablation(
        cfg=cfg, qids=["qx"], llm_client=client, max_new_llm_calls=100,
    )
    # qx should appear as skipped in each of 6 type rows.
    assert all(r.get("skipped") for r in rows)
    assert all("missing" in r.get("skip_reason", "") for r in rows)
