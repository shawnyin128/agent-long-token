from __future__ import annotations

from pathlib import Path

import pytest

from agentdiet.compress import Policy
from agentdiet.config import Config
from agentdiet.evaluate import run_sweep
from agentdiet.llm_client import DummyBackend, LLMClient
from agentdiet.types import Dialogue, Message


def _cfg(tmp_path: Path) -> Config:
    return Config(
        artifacts_dir=tmp_path, hf_cache_dir=tmp_path / "hf",
        model="test-model", seed=42,
    )


def _client(tmp_path, responder) -> LLMClient:
    return LLMClient(DummyBackend(responder), cache_path=tmp_path / "c.jsonl")


def _dialogue(qid: str, gold: str = "7") -> Dialogue:
    return Dialogue(
        question_id=qid, question=f"Q for {qid}", gold_answer=gold,
        messages=[
            Message(agent_id=0, round=1, text="thinking... #### 7"),
            Message(agent_id=1, round=1, text="same #### 7"),
            Message(agent_id=0, round=2, text="#### 7"),
            Message(agent_id=1, round=2, text="#### 7"),
        ],
        final_answer="7",
    )


def _claims(d: Dialogue) -> dict:
    return {"qid": d.question_id, "claims": [
        {"id": f"{d.question_id}_r1_a0_c0", "text": "t",
         "agent_id": 0, "round": 1, "type": "evidence",
         "source_message_span": [0, 8]},
    ], "per_message_status": [], "extraction_failed": False}


def _all_policies() -> dict[str, Policy]:
    return {
        "b1": Policy(mode="b1"),
        "b2": Policy(mode="b2"),
        "b3": Policy(mode="b3", last_k=1),
        "b5": Policy(mode="b5", drop_rate=0.3, random_seed=42),
        "ours": Policy(mode="ours", drop_types=["other"]),
    }


def test_run_sweep_produces_5_rows_per_qid(tmp_path):
    cfg = _cfg(tmp_path)
    d1 = _dialogue("qa")
    d2 = _dialogue("qb")

    def loader(cfg, qid):
        d = {"qa": d1, "qb": d2}[qid]
        return d, _claims(d), None

    client = _client(tmp_path, lambda m, mo, t: "#### 7")
    result = run_sweep(
        cfg=cfg, qids=["qa", "qb"], policies=_all_policies(),
        llm_client=client, loader=loader,
    )
    assert len(result["per_question"]) == 10        # 2 qids × 5 methods
    assert len(result["per_method"]) == 5
    methods = {m["method"] for m in result["per_method"]}
    assert methods == {"b1", "b2", "b3", "b5", "ours"}


def test_run_sweep_invariant_violations_empty_when_sane(tmp_path):
    cfg = _cfg(tmp_path)
    d = _dialogue("qa")

    def loader(c, qid):
        return d, _claims(d), None

    client = _client(tmp_path, lambda m, mo, t: "#### 7")
    result = run_sweep(
        cfg=cfg, qids=["qa"], policies=_all_policies(),
        llm_client=client, loader=loader,
    )
    # All methods return correct answer → accuracies equal → no violation.
    # Token ordering: b1 > b3 (last_k=1) by construction.
    assert result["invariant_violations"] == []


def test_run_sweep_detects_acc_b1_lt_b2(tmp_path):
    cfg = _cfg(tmp_path)
    d = _dialogue("qa")

    def loader(c, qid):
        return d, _claims(d), None

    # b1 uses full history (contains "#### 7" multiple times); b2 uses
    # only round-1 agent-0 (the shortest prompt). We need b1 to be WRONG
    # but b2 RIGHT — tune by prompt-length detection.
    def tuned(msgs, mo, t):
        user = msgs[-1]["content"]
        if "(round 2)" in user:
            return "#### 3"  # full-history responses are wrong
        return "#### 7"

    client = _client(tmp_path, tuned)
    result = run_sweep(
        cfg=cfg, qids=["qa"], policies=_all_policies(),
        llm_client=client, loader=loader,
    )
    violations = result["invariant_violations"]
    summaries = {m["method"]: m for m in result["per_method"]}
    if summaries["b1"]["accuracy"] < summaries["b2"]["accuracy"]:
        assert any("acc(b1)" in v for v in violations)


def test_run_sweep_missing_method_raises(tmp_path):
    cfg = _cfg(tmp_path)

    def loader(c, qid):
        d = _dialogue(qid)
        return d, _claims(d), None

    policies = {"b1": Policy(mode="b1")}  # missing others
    client = _client(tmp_path, lambda m, mo, t: "#### 7")
    with pytest.raises(KeyError):
        run_sweep(
            cfg=cfg, qids=["qa"], policies=policies,
            llm_client=client, loader=loader,
        )


def test_per_method_totals_equal_per_question_sum(tmp_path):
    cfg = _cfg(tmp_path)

    def loader(c, qid):
        d = _dialogue(qid)
        return d, _claims(d), None

    client = _client(tmp_path, lambda m, mo, t: "#### 7")
    result = run_sweep(
        cfg=cfg, qids=["qa", "qb", "qc"], policies=_all_policies(),
        llm_client=client, loader=loader,
    )
    for summary in result["per_method"]:
        rows = [r for r in result["per_question"] if r["method"] == summary["method"]]
        assert summary["n_evaluated"] == len(rows)
        assert summary["total_tokens"] == sum(r["compressed_tokens"] for r in rows)
        assert summary["accuracy"] == pytest.approx(
            sum(1 for r in rows if r["correct"]) / len(rows)
        )
