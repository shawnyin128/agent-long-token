from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentdiet.cli.pilot import run_pilot
from agentdiet.config import Config
from agentdiet.dataset import Question
from agentdiet.llm_client import DummyBackend


QS = [
    Question(qid=f"q{i}", question=f"Add {i} + 1", gold_answer=str(i + 1))
    for i in range(3)
]


def _cfg(tmp_path: Path) -> Config:
    return Config(
        artifacts_dir=tmp_path,
        hf_cache_dir=tmp_path / "hf",
        model="test-model",
        n_agents=2,
        n_rounds=1,
        n_pilot=3,
    )


def test_pilot_produces_both_methods(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()

    def responder(msgs, model, temp):
        last = msgs[-1]["content"]
        for q in QS:
            if q.question in last:
                return f"#### {q.gold_answer}"
        return "#### 0"

    backend = DummyBackend(responder)
    manifest = run_pilot(cfg, QS, run_single=True, run_debate_flag=True, backend=backend)

    single_dir = tmp_path / "pilot" / "single" / "test-model"
    debate_dir = tmp_path / "pilot" / "debate" / "test-model"
    assert {p.stem for p in single_dir.glob("*.json")} == {q.qid for q in QS}
    assert {p.stem for p in debate_dir.glob("*.json")} == {q.qid for q in QS}
    assert all(o["single"] == "ok" and o["debate"] == "ok" for o in manifest["outcomes"])
    assert (tmp_path / "pilot" / "manifest.json").exists()


def test_resume_makes_no_calls(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()

    b1 = DummyBackend(lambda m, mo, t: "#### 42")
    run_pilot(cfg, QS, run_single=True, run_debate_flag=True, backend=b1)
    first = b1.call_count

    b2 = DummyBackend(lambda m, mo, t: "NOT_CALLED")
    manifest = run_pilot(cfg, QS, run_single=True, run_debate_flag=True, backend=b2)
    assert b2.call_count == 0
    assert all(o["single"] == "cached" and o["debate"] == "cached" for o in manifest["outcomes"])
    assert first > 0


def test_no_debate_flag(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    backend = DummyBackend(lambda m, mo, t: "#### 42")
    manifest = run_pilot(cfg, QS, run_single=True, run_debate_flag=False, backend=backend)
    assert all(o["single"] == "ok" and o["debate"] == "skip" for o in manifest["outcomes"])
    assert not list((tmp_path / "pilot" / "debate").rglob("*.json"))


def test_per_qid_failure_does_not_stop(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()

    call = {"n": 0}

    def flaky(msgs, model, temp):
        call["n"] += 1
        # fail the single-agent call for q1 only
        if "Add 1 + 1" in msgs[-1]["content"] and "other agents" not in msgs[-1]["content"]:
            raise RuntimeError("synthetic fail on q1 single")
        return "#### 99"

    backend = DummyBackend(flaky)
    manifest = run_pilot(cfg, QS, run_single=True, run_debate_flag=False, backend=backend)

    # q0 + q2 ok, q1 failed
    outcomes = {o["qid"]: o for o in manifest["outcomes"]}
    assert outcomes["q0"]["single"] == "ok"
    assert outcomes["q1"]["single"] == "failed"
    assert outcomes["q2"]["single"] == "ok"
    # failure artifact written
    fail_files = list((tmp_path / "failures" / "pilot").glob("single__q1.json"))
    assert fail_files
    failure = json.loads(fail_files[0].read_text())
    assert failure["qid"] == "q1"


def test_unparsed_marked_but_written(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()

    def responder(msgs, model, temp):
        return "no number here"

    backend = DummyBackend(responder)
    manifest = run_pilot(cfg, QS, run_single=True, run_debate_flag=False, backend=backend)
    assert all(o["single"] == "unparsed" for o in manifest["outcomes"])
    # artifact still on disk
    assert {p.stem for p in (tmp_path / "pilot" / "single" / "test-model").glob("*.json")} == {q.qid for q in QS}
