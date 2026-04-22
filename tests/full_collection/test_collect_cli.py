from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentdiet.cli.collect import _manifest_path, main, run_collection
from agentdiet.config import Config
from agentdiet.dataset import Question
from agentdiet.llm_client import DummyBackend


QS = [
    Question(qid=f"q{i}", question=f"Compute {i} + 1", gold_answer=str(i + 1))
    for i in range(3)
]


def _cfg(tmp_path: Path) -> Config:
    return Config(
        artifacts_dir=tmp_path,
        hf_cache_dir=tmp_path / "hf",
        model="test-model",
        n_agents=2,
        n_rounds=1,
        n_questions=3,
    )


def test_run_collection_produces_n_artifacts(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    backend = DummyBackend(lambda msgs, m, t: "#### 1")
    manifest = run_collection(cfg, QS, backend=backend)

    out_dir = cfg.dialogues_dir
    assert {p.stem for p in out_dir.glob("*.json")} == {q.qid for q in QS}
    assert manifest["counts"]["ok"] == 3
    assert manifest["counts"]["failed"] == 0
    assert _manifest_path(cfg).exists()


def test_resume_makes_zero_calls(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()

    b1 = DummyBackend(lambda m, mo, t: "#### 1")
    run_collection(cfg, QS, backend=b1)
    assert b1.call_count > 0

    b2 = DummyBackend(lambda m, mo, t: "SHOULD_NOT_BE_CALLED")
    manifest = run_collection(cfg, QS, backend=b2)
    assert b2.call_count == 0
    assert all(o["outcome"] == "cached" for o in manifest["outcomes"])


def test_per_qid_failure_does_not_stop(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()

    def responder(msgs, model, temp):
        last = msgs[-1]["content"]
        if "Compute 1 +" in last:
            raise RuntimeError("synthetic fail on q1")
        return "#### 1"

    backend = DummyBackend(responder)
    manifest = run_collection(cfg, QS, backend=backend)

    outcomes = {o["qid"]: o["outcome"] for o in manifest["outcomes"]}
    assert outcomes["q0"] == "ok"
    assert outcomes["q1"] == "failed"
    assert outcomes["q2"] == "ok"
    assert (tmp_path / "failures" / "debate" / "q1.json").exists()
    fail = json.loads((tmp_path / "failures" / "debate" / "q1.json").read_text())
    assert fail["qid"] == "q1"
    assert "traceback" in fail


def test_unparsed_marked(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    backend = DummyBackend(lambda m, mo, t: "no number here")
    manifest = run_collection(cfg, QS, backend=backend)
    assert all(o["outcome"] == "unparsed" for o in manifest["outcomes"])


def test_dry_run_still_exits_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTDIET_ARTIFACTS_DIR", str(tmp_path))
    rc = main(["--dry-run"])
    assert rc == 0
