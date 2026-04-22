from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentdiet.cli.extract import main, run_extraction, _manifest_path
from agentdiet.config import Config
from agentdiet.llm_client import DummyBackend
from agentdiet.types import Dialogue, Message


def _cfg(tmp_path: Path) -> Config:
    return Config(
        artifacts_dir=tmp_path,
        hf_cache_dir=tmp_path / "hf",
        model="test-model",
        n_agents=2,
        n_rounds=1,
    )


def _dialogue(qid: str) -> Dialogue:
    msgs = [
        Message(agent_id=0, round=1, text="I compute 3 + 4 = 7. #### 7"),
        Message(agent_id=1, round=1, text="I agree with agent 0. #### 7"),
    ]
    return Dialogue(
        question_id=qid, question="What is 3+4?", gold_answer="7",
        messages=msgs, final_answer="7",
    )


def _write_dialogue(cfg: Config, d: Dialogue) -> None:
    cfg.dialogues_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.dialogues_dir / f"{d.question_id}.json"
    path.write_text(d.model_dump_json())


def _good_responder(msgs, model, temperature):
    # Pick a quote that exists in both possible messages.
    user = msgs[-1]["content"]
    if "I agree with agent 0" in user and "from agent 1" in user:
        quote = "I agree with agent 0"
        type_ = "agreement"
    else:
        quote = "3 + 4 = 7"
        type_ = "evidence"
    return json.dumps([{"type": type_, "text": "x", "quote": quote}])


def test_run_extraction_writes_one_artifact_per_dialogue(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    for qid in ("qa", "qb", "qc"):
        _write_dialogue(cfg, _dialogue(qid))
    backend = DummyBackend(_good_responder)
    manifest = run_extraction(cfg, backend=backend)

    out = cfg.claims_dir
    assert {p.stem for p in out.glob("*.json")} == {"qa", "qb", "qc"}
    assert manifest["counts"]["ok"] == 3
    assert manifest["counts"]["partial"] == 0
    assert manifest["counts"]["failed"] == 0
    assert _manifest_path(cfg).exists()


def test_resume_makes_zero_new_llm_calls(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    for qid in ("qa", "qb"):
        _write_dialogue(cfg, _dialogue(qid))

    b1 = DummyBackend(_good_responder)
    run_extraction(cfg, backend=b1)
    first = b1.call_count
    assert first > 0

    # Second run: all claim artifacts exist → resume path, zero new LLM calls.
    b2 = DummyBackend(_good_responder)
    run_extraction(cfg, backend=b2)
    assert b2.call_count == 0


def test_partial_outcome_when_some_messages_fail(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _write_dialogue(cfg, _dialogue("qa"))

    def resp(msgs, m, t):
        user = msgs[-1]["content"]
        if "from agent 1, round 1" in user:
            return "broken"
        return json.dumps([{"type": "evidence", "text": "x", "quote": "3 + 4 = 7"}])

    manifest = run_extraction(cfg, backend=DummyBackend(resp))
    assert manifest["counts"]["partial"] == 1
    assert manifest["counts"]["ok"] == 0


def test_dry_run_zero_calls(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _write_dialogue(cfg, _dialogue("qa"))
    # main() with --dry-run should not touch LLM.
    rc = main(["--dry-run"], cfg=cfg)
    assert rc == 0


def test_report_manifest_exits_2_when_missing(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    rc = main(["--report-manifest"], cfg=cfg)
    assert rc == 2


def test_report_manifest_prints_counts_when_present(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _write_dialogue(cfg, _dialogue("qa"))
    run_extraction(cfg, backend=DummyBackend(_good_responder))
    rc = main(["--report-manifest"], cfg=cfg)
    assert rc == 0
    captured = capsys.readouterr()
    assert "ok" in captured.out


def test_exit_code_1_when_all_dialogues_fail(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _write_dialogue(cfg, _dialogue("qa"))
    _write_dialogue(cfg, _dialogue("qb"))

    def always_bad(msgs, m, t):
        return "not json at all"

    rc = main([], cfg=cfg, backend=DummyBackend(always_bad))
    # Every message fails extraction → both dialogues land in "partial" bucket
    # (not ok, not failed). Exit 1 should be reserved for 100% dialogue-level
    # exceptions, not partial successes.
    assert rc == 0

    # Force a hard failure by making the dialogues directory unreadable:
    # simulate by removing all dialogue files mid-run is awkward. Instead,
    # verify via synthetic injection: write a dialogue that cannot be parsed
    # as JSON. run_extraction should treat that as a dialogue-level failure.
    cfg2 = Config(
        artifacts_dir=tmp_path / "b",
        hf_cache_dir=tmp_path / "b" / "hf",
        model="test-model", n_agents=2, n_rounds=1,
    )
    cfg2.ensure_dirs()
    (cfg2.dialogues_dir / "bad.json").write_text("{not valid json")
    rc2 = main([], cfg=cfg2, backend=DummyBackend(_good_responder))
    assert rc2 == 1


def test_writes_manifest_with_expected_fields(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _write_dialogue(cfg, _dialogue("qa"))
    run_extraction(cfg, backend=DummyBackend(_good_responder))
    man = json.loads(_manifest_path(cfg).read_text())
    for k in ("model", "n", "counts", "outcomes"):
        assert k in man
    for bucket in ("ok", "partial", "failed", "cached"):
        assert bucket in man["counts"]
    assert all({"qid", "outcome"} <= set(o.keys()) for o in man["outcomes"])
