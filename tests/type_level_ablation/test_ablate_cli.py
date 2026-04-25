from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentdiet.cli.ablate import (
    main, run_ablation_cli, _manifest_path, _summary_path, _jsonl_path,
)
from agentdiet.config import Config
from agentdiet.llm_client import DummyBackend
from agentdiet.types import Dialogue, Message


def _cfg(tmp_path: Path) -> Config:
    return Config(
        artifacts_dir=tmp_path, hf_cache_dir=tmp_path / "hf",
        model="test-model", n_agents=2, n_rounds=2, seed=42,
    )


def _seed_eligible(cfg: Config, qid: str) -> None:
    d = Dialogue(
        question_id=qid, question="Q", gold_answer="7",
        messages=[
            Message(agent_id=0, round=1, text="ok 3+4=7. #### 7"),
            Message(agent_id=1, round=1, text="agree #### 7"),
            Message(agent_id=0, round=2, text="#### 7"),
            Message(agent_id=1, round=2, text="#### 7"),
        ],
        final_answer="7",
    )
    cfg.dialogues_dir.mkdir(parents=True, exist_ok=True)
    (cfg.dialogues_dir / f"{qid}.json").write_text(d.model_dump_json())

    claims = [
        {"id": f"{qid}_r1_a0_c0", "text": "e",
         "agent_id": 0, "round": 1, "type": "evidence",
         "source_message_span": [3, 8]},
        {"id": f"{qid}_r1_a1_c0", "text": "a",
         "agent_id": 1, "round": 1, "type": "agreement",
         "source_message_span": [0, 5]},
    ]
    cfg.claims_dir.mkdir(parents=True, exist_ok=True)
    (cfg.claims_dir / f"{qid}.json").write_text(json.dumps({
        "qid": qid, "claims": claims, "per_message_status": [],
        "extraction_failed": False,
    }))

    # Pilot single doc: single_wrong (gold=7, single returns 3).
    pilot_dir = cfg.artifacts_dir / "pilot" / "single" / cfg.model_slug
    pilot_dir.mkdir(parents=True, exist_ok=True)
    (pilot_dir / f"{qid}.json").write_text(json.dumps({
        "question_id": qid, "question": "Q", "gold_answer": "7",
        "messages": [{"agent_id": 0, "round": 1, "text": "#### 3"}],
        "final_answer": "3", "meta": {},
    }))


def test_cli_produces_jsonl_and_summary(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _seed_eligible(cfg, "qa")
    _seed_eligible(cfg, "qb")

    backend = DummyBackend(lambda m, mo, t: "#### 7")
    manifest = run_ablation_cli(
        cfg=cfg, target_size=2, max_new_llm_calls=500, backend=backend,
    )
    assert _jsonl_path(cfg).exists()
    assert _summary_path(cfg).exists()
    assert _manifest_path(cfg).exists()

    # Summary has exactly 6 type entries.
    summary = json.loads(_summary_path(cfg).read_text())
    assert len(summary["per_type"]) == 6
    drop_types = {entry["type"] for entry in summary["per_type"]}
    assert drop_types == {"proposal", "evidence", "correction",
                          "agreement", "question", "other"}


def test_cli_delta_is_acc_with_minus_acc_without(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _seed_eligible(cfg, "qa")

    # Replay always returns "#### 3" → correct_without=False.
    # pre_final is "7" → correct_with=True → delta = 1.0 - 0.0 = 1.0.
    backend = DummyBackend(lambda m, mo, t: "#### 3")
    run_ablation_cli(cfg=cfg, target_size=1, max_new_llm_calls=500, backend=backend)
    summary = json.loads(_summary_path(cfg).read_text())
    for entry in summary["per_type"]:
        if entry["n_used"] > 0:
            assert entry["acc_with"] == 1.0
            assert entry["acc_without"] == 0.0
            assert entry["delta"] == pytest.approx(1.0)


def test_cli_dry_run_returns_0_without_writing(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _seed_eligible(cfg, "qa")
    rc = main(["--dry-run"], cfg=cfg)
    assert rc == 0
    assert not _summary_path(cfg).exists()


def test_cli_exit_1_when_subset_empty(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    rc = main(["--n", "5"], cfg=cfg,
              backend=DummyBackend(lambda m, mo, t: "#### 7"))
    assert rc == 1


def test_cli_report_returns_2_when_missing(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    rc = main(["--report"], cfg=cfg)
    assert rc == 2


def test_cli_report_prints_when_present(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _seed_eligible(cfg, "qa")
    run_ablation_cli(cfg=cfg, target_size=1, max_new_llm_calls=500,
                     backend=DummyBackend(lambda m, mo, t: "#### 7"))
    rc = main(["--report"], cfg=cfg)
    assert rc == 0
    assert "delta" in capsys.readouterr().out


def test_cli_enforces_max_calls_flag(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _seed_eligible(cfg, "qa")
    _seed_eligible(cfg, "qb")
    _seed_eligible(cfg, "qc")
    # Vary responses so no cache reuse.
    counter = {"n": 0}

    def resp(m, mo, t):
        counter["n"] += 1
        return f"#### {counter['n']}"

    backend = DummyBackend(resp)
    manifest = run_ablation_cli(cfg=cfg, target_size=3, max_new_llm_calls=6,
                                backend=backend)
    skipped = manifest["counts"]["skipped"]
    assert skipped > 0


def _seed_dialogue_only(cfg: Config, qid: str) -> None:
    """Seed dialogue + pilot single but NO claims — matches L1 HPC state."""
    d = Dialogue(
        question_id=qid, question="Q", gold_answer="7",
        messages=[
            Message(agent_id=0, round=1, text="ok #### 7"),
            Message(agent_id=1, round=1, text="agree #### 7"),
            Message(agent_id=0, round=2, text="#### 7"),
            Message(agent_id=1, round=2, text="#### 7"),
        ],
        final_answer="7",
    )
    cfg.dialogues_dir.mkdir(parents=True, exist_ok=True)
    (cfg.dialogues_dir / f"{qid}.json").write_text(d.model_dump_json())
    pilot_dir = cfg.artifacts_dir / "pilot" / "single" / cfg.model_slug
    pilot_dir.mkdir(parents=True, exist_ok=True)
    (pilot_dir / f"{qid}.json").write_text(json.dumps({
        "question_id": qid, "question": "Q", "gold_answer": "7",
        "messages": [{"agent_id": 0, "round": 1, "text": "#### 3"}],
        "final_answer": "3", "meta": {},
    }))


def test_control_works_without_claim_artifacts(tmp_path):
    """--control must succeed even when claims/ dir has no files."""
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _seed_dialogue_only(cfg, "qa")
    _seed_dialogue_only(cfg, "qb")
    backend = DummyBackend(lambda m, mo, t: "#### 7")
    rc = main(["--control", "--n", "5"], cfg=cfg, backend=backend)
    assert rc == 0
