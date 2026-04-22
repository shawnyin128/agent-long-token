from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentdiet.cli.evaluate import main, run_evaluation_cli, _results_path
from agentdiet.config import Config
from agentdiet.llm_client import DummyBackend
from agentdiet.types import Dialogue, Message


def _cfg(tmp_path: Path) -> Config:
    return Config(
        artifacts_dir=tmp_path, hf_cache_dir=tmp_path / "hf",
        model="test-model", seed=42,
    )


def _seed_pair(cfg: Config, qid: str, gold: str = "7") -> None:
    d = Dialogue(
        question_id=qid, question=f"Q {qid}", gold_answer=gold,
        messages=[
            Message(agent_id=0, round=1, text="work through it #### 7"),
            Message(agent_id=1, round=1, text="agree #### 7"),
            Message(agent_id=0, round=2, text="#### 7"),
            Message(agent_id=1, round=2, text="#### 7"),
        ],
        final_answer="7",
    )
    cfg.dialogues_dir.mkdir(parents=True, exist_ok=True)
    (cfg.dialogues_dir / f"{qid}.json").write_text(d.model_dump_json())

    claims = [
        {"id": f"{qid}_r1_a0_c0", "text": "t", "agent_id": 0, "round": 1,
         "type": "evidence", "source_message_span": [0, 8]},
    ]
    cfg.claims_dir.mkdir(parents=True, exist_ok=True)
    (cfg.claims_dir / f"{qid}.json").write_text(json.dumps({
        "qid": qid, "claims": claims, "per_message_status": [],
        "extraction_failed": False,
    }))


def _write_policy(cfg: Config) -> Path:
    path = cfg.compression_dir / "policy.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mode": "ours", "drop_types": ["other"]}))
    return path


def test_cli_writes_results_json(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _seed_pair(cfg, "qa")
    _seed_pair(cfg, "qb")
    _write_policy(cfg)

    backend = DummyBackend(lambda m, mo, t: "#### 7")
    manifest = run_evaluation_cli(cfg=cfg, n=2, backend=backend)
    assert _results_path(cfg).exists()
    data = json.loads(_results_path(cfg).read_text())
    for k in ("per_question", "per_method", "invariant_violations", "config"):
        assert k in data


def test_cli_dry_run_exits_0_without_writing(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _seed_pair(cfg, "qa")
    _write_policy(cfg)
    rc = main(["--dry-run"], cfg=cfg)
    assert rc == 0
    assert not _results_path(cfg).exists()


def test_cli_exit_1_when_no_eligible_qids(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _write_policy(cfg)
    rc = main([], cfg=cfg, backend=DummyBackend(lambda m, mo, t: "#### 7"))
    assert rc == 1


def test_cli_exit_2_when_policy_missing(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _seed_pair(cfg, "qa")
    # No policy.json written.
    rc = main([], cfg=cfg, backend=DummyBackend(lambda m, mo, t: "#### 7"))
    assert rc == 2


def test_cli_report_exits_2_when_missing(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    rc = main(["--report"], cfg=cfg)
    assert rc == 2


def test_cli_loads_signal_scores_from_parquet(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _seed_pair(cfg, "qa")
    # Write a policy that uses signal-based filter so the loader path
    # is exercised end-to-end.
    (cfg.compression_dir / "policy.json").write_text(json.dumps({
        "mode": "ours", "drop_low_novelty": 0.5,
    }))
    # Write signal_scores.parquet with one row for qa's claim.
    table = pa.table({
        "qid": ["qa"], "claim_id": ["qa_r1_a0_c0"],
        "flip_coincidence": [False], "novelty": [0.1],
        "referenced_later": [False], "position": [1],
    })
    pq.write_table(table, cfg.analysis_dir / "signal_scores.parquet")

    backend = DummyBackend(lambda m, mo, t: "#### 7")
    run_evaluation_cli(cfg=cfg, n=1, backend=backend)
    data = json.loads(_results_path(cfg).read_text())
    # At least one per_question row for method=ours exists.
    ours_rows = [r for r in data["per_question"] if r["method"] == "ours"]
    assert len(ours_rows) == 1


def test_cli_report_prints_per_method_summary(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _seed_pair(cfg, "qa")
    _write_policy(cfg)
    run_evaluation_cli(cfg=cfg, n=1,
                       backend=DummyBackend(lambda m, mo, t: "#### 7"))
    rc = main(["--report"], cfg=cfg)
    assert rc == 0
    out = capsys.readouterr().out
    assert "b1" in out and "acc_per_1k" in out
