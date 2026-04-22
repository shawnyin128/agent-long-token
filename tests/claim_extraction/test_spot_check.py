from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from agentdiet.cli.spot_check import main, sample_and_write, SPOT_CHECK_COLUMNS
from agentdiet.config import Config
from agentdiet.types import Dialogue, Message


def _cfg(tmp_path: Path) -> Config:
    return Config(
        artifacts_dir=tmp_path,
        hf_cache_dir=tmp_path / "hf",
        model="test-model",
        n_agents=2, n_rounds=1, seed=42,
    )


def _write_pair(cfg: Config, qid: str) -> None:
    # Write a dialogue artifact.
    d = Dialogue(
        question_id=qid, question="What is 3+4?", gold_answer="7",
        messages=[
            Message(agent_id=0, round=1, text="3 + 4 = 7. #### 7"),
            Message(agent_id=1, round=1, text="I agree. #### 7"),
        ],
        final_answer="7",
    )
    cfg.dialogues_dir.mkdir(parents=True, exist_ok=True)
    (cfg.dialogues_dir / f"{qid}.json").write_text(d.model_dump_json())

    # Write a claim artifact in the format run_extraction produces.
    claim = {
        "id": f"{qid}_r1_a0_c0", "text": "3+4=7",
        "agent_id": 0, "round": 1, "type": "evidence",
        "source_message_span": [0, 9],
    }
    result = {
        "qid": qid,
        "claims": [claim],
        "per_message_status": [],
        "extraction_failed": False,
    }
    cfg.claims_dir.mkdir(parents=True, exist_ok=True)
    (cfg.claims_dir / f"{qid}.json").write_text(json.dumps(result))


def test_csv_columns_exact(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _write_pair(cfg, "qa")
    sample_and_write(cfg, k=1)

    csv_path = cfg.artifacts_dir / "spot_check.csv"
    assert csv_path.exists()
    with csv_path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
    assert header == list(SPOT_CHECK_COLUMNS)


def test_k_selects_distinct_qids(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    for qid in ("qa", "qb", "qc", "qd", "qe"):
        _write_pair(cfg, qid)
    sample_and_write(cfg, k=3)

    csv_path = cfg.artifacts_dir / "spot_check.csv"
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    qids = {r["qid"] for r in rows}
    assert len(qids) == 3
    assert qids.issubset({"qa", "qb", "qc", "qd", "qe"})


def test_seeded_sample_is_deterministic(tmp_path):
    cfg1 = _cfg(tmp_path / "a")
    cfg1.ensure_dirs()
    for qid in ("qa", "qb", "qc", "qd", "qe", "qf"):
        _write_pair(cfg1, qid)
    sample_and_write(cfg1, k=3)
    rows1 = (cfg1.artifacts_dir / "spot_check.csv").read_text()

    cfg2 = _cfg(tmp_path / "b")
    cfg2.ensure_dirs()
    for qid in ("qa", "qb", "qc", "qd", "qe", "qf"):
        _write_pair(cfg2, qid)
    sample_and_write(cfg2, k=3)
    rows2 = (cfg2.artifacts_dir / "spot_check.csv").read_text()

    assert rows1 == rows2  # same seed, same corpus, same output


def test_raises_when_fewer_artifacts_than_k(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _write_pair(cfg, "qa")
    _write_pair(cfg, "qb")
    with pytest.raises(ValueError):
        sample_and_write(cfg, k=5)


def test_markdown_companion_is_written(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    for qid in ("qa", "qb"):
        _write_pair(cfg, qid)
    sample_and_write(cfg, k=2)
    md = (cfg.artifacts_dir / "spot_check_notes.md").read_text()
    assert "qa" in md and "qb" in md
    assert "3 + 4 = 7" in md


def test_blank_manual_columns(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _write_pair(cfg, "qa")
    sample_and_write(cfg, k=1)
    with (cfg.artifacts_dir / "spot_check.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["manual_pass"] == ""
    assert rows[0]["notes"] == ""


def test_main_cli_happy_path(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    for qid in ("qa", "qb"):
        _write_pair(cfg, qid)
    rc = main(["--k", "2"], cfg=cfg)
    assert rc == 0
    assert (cfg.artifacts_dir / "spot_check.csv").exists()


def test_main_cli_returns_nonzero_when_corpus_too_small(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _write_pair(cfg, "qa")
    rc = main(["--k", "10"], cfg=cfg)
    assert rc == 2
