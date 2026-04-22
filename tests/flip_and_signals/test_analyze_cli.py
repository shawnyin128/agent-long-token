from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from agentdiet.cli.analyze import main, run_analysis, _manifest_path
from agentdiet.config import Config
from agentdiet.types import Dialogue, Message


def _cfg(tmp_path: Path) -> Config:
    return Config(
        artifacts_dir=tmp_path,
        hf_cache_dir=tmp_path / "hf",
        model="test-model",
        n_agents=3, n_rounds=2,
    )


def _write_pair(cfg: Config, qid: str, *, flip: bool) -> None:
    # Round 1 wrong, round 2 right (if flip=True) → creates a flip event.
    r1 = ["3", "3", "3"] if flip else ["7", "7", "7"]
    r2 = ["7", "7", "7"]
    messages = []
    for r_idx, ans_per_agent in enumerate((r1, r2), 1):
        for a_idx, ans in enumerate(ans_per_agent):
            messages.append(Message(
                agent_id=a_idx, round=r_idx,
                text=f"Answer is {ans}. #### {ans}",
            ))
    d = Dialogue(
        question_id=qid, question="What is 3+4?", gold_answer="7",
        messages=messages, final_answer="7",
    )
    cfg.dialogues_dir.mkdir(parents=True, exist_ok=True)
    (cfg.dialogues_dir / f"{qid}.json").write_text(d.model_dump_json())

    claims = []
    for m in messages:
        claims.append({
            "id": f"{qid}_r{m.round}_a{m.agent_id}_c0",
            "text": m.text,
            "agent_id": m.agent_id, "round": m.round,
            "type": "proposal",
            "source_message_span": [0, min(10, len(m.text))],
        })
    cfg.claims_dir.mkdir(parents=True, exist_ok=True)
    (cfg.claims_dir / f"{qid}.json").write_text(json.dumps({
        "qid": qid, "claims": claims, "per_message_status": [],
        "extraction_failed": False,
    }))


def test_run_analysis_writes_all_three_artifacts(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _write_pair(cfg, "qa", flip=True)
    _write_pair(cfg, "qb", flip=False)

    manifest = run_analysis(cfg, use_fake_embedder=True)

    flip_path = cfg.analysis_dir / "flip_events.jsonl"
    sig_path = cfg.analysis_dir / "signal_scores.parquet"
    assert flip_path.exists()
    assert sig_path.exists()
    assert _manifest_path(cfg).exists()

    # qa has a flip event, qb does not.
    lines = [json.loads(l) for l in flip_path.read_text().splitlines() if l]
    assert len(lines) == 1
    assert lines[0]["qid"] == "qa"
    assert lines[0]["round"] == 2
    assert manifest["counts"]["qids_processed"] == 2
    assert manifest["counts"]["flip_events"] == 1


def test_parquet_schema_and_row_count(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _write_pair(cfg, "qa", flip=True)

    run_analysis(cfg, use_fake_embedder=True)
    table = pq.read_table(cfg.analysis_dir / "signal_scores.parquet")
    cols = set(table.column_names)
    assert cols == {"qid", "claim_id", "flip_coincidence", "novelty",
                    "referenced_later", "position"}
    # 3 agents × 2 rounds × 1 claim/message = 6 signal rows.
    assert table.num_rows == 6


def test_manifest_counts_match_output(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _write_pair(cfg, "qa", flip=True)
    _write_pair(cfg, "qb", flip=True)
    _write_pair(cfg, "qc", flip=False)

    manifest = run_analysis(cfg, use_fake_embedder=True)
    assert manifest["counts"]["qids_processed"] == 3
    assert manifest["counts"]["flip_events"] == 2
    # 3 qids × 3 agents × 2 rounds = 18 signal rows total.
    assert manifest["counts"]["signal_rows"] == 18


def test_dry_run_returns_0_and_does_not_touch_embedder(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _write_pair(cfg, "qa", flip=True)

    rc = main(["--dry-run"], cfg=cfg)
    assert rc == 0
    # Output artifacts should NOT have been written.
    assert not (cfg.analysis_dir / "signal_scores.parquet").exists()


def test_exit_1_when_no_eligible_qids(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    # No dialogues or claims — nothing to process.
    rc = main([], cfg=cfg, use_fake_embedder=True)
    assert rc == 1


def test_qid_skipped_when_claims_missing(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _write_pair(cfg, "qa", flip=True)
    # Write a dialogue with no matching claims artifact.
    d = Dialogue(
        question_id="qb", question="Q", gold_answer="7",
        messages=[Message(agent_id=0, round=1, text="#### 7")],
        final_answer="7",
    )
    (cfg.dialogues_dir / "qb.json").write_text(d.model_dump_json())

    manifest = run_analysis(cfg, use_fake_embedder=True)
    assert manifest["counts"]["qids_processed"] == 1
    assert manifest["counts"]["qids_skipped_no_claims"] == 1


def test_report_returns_2_when_missing(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    rc = main(["--report"], cfg=cfg)
    assert rc == 2


def test_report_prints_counts_when_present(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _write_pair(cfg, "qa", flip=True)
    run_analysis(cfg, use_fake_embedder=True)
    rc = main(["--report"], cfg=cfg)
    assert rc == 0
    assert "flip_events" in capsys.readouterr().out
