from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentdiet.cli.report import main, run_report_cli
from agentdiet.config import Config


def _cfg(tmp_path: Path) -> Config:
    return Config(
        artifacts_dir=tmp_path, hf_cache_dir=tmp_path / "hf",
        model="test-model",
    )


def _seed_artifacts(cfg: Config) -> None:
    # Claims
    cfg.claims_dir.mkdir(parents=True, exist_ok=True)
    claim = {"id": "q1_r1_a0_c0", "text": "x", "agent_id": 0, "round": 1,
             "type": "proposal", "source_message_span": [0, 1]}
    (cfg.claims_dir / "q1.json").write_text(json.dumps({
        "qid": "q1", "claims": [claim], "per_message_status": [],
        "extraction_failed": False,
    }))
    # Ablation summary
    cfg.analysis_dir.mkdir(parents=True, exist_ok=True)
    (cfg.analysis_dir / "ablation_summary.json").write_text(json.dumps({
        "per_type": [
            {"type": "proposal", "delta": 0.1, "n_used": 5,
             "acc_with": 1.0, "acc_without": 0.9},
            {"type": "evidence", "delta": 0.05, "n_used": 5,
             "acc_with": 1.0, "acc_without": 0.95},
        ],
    }))
    # Evaluation results
    cfg.evaluation_dir.mkdir(parents=True, exist_ok=True)
    (cfg.evaluation_dir / "results.json").write_text(json.dumps({
        "per_question": [],
        "per_method": [
            {"method": "b1", "accuracy": 0.9, "total_tokens": 10000,
             "acc_per_1k": 0.09, "n_evaluated": 100},
            {"method": "ours", "accuracy": 0.85, "total_tokens": 4000,
             "acc_per_1k": 0.21, "n_evaluated": 100},
        ],
        "invariant_violations": [],
    }))


def test_run_report_cli_writes_tables(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _seed_artifacts(cfg)

    reports_dir = tmp_path / "reports"
    result = run_report_cli(cfg=cfg, reports_dir=reports_dir)
    # Tables emitted regardless of matplotlib availability.
    assert (reports_dir / "tables" / "baselines.tex").exists()
    assert (reports_dir / "tables" / "claim_stats.tex").exists()
    assert result["tables"]


def test_run_report_cli_skips_figures_without_matplotlib(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _seed_artifacts(cfg)

    # Force matplotlib unavailable.
    import agentdiet.report as rpt

    def boom():
        raise ImportError("matplotlib not installed")

    monkeypatch.setattr(rpt, "_ensure_matplotlib", boom)
    reports_dir = tmp_path / "reports"
    result = run_report_cli(cfg=cfg, reports_dir=reports_dir)
    # Figures skipped but tables + metadata still returned.
    assert result["figures_skipped"] is True
    assert (reports_dir / "tables" / "baselines.tex").exists()


def test_cli_dry_run_exits_0(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    rc = main(["--dry-run"], cfg=cfg)
    assert rc == 0


def test_cli_exit_2_when_artifacts_missing(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    # No ablation_summary or results.json.
    rc = main([], cfg=cfg)
    assert rc == 2


def test_cli_happy_path_returns_0(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _seed_artifacts(cfg)
    rc = main([], cfg=cfg, reports_dir=tmp_path / "reports")
    assert rc == 0
