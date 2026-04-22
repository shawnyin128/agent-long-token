from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentdiet.cli.ablate import (
    GATE2_LIKELY_THRESHOLD, GATE2_NOISE_THRESHOLD, GATE2_PASS_THRESHOLD,
    _classify_delta, _render_gate2, _gate2_path, _summary_path, main,
)
from agentdiet.config import Config


def _cfg(tmp_path: Path) -> Config:
    return Config(artifacts_dir=tmp_path, hf_cache_dir=tmp_path / "hf",
                  model="test-model")


def _summary_with_deltas(deltas: dict[str, float]) -> dict:
    return {
        "per_type": [
            {"type": t, "n_used": 10, "n_skipped": 0,
             "acc_with": 1.0, "acc_without": 1.0 - d, "delta": d}
            for t, d in deltas.items()
        ]
    }


def test_classify_likely():
    assert _classify_delta(0.15) == "likely"
    assert _classify_delta(-0.20) == "likely"
    assert _classify_delta(GATE2_LIKELY_THRESHOLD) == "likely"


def test_classify_noise():
    assert _classify_delta(0.01) == "noise"
    assert _classify_delta(-0.02) == "noise"
    assert _classify_delta(GATE2_NOISE_THRESHOLD) == "noise"


def test_classify_unclear():
    assert _classify_delta(0.05) == "unclear"
    assert _classify_delta(-0.08) == "unclear"


def test_render_pass_verdict_at_pass_threshold():
    summary = _summary_with_deltas({
        "proposal": 0.12, "evidence": 0.01, "correction": 0.0,
        "agreement": 0.02, "question": 0.0, "other": 0.0,
    })
    report, exit_code = _render_gate2(summary)
    assert "PASS" in report
    assert exit_code == 0


def test_render_null_result_when_all_within_noise():
    summary = _summary_with_deltas({
        "proposal": 0.01, "evidence": 0.0, "correction": -0.02,
        "agreement": 0.03, "question": -0.01, "other": 0.0,
    })
    report, exit_code = _render_gate2(summary)
    assert "NULL_RESULT" in report
    assert exit_code == 10
    assert "descriptive comparison" in report


def test_render_inconclusive_when_between_thresholds():
    summary = _summary_with_deltas({
        "proposal": 0.04, "evidence": 0.04, "correction": 0.04,
        "agreement": 0.04, "question": 0.04, "other": 0.04,
    })
    report, exit_code = _render_gate2(summary)
    assert "INCONCLUSIVE" in report
    assert exit_code == 20


def test_gate2_cli_writes_report_and_returns_exit_code(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    summary = _summary_with_deltas({
        "proposal": 0.15, "evidence": 0.0, "correction": 0.0,
        "agreement": 0.0, "question": 0.0, "other": 0.0,
    })
    _summary_path(cfg).parent.mkdir(parents=True, exist_ok=True)
    _summary_path(cfg).write_text(json.dumps(summary))
    rc = main(["--gate2"], cfg=cfg)
    assert rc == 0
    assert _gate2_path(cfg).exists()
    body = _gate2_path(cfg).read_text()
    assert "| proposal " in body


def test_gate2_cli_returns_2_when_summary_missing(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    rc = main(["--gate2"], cfg=cfg)
    assert rc == 2


def test_gate2_thresholds_ordered():
    assert GATE2_NOISE_THRESHOLD < GATE2_PASS_THRESHOLD < GATE2_LIKELY_THRESHOLD
