"""Schema validator for smoke artifacts produced by the HPC stand-up.

This test SKIPs when the artifact files are absent (the expected
state at feature merge); after the user runs scripts/serving/smoke_*.py
on HPC and commits the JSON, this test turns from skip to pass.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _artifact_dir() -> Path:
    return Path(
        os.environ.get(
            "AGENTDIET_SERVING_ARTIFACTS_DIR",
            str(REPO_ROOT / "artifacts" / "serving"),
        )
    )


REQUIRED_TOP_KEYS = ("model", "tested_at", "thinking_on", "thinking_off", "summary")
REQUIRED_SUMMARY_KEYS = ("mean_tokens_on", "mean_tokens_off", "delta", "passed")
REQUIRED_PER_CALL_KEYS = (
    "qid", "thinking", "response_chars",
    "response_preview", "prompt_tokens", "completion_tokens",
)


def _validate_payload(payload: dict, label: str) -> None:
    for key in REQUIRED_TOP_KEYS:
        assert key in payload, f"{label}: missing top-level key {key!r}"
    assert isinstance(payload["thinking_on"], list), f"{label}: thinking_on not list"
    assert isinstance(payload["thinking_off"], list), f"{label}: thinking_off not list"
    assert len(payload["thinking_on"]) >= 1, f"{label}: empty thinking_on"
    assert len(payload["thinking_off"]) >= 1, f"{label}: empty thinking_off"
    for row in payload["thinking_on"] + payload["thinking_off"]:
        for k in REQUIRED_PER_CALL_KEYS:
            assert k in row, f"{label}: per-call row missing {k!r}"
    summary = payload["summary"]
    for k in REQUIRED_SUMMARY_KEYS:
        assert k in summary, f"{label}: summary missing {k!r}"
    assert summary["passed"] is True, (
        f"{label}: summary.passed is False — thinking toggle did not "
        f"increase output tokens (delta={summary['delta']})"
    )


@pytest.mark.parametrize(
    "filename,model_substring",
    [
        ("qwen3_smoke.json", "Qwen3"),
        ("gpt_oss_smoke.json", "gpt-oss"),
    ],
)
def test_smoke_artifact_schema_and_verdict(filename, model_substring):
    artifact_path = _artifact_dir() / filename
    if not artifact_path.is_file():
        pytest.skip(
            f"{artifact_path} not present — run "
            f"scripts/serving/smoke_*.py on HPC and commit the artifact"
        )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    _validate_payload(payload, label=filename)
    assert model_substring.lower() in payload["model"].lower()


def test_validator_catches_schema_violation(tmp_path):
    """Sanity check: a malformed payload trips the validator."""
    bad = {"model": "x", "summary": {"passed": True}}  # missing many keys
    with pytest.raises(AssertionError, match="missing top-level key"):
        _validate_payload(bad, label="bad")


def test_validator_catches_failed_summary(tmp_path):
    """A well-formed but failed-verdict payload also trips."""
    payload = {
        "model": "Qwen3-test",
        "tested_at": "2026-04-27T00:00:00Z",
        "thinking_on": [{
            "qid": "q1", "thinking": True, "response_chars": 10,
            "response_preview": "x", "prompt_tokens": 5, "completion_tokens": 5,
        }],
        "thinking_off": [{
            "qid": "q1", "thinking": False, "response_chars": 10,
            "response_preview": "x", "prompt_tokens": 5, "completion_tokens": 5,
        }],
        "summary": {
            "mean_tokens_on": 5.0, "mean_tokens_off": 5.0,
            "delta": 0.0, "passed": False,
        },
    }
    with pytest.raises(AssertionError, match="summary.passed is False"):
        _validate_payload(payload, label="failed")


def test_validator_works_on_well_formed_payload(tmp_path):
    """A well-formed, passing payload validates cleanly."""
    payload = {
        "model": "Qwen/Qwen3-30B-A3B",
        "tested_at": "2026-04-27T00:00:00Z",
        "thinking_on": [{
            "qid": "q1", "thinking": True, "response_chars": 100,
            "response_preview": "...", "prompt_tokens": 80, "completion_tokens": 400,
        }],
        "thinking_off": [{
            "qid": "q1", "thinking": False, "response_chars": 30,
            "response_preview": "...", "prompt_tokens": 80, "completion_tokens": 80,
        }],
        "summary": {
            "mean_tokens_on": 400.0, "mean_tokens_off": 80.0,
            "delta": 320.0, "passed": True,
        },
    }
    _validate_payload(payload, label="good")  # no raise


def test_artifact_dir_env_override(tmp_path, monkeypatch):
    """Env override repoints the artifact directory."""
    monkeypatch.setenv("AGENTDIET_SERVING_ARTIFACTS_DIR", str(tmp_path))
    assert _artifact_dir() == tmp_path
