from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from agentdiet.cli.collect import main
from agentdiet.config import Config


REPO = Path(__file__).resolve().parent.parent.parent


def test_report_manifest_missing_exits_2(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTDIET_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("AGENTDIET_HF_CACHE_DIR", str(tmp_path / "hf"))
    rc = main(["--report-manifest"])
    assert rc == 2


def test_report_manifest_prints_counts(tmp_path, monkeypatch, capsys):
    # seed a fake manifest
    monkeypatch.setenv("AGENTDIET_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("AGENTDIET_HF_CACHE_DIR", str(tmp_path / "hf"))
    monkeypatch.setenv("AGENTDIET_MODEL", "test-model")
    dialogues_dir = tmp_path / "dialogues"
    dialogues_dir.mkdir(parents=True)
    manifest = {
        "model": "test-model",
        "n": 5,
        "seed": 42,
        "n_agents": 3,
        "n_rounds": 3,
        "counts": {"ok": 4, "cached": 0, "unparsed": 0, "failed": 1},
        "outcomes": [],
    }
    (dialogues_dir / "manifest.json").write_text(json.dumps(manifest))
    rc = main(["--report-manifest"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "ok" in captured.out and "4" in captured.out
    assert "failed" in captured.out and "1" in captured.out


@pytest.mark.skipif(shutil.which("make") is None, reason="make not installed")
@pytest.mark.parametrize("target", ["collect", "collect-report", "collect-clean"])
def test_make_target_dry_run(target):
    result = subprocess.run(
        ["make", "-n", target],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
