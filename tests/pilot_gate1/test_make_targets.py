from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent.parent


@pytest.mark.skipif(shutil.which("make") is None, reason="make not installed")
@pytest.mark.parametrize("target", ["serve", "stop", "status", "health", "pilot", "gate", "pilot-clean"])
def test_make_target_dry_run(target):
    result = subprocess.run(
        ["make", "-n", target],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"make -n {target} failed: {result.stderr}"


def test_readme_contains_runbook():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    for marker in [
        "## Gate 1 Runbook",
        "make serve",
        "make health",
        "make pilot",
        "make gate",
    ]:
        assert marker in readme, f"missing {marker!r} in README"


def test_scripts_referenced_by_makefile_exist():
    mk = (REPO / "Makefile").read_text(encoding="utf-8")
    for path in ["scripts/serve_vllm.sh", "scripts/wait_healthy.py"]:
        assert path in mk
        assert (REPO / path).exists(), f"{path} referenced in Makefile but missing"


def test_serve_script_is_executable():
    p = REPO / "scripts" / "serve_vllm.sh"
    assert p.exists()
    # executable bit set
    assert p.stat().st_mode & 0o111, "serve_vllm.sh missing executable bit"
