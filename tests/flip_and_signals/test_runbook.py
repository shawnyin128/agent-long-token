from __future__ import annotations

import re
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _makefile() -> str:
    return (PROJECT_ROOT / "Makefile").read_text()


def _readme() -> str:
    return (PROJECT_ROOT / "README.md").read_text()


@pytest.mark.parametrize("target", ["analyze", "analyze-report", "analyze-clean"])
def test_makefile_declares_target(target):
    mf = _makefile()
    phony_block = re.search(r"\.PHONY:\s*([\s\S]*?)\n\n", mf)
    assert phony_block is not None
    assert target in phony_block.group(1)
    recipe = re.search(rf"^{re.escape(target)}:\s*\n(\t.+\n)+", mf, re.MULTILINE)
    assert recipe is not None, f"no recipe for {target}"


def test_readme_mentions_analysis_runbook():
    r = _readme()
    assert re.search(r"#{1,3}\s*Analysis", r, re.IGNORECASE)
    assert "make analyze" in r


def test_analyze_clean_only_removes_analysis_dir():
    mf = _makefile()
    match = re.search(r"^analyze-clean:\s*\n\t(.+)\n", mf, re.MULTILINE)
    assert match is not None
    recipe = match.group(1)
    assert "artifacts/analysis" in recipe
    assert "llm_cache" not in recipe
    assert "artifacts/dialogues" not in recipe
    assert "artifacts/claims" not in recipe


def test_readme_documents_analysis_extras_requirement():
    r = _readme()
    assert ".[analysis]" in r or "[analysis]" in r, \
        "README must tell users to install .[analysis] extras for real embedder"
