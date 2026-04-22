from __future__ import annotations

import re
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _makefile() -> str:
    return (PROJECT_ROOT / "Makefile").read_text()


def _readme() -> str:
    return (PROJECT_ROOT / "README.md").read_text()


@pytest.mark.parametrize("target", ["evaluate", "evaluate-report", "evaluate-clean"])
def test_makefile_declares_target(target):
    mf = _makefile()
    phony = re.search(r"\.PHONY:\s*([\s\S]*?)\n\n", mf)
    assert phony and target in phony.group(1)
    assert re.search(rf"^{re.escape(target)}:\s*\n(\t.+\n)+", mf, re.MULTILINE)


def test_readme_has_evaluation_section():
    r = _readme()
    assert re.search(r"#{1,3}\s*Evaluation", r, re.IGNORECASE)
    assert "make evaluate" in r


def test_readme_mentions_all_4_sanity_invariants():
    r = _readme()
    # acc(b1) >= acc(b2)
    assert "acc(b1)" in r and "acc(b2)" in r
    assert "acc(ours)" in r and "acc(b5)" in r
    assert "tokens(b1)" in r and "tokens(b3)" in r
    assert "tokens(ours)" in r


def test_evaluate_clean_only_removes_evaluation_dir():
    mf = _makefile()
    m = re.search(r"^evaluate-clean:\s*\n\t(.+)\n", mf, re.MULTILINE)
    assert m
    recipe = m.group(1)
    assert "artifacts/evaluation" in recipe
    for other in ("artifacts/dialogues", "artifacts/claims",
                  "artifacts/analysis", "llm_cache"):
        assert other not in recipe
