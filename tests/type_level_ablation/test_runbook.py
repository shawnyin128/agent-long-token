from __future__ import annotations

import re
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _makefile() -> str:
    return (PROJECT_ROOT / "Makefile").read_text()


def _readme() -> str:
    return (PROJECT_ROOT / "README.md").read_text()


@pytest.mark.parametrize("target", ["ablate", "ablate-report", "ablate-clean", "gate2"])
def test_makefile_declares_target(target):
    mf = _makefile()
    phony = re.search(r"\.PHONY:\s*([\s\S]*?)\n\n", mf)
    assert phony and target in phony.group(1)
    recipe = re.search(rf"^{re.escape(target)}:\s*\n(\t.+\n)+", mf, re.MULTILINE)
    assert recipe, f"no recipe for {target}"


def test_readme_mentions_type_level_ablation():
    r = _readme()
    assert re.search(r"type.?level ablation|Gate 2", r, re.IGNORECASE)
    assert "make ablate" in r
    assert "make gate2" in r


def test_readme_documents_500_call_cap():
    r = _readme()
    assert "500" in r


def test_ablate_clean_does_not_touch_other_artifacts():
    mf = _makefile()
    m = re.search(r"^ablate-clean:\s*\n\t(.+)\n", mf, re.MULTILINE)
    assert m
    recipe = m.group(1)
    assert "artifacts/analysis/ablation" in recipe
    assert "artifacts/dialogues" not in recipe
    assert "artifacts/claims" not in recipe
    assert "llm_cache" not in recipe
