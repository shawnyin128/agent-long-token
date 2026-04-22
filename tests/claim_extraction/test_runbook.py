from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _makefile() -> str:
    return (PROJECT_ROOT / "Makefile").read_text()


def _readme() -> str:
    return (PROJECT_ROOT / "README.md").read_text()


@pytest.mark.parametrize("target", ["extract", "extract-report", "extract-clean", "spot-check"])
def test_makefile_declares_target(target):
    mf = _makefile()
    # Target appears in .PHONY
    phony_block = re.search(r"\.PHONY:\s*([\s\S]*?)\n\n", mf)
    assert phony_block is not None
    assert target in phony_block.group(1)
    # Target has a recipe.
    recipe = re.search(rf"^{re.escape(target)}:\s*\n(\t.+\n)+", mf, re.MULTILINE)
    assert recipe is not None, f"no recipe for {target}"


def test_readme_has_claim_extraction_section():
    r = _readme()
    assert re.search(r"#{1,3}\s*Claim extraction", r, re.IGNORECASE), \
        "README missing a 'Claim extraction' heading"
    assert "make extract" in r
    assert "make spot-check" in r


def test_extract_clean_only_removes_claims_dir():
    mf = _makefile()
    match = re.search(r"^extract-clean:\s*\n\t(.+)\n", mf, re.MULTILINE)
    assert match is not None
    recipe = match.group(1)
    assert "artifacts/claims" in recipe
    # Must NOT touch the LLM cache or dialogues in this recipe line.
    assert "llm_cache" not in recipe
    assert "artifacts/dialogues" not in recipe


def test_make_help_mentions_claim_extraction():
    # Just verify the help line mentions "Claim extraction" so the runbook
    # is discoverable.
    mf = _makefile()
    assert "Claim extraction" in mf or "claim extraction" in mf
