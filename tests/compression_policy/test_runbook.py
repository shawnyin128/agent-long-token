from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from agentdiet.compress import Policy


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _makefile() -> str:
    return (PROJECT_ROOT / "Makefile").read_text()


def _readme() -> str:
    return (PROJECT_ROOT / "README.md").read_text()


def _sample_path() -> Path:
    return PROJECT_ROOT / "docs" / "policy.sample.json"


def test_sample_policy_is_valid_json():
    data = json.loads(_sample_path().read_text())
    assert "mode" in data


def test_sample_policy_passes_model_validation():
    data = json.loads(_sample_path().read_text())
    p = Policy.model_validate(data)
    assert p.mode == "ours"


def test_makefile_declares_policy_sample_target():
    mf = _makefile()
    phony = re.search(r"\.PHONY:\s*([\s\S]*?)\n\n", mf)
    assert phony and "policy-sample" in phony.group(1)
    recipe = re.search(r"^policy-sample:\s*\n(\t.+\n)+", mf, re.MULTILINE)
    assert recipe, "no recipe for policy-sample"


def test_readme_has_compression_section():
    r = _readme()
    assert re.search(r"Day.?3 compression", r, re.IGNORECASE) \
        or re.search(r"#{1,3}\s*Compression policy", r, re.IGNORECASE)


def test_readme_mentions_all_five_modes():
    r = _readme()
    for mode in ("b1", "b2", "b3", "b5", "ours"):
        assert f"`{mode}`" in r or f" {mode} " in r, f"README missing mode {mode}"


def test_policy_sample_mode_copy_does_not_clobber_existing_policy(tmp_path):
    # Semantics of "policy-sample": cp -n (no-clobber). Confirm recipe uses it.
    mf = _makefile()
    m = re.search(r"^policy-sample:\s*\n\t(.+)\n", mf, re.MULTILINE)
    assert m
    recipe = m.group(1)
    assert "-n" in recipe or "no-clobber" in recipe, \
        "policy-sample must be no-clobber so user edits aren't overwritten"
