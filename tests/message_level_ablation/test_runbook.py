from __future__ import annotations

import re
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _readme() -> str:
    return (PROJECT_ROOT / "README.md").read_text()


def test_readme_mentions_message_level_default():
    r = _readme()
    # Either an explicit "message-level" phrase, or the --granularity flag name.
    assert "message-level" in r or "--granularity" in r


def test_readme_mentions_gate2_null_cause():
    r = _readme()
    # Some hint that span-level causes artifact null results on sparse claims.
    assert (
        "span-level" in r or
        "granularity" in r or
        "coverage" in r.lower()
    )
