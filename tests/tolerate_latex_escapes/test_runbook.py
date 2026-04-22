from __future__ import annotations

import re
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _readme() -> str:
    return (PROJECT_ROOT / "README.md").read_text()


def test_readme_documents_latex_escape_recovery():
    r = _readme()
    assert "extract-clean" in r and "extract" in r
    # Root-cause hint so readers know when to invoke the recovery flow.
    assert "\\(" in r or "LaTeX" in r or "\\escape" in r
