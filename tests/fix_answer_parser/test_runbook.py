from __future__ import annotations

import re
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _makefile() -> str:
    return (PROJECT_ROOT / "Makefile").read_text()


def _readme() -> str:
    return (PROJECT_ROOT / "README.md").read_text()


def test_makefile_declares_reparse():
    mf = _makefile()
    phony = re.search(r"\.PHONY:\s*([\s\S]*?)\n\n", mf)
    assert phony and "reparse" in phony.group(1)
    assert re.search(r"^reparse:\s*\n(\t.+\n)+", mf, re.MULTILINE)


def test_readme_mentions_reparse_and_root_cause():
    r = _readme()
    assert "make reparse" in r
    # Root-cause hint so future readers understand why the step exists.
    assert "#### N" in r or "literal 'N'" in r or "parser bug" in r.lower()
