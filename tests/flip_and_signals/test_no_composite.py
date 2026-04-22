"""Architectural guard: spec §5.2 prohibits a composite importance
score. This test scans ``agentdiet/analysis/`` for any identifier that
would implement such a shortcut and fails if one sneaks in."""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from agentdiet.analysis.signals import SIGNAL_KEYS, compute_signals, HashingFakeEmbedder


ANALYSIS_DIR = (Path(__file__).resolve().parents[2] / "agentdiet" / "analysis")
FORBIDDEN_REGEX = re.compile(
    r"^(?:composite|importance_score|weighted_score|combined_score|importance)$",
    re.IGNORECASE,
)


def _collect_identifiers(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            found.add(node.target.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
    return found


def test_no_composite_identifier_in_analysis_package():
    hits: dict[str, set[str]] = {}
    for py in ANALYSIS_DIR.rglob("*.py"):
        names = _collect_identifiers(py)
        bad = {n for n in names if FORBIDDEN_REGEX.match(n)}
        if bad:
            hits[str(py.relative_to(ANALYSIS_DIR.parent))] = bad
    assert not hits, (
        f"Forbidden composite-score identifiers found "
        f"(spec §5.2 prohibits): {hits}"
    )


def test_compute_signals_returns_exactly_5_keys_per_row():
    rows = compute_signals(
        [{
            "id": "c0", "text": "x", "agent_id": 0, "round": 1,
            "type": "proposal", "source_message_span": [0, 1],
        }],
        flip_events=[],
        embedder=HashingFakeEmbedder(),
    )
    assert len(rows) == 1
    assert set(rows[0].keys()) == set(SIGNAL_KEYS)
    assert len(SIGNAL_KEYS) == 5


def test_signal_keys_ordering_is_documented():
    # Freeze the tuple so downstream consumers can rely on ordering.
    assert SIGNAL_KEYS == (
        "claim_id", "flip_coincidence", "novelty",
        "referenced_later", "position",
    )
