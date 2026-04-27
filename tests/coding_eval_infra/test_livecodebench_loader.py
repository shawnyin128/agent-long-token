"""LiveCodeBench loader: contest_date filter + cap."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentdiet.eval.datasets import LiveCodeBenchDataset


def _write_fixture(path: Path, entries: list[dict]) -> None:
    path.write_text(json.dumps(entries), encoding="utf-8")


def _entry(qid: str, contest_date: str) -> dict:
    return {
        "qid": qid,
        "contest_date": contest_date,
        "prompt": f"# {qid}\nWrite a function f.",
        "entry_point": "f",
        "public_tests": [{"name": "ex", "script": "assert f() == 0"}],
        "hidden_tests": [{"name": "h", "script": "assert f() == 0"}],
    }


def test_filters_pre_cutoff_dates(tmp_path):
    fixture = tmp_path / "lcb.json"
    _write_fixture(fixture, [
        _entry("q1", "2024-07-15"),  # before cutoff
        _entry("q2", "2024-08-01"),  # at cutoff
        _entry("q3", "2025-03-12"),  # after cutoff
    ])
    ds = LiveCodeBenchDataset(fixture_path=fixture)
    qs = ds.load()
    qids = [q.qid for q in qs]
    assert qids == ["q2", "q3"]


def test_caps_at_80(tmp_path):
    fixture = tmp_path / "lcb.json"
    entries = [_entry(f"q{i:03d}", "2025-01-01") for i in range(120)]
    _write_fixture(fixture, entries)
    ds = LiveCodeBenchDataset(fixture_path=fixture)
    qs = ds.load()
    assert len(qs) == 80


def test_custom_cap(tmp_path):
    fixture = tmp_path / "lcb.json"
    entries = [_entry(f"q{i:03d}", "2025-01-01") for i in range(50)]
    _write_fixture(fixture, entries)
    ds = LiveCodeBenchDataset(fixture_path=fixture, cap=10)
    qs = ds.load()
    assert len(qs) == 10


def test_dataset_attrs():
    ds = LiveCodeBenchDataset(fixture_path=Path("/dev/null"))
    assert ds.name == "livecodebench"
    assert ds.domain == "code"


def test_no_fixture_no_package_raises():
    """Without fixture and without livecodebench/evalscope installed,
    load() should raise ImportError pointing at the extras group."""
    ds = LiveCodeBenchDataset(fixture_path=None)
    with pytest.raises(ImportError, match="code_eval"):
        ds.load()


def test_test_cases_round_trip(tmp_path):
    fixture = tmp_path / "lcb.json"
    _write_fixture(fixture, [{
        "qid": "qx",
        "contest_date": "2025-06-15",
        "prompt": "p",
        "entry_point": "f",
        "public_tests": [
            {"name": "p1", "script": "assert f(1)==1"},
            {"name": "p2", "script": "assert f(2)==4"},
        ],
        "hidden_tests": [
            {"name": "h1", "script": "assert f(3)==9"},
        ],
    }])
    qs = LiveCodeBenchDataset(fixture_path=fixture).load()
    assert len(qs) == 1
    q = qs[0]
    assert len(q.public_tests) == 2
    assert q.public_tests[1].name == "p2"
    assert len(q.hidden_tests) == 1
