"""AIMEMultiYearDataset loader: stratified sampling 30/30/20 with seed 42."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentdiet.eval.datasets import (
    AIME_2024_SAMPLE,
    AIME_FULL_PER_YEAR,
    AIME_TOTAL,
    AIMEMultiYearDataset,
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "aime"


def test_total_size_is_80():
    qs = AIMEMultiYearDataset(data_dir=FIXTURE_DIR).load()
    assert len(qs) == AIME_TOTAL == 80


def test_year_breakdown_30_30_20():
    qs = AIMEMultiYearDataset(data_dir=FIXTURE_DIR).load()
    by_year: dict[int, int] = {}
    for q in qs:
        # qid format: aime-<year>-q<idx>
        year = int(q.qid.split("-")[1])
        by_year[year] = by_year.get(year, 0) + 1
    assert by_year == {2024: AIME_2024_SAMPLE,
                       2025: AIME_FULL_PER_YEAR,
                       2026: AIME_FULL_PER_YEAR}


def test_sampling_2024_deterministic_seed_42():
    a = AIMEMultiYearDataset(data_dir=FIXTURE_DIR).load()
    b = AIMEMultiYearDataset(data_dir=FIXTURE_DIR).load()
    a_2024 = sorted([q.qid for q in a if "-2024-" in q.qid])
    b_2024 = sorted([q.qid for q in b if "-2024-" in q.qid])
    assert a_2024 == b_2024


def test_sampling_2024_changes_with_different_seed():
    a = AIMEMultiYearDataset(data_dir=FIXTURE_DIR, sample_seed=42).load()
    b = AIMEMultiYearDataset(data_dir=FIXTURE_DIR, sample_seed=99).load()
    a_2024 = {q.qid for q in a if "-2024-" in q.qid}
    b_2024 = {q.qid for q in b if "-2024-" in q.qid}
    # Different seeds should produce a different sample (at least one
    # qid should differ across the 20-of-30 picks)
    assert a_2024 != b_2024


def test_year_question_count_mismatch_raises(tmp_path):
    """If 2026 has !=30 questions, loader raises."""
    bad = tmp_path / "aime_2026.json"
    bad.write_text(json.dumps({
        "year": 2026,
        "questions": [
            {"id": f"2026-{i}", "problem": "p", "answer": "0"}
            for i in range(28)  # not 30
        ],
    }))
    # Mirror the other two years from fixtures
    (tmp_path / "aime_2024.json").write_text(
        (FIXTURE_DIR / "aime_2024.json").read_text())
    (tmp_path / "aime_2025.json").write_text(
        (FIXTURE_DIR / "aime_2025.json").read_text())
    with pytest.raises(ValueError, match="2026 has 28"):
        AIMEMultiYearDataset(data_dir=tmp_path).load()


def test_missing_year_file_raises(tmp_path):
    # Only ship 2024 + 2025; 2026 is missing.
    (tmp_path / "aime_2024.json").write_text(
        (FIXTURE_DIR / "aime_2024.json").read_text())
    (tmp_path / "aime_2025.json").write_text(
        (FIXTURE_DIR / "aime_2025.json").read_text())
    with pytest.raises(FileNotFoundError, match="aime_2026.json"):
        AIMEMultiYearDataset(data_dir=tmp_path).load()


def test_qid_format_encodes_year_and_index():
    qs = AIMEMultiYearDataset(data_dir=FIXTURE_DIR).load()
    for q in qs:
        parts = q.qid.split("-")
        assert parts[0] == "aime"
        assert int(parts[1]) in {2024, 2025, 2026}
        assert parts[2].startswith("q")


def test_questions_carry_problem_and_gold_answer():
    qs = AIMEMultiYearDataset(data_dir=FIXTURE_DIR).load()
    q = qs[0]
    assert q.question
    assert q.gold_answer
    # Fixture answer for index i is str(i); should match
    idx = int(q.qid.split("q")[-1])
    assert q.gold_answer == str(idx)


def test_dataset_attrs():
    ds = AIMEMultiYearDataset(data_dir=FIXTURE_DIR)
    assert ds.name == "aime_multi_year"
    assert ds.domain == "math"
