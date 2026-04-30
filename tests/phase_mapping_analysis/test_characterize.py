"""Per-axis characterization helpers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentdiet.analysis_phase.bootstrap import CellAnalysis
from agentdiet.analysis_phase.characterize import (
    aime_per_year,
    cross_model_agreement,
    thinking_axis_observations,
    voting_wrong_debate_right,
)


def _cell(
    *,
    cell_dirname: str = "cell",
    family: str = "qwen3",
    dataset_name: str = "gsm8k",
    thinking: bool = False,
    prompt_variant: str = "cooperative",
    sa: float = 0.7,
    voting: float = 0.7,
    debate: float = 0.75,
    ci_low: float = 0.0,
    ci_high: float = 0.10,
    qids: list[str] | None = None,
    sa_per_q: list[bool] | None = None,
    voting_per_q: list[bool] | None = None,
    debate_per_q: list[bool] | None = None,
) -> CellAnalysis:
    qs = qids or [f"q{i}" for i in range(5)]
    n = len(qs)
    return CellAnalysis(
        cell_dirname=cell_dirname,
        model=f"{family}/test", model_family=family,
        dataset_name=dataset_name, thinking=thinking,
        prompt_variant=prompt_variant,
        sa_accuracy=sa, voting_accuracy=voting, debate_accuracy=debate,
        delta_debate_voting=debate - voting,
        delta_ci_low=ci_low, delta_ci_high=ci_high,
        over_budget_factor=1.0,
        sa_per_q_correct=sa_per_q or [True] * n,
        voting_per_q_correct=voting_per_q or [True] * n,
        debate_per_q_correct=debate_per_q or [True] * n,
        qids=qs, n_questions=n,
    )


# --- aime_per_year ----------------------------------------------------------


def test_aime_per_year_partitions_by_year_prefix():
    qids = (
        [f"aime-2026-q{i:02d}" for i in range(3)] +
        [f"aime-2025-q{i:02d}" for i in range(3)] +
        [f"aime-2024-q{i:02d}" for i in range(2)]
    )
    n = len(qids)
    cell = _cell(
        dataset_name="aime", qids=qids,
        sa_per_q=[True] * n, voting_per_q=[True] * n,
        debate_per_q=[True, False, True, True, False, True, True, True],
    )
    rows = aime_per_year([cell])
    assert len(rows) == 3
    by_year = {r["year"]: r for r in rows}
    assert by_year[2026]["n"] == 3
    assert by_year[2025]["n"] == 3
    assert by_year[2024]["n"] == 2


def test_aime_per_year_skips_non_aime_cells():
    cell = _cell(dataset_name="gsm8k")
    rows = aime_per_year([cell])
    assert rows == []


def test_aime_per_year_delta_uses_per_year_acc():
    """If 2024 is all correct in debate but voting all wrong, delta=1.0
    just for 2024."""
    qids = ([f"aime-2024-q{i}" for i in range(3)] +
            [f"aime-2025-q{i}" for i in range(3)])
    cell = _cell(
        dataset_name="aime", qids=qids,
        sa_per_q=[True] * 6, voting_per_q=[False] * 3 + [True] * 3,
        debate_per_q=[True] * 3 + [True] * 3,
    )
    rows = aime_per_year([cell])
    by_year = {r["year"]: r for r in rows}
    assert by_year[2024]["delta_debate_voting"] == 1.0
    assert by_year[2025]["delta_debate_voting"] == 0.0


# --- cross_model_agreement -------------------------------------------------


def test_cross_model_agreement_two_families_same_dataset_thinking():
    cells = [
        _cell(family="qwen3", debate=0.8, voting=0.7,
              ci_low=0.05, ci_high=0.15),
        _cell(family="gpt-oss", debate=0.85, voting=0.75,
              ci_low=0.04, ci_high=0.16),
    ]
    rows = cross_model_agreement(cells)
    assert len(rows) == 1
    row = rows[0]
    assert sorted(row["families"]) == ["gpt-oss", "qwen3"]
    assert row["signs_agree"] is True
    assert row["cis_overlap"] is True


def test_cross_model_agreement_signs_disagree():
    cells = [
        _cell(family="qwen3", debate=0.8, voting=0.7,
              ci_low=0.05, ci_high=0.15),
        _cell(family="gpt-oss", debate=0.55, voting=0.75,
              ci_low=-0.30, ci_high=-0.05),
    ]
    rows = cross_model_agreement(cells)
    assert rows[0]["signs_agree"] is False
    assert rows[0]["cis_overlap"] is False


def test_cross_model_agreement_single_family_returns_empty():
    cells = [_cell(family="qwen3"), _cell(family="qwen3", thinking=True)]
    rows = cross_model_agreement(cells)
    assert rows == []


def test_cross_model_agreement_groups_by_dataset_and_thinking():
    cells = [
        # gsm8k thinking-off pair
        _cell(family="qwen3", dataset_name="gsm8k", thinking=False),
        _cell(family="gpt-oss", dataset_name="gsm8k", thinking=False),
        # gsm8k thinking-on pair (different group)
        _cell(family="qwen3", dataset_name="gsm8k", thinking=True),
        _cell(family="gpt-oss", dataset_name="gsm8k", thinking=True),
    ]
    rows = cross_model_agreement(cells)
    assert len(rows) == 2
    keys = sorted((r["dataset"], r["thinking"]) for r in rows)
    assert keys == [("gsm8k", False), ("gsm8k", True)]


# --- voting_wrong_debate_right --------------------------------------------


def _write_cell_for_case_study(
    grid_dir: Path, cell_dirname: str,
    voting_correct: list[bool], debate_correct: list[bool],
):
    cell_path = grid_dir / cell_dirname
    cell_path.mkdir(parents=True)

    def _record(condition: str, correct: list[bool]) -> dict:
        return {
            "condition": condition,
            "questions": [
                {"qid": f"q{i}", "gold": "1",
                 "final_answer": "1" if c else "9",
                 "correct": c, "prompt_tokens": 0,
                 "completion_tokens": 0, "total_tokens": 30,
                 "meta": {"response_preview": f"resp-{condition}-{i}"}}
                for i, c in enumerate(correct)
            ],
        }

    (cell_path / "voting.json").write_text(json.dumps(_record("voting", voting_correct)))
    (cell_path / "debate.json").write_text(json.dumps(_record("debate", debate_correct)))


def test_voting_wrong_debate_right_finds_matching_qids(tmp_path):
    _write_cell_for_case_study(
        tmp_path, "cell",
        voting_correct=[False, True, False, True, False],
        debate_correct=[True, True, False, True, True],
    )
    rows = voting_wrong_debate_right(tmp_path, "cell")
    qids = [r["qid"] for r in rows]
    # voting=False, debate=True at indices 0 and 4
    assert qids == ["q0", "q4"]


def test_voting_wrong_debate_right_respects_max_results(tmp_path):
    _write_cell_for_case_study(
        tmp_path, "cell",
        voting_correct=[False] * 10,
        debate_correct=[True] * 10,
    )
    rows = voting_wrong_debate_right(tmp_path, "cell", max_results=3)
    assert len(rows) == 3


def test_voting_wrong_debate_right_empty_when_no_match(tmp_path):
    _write_cell_for_case_study(
        tmp_path, "cell",
        voting_correct=[True, True, True],
        debate_correct=[True, True, True],
    )
    rows = voting_wrong_debate_right(tmp_path, "cell")
    assert rows == []


# --- thinking_axis_observations -------------------------------------------


def test_thinking_axis_observations_pairs_on_off():
    cells = [
        _cell(family="qwen3", dataset_name="gsm8k", thinking=False,
              debate=0.8, voting=0.75, sa=0.70),
        _cell(family="qwen3", dataset_name="gsm8k", thinking=True,
              debate=0.85, voting=0.82, sa=0.80),
    ]
    out = thinking_axis_observations(cells)
    assert len(out["o1"]) == 1
    o1 = out["o1"][0]
    assert o1["delta_off"] == pytest.approx(0.05)
    assert o1["delta_on"] == pytest.approx(0.03)
    assert o1["delta_change"] == pytest.approx(-0.02)

    assert len(out["o2"]) == 1
    o2 = out["o2"][0]
    # debate_off=0.8 vs sa_on=0.80 -> tie
    assert o2["winner"] == "tie"
    assert o2["magnitude"] == pytest.approx(0.0)


def test_thinking_axis_observations_o2_winner_picks_larger():
    cells = [
        _cell(thinking=False, debate=0.7, voting=0.6, sa=0.5),
        _cell(thinking=True, debate=0.85, voting=0.8, sa=0.6),
    ]
    out = thinking_axis_observations(cells)
    o2 = out["o2"][0]
    assert o2["debate_thinking_off"] == 0.7
    assert o2["sa_thinking_on"] == 0.6
    assert o2["winner"] == "debate_off"
    assert o2["magnitude"] == pytest.approx(0.1)


def test_thinking_axis_observations_skips_unmatched_groups():
    """If only thinking-off exists for a group, no row appears."""
    cells = [
        _cell(family="qwen3", dataset_name="gsm8k", thinking=False),
        # No qwen3+gsm8k thinking=True paired here
        _cell(family="gpt-oss", dataset_name="aime", thinking=True),
    ]
    out = thinking_axis_observations(cells)
    assert out["o1"] == []
    assert out["o2"] == []


def test_thinking_axis_observations_groups_by_prompt_variant():
    cells = [
        _cell(family="qwen3", thinking=False, prompt_variant="cooperative"),
        _cell(family="qwen3", thinking=True, prompt_variant="cooperative"),
        _cell(family="qwen3", thinking=False, prompt_variant="symmetric"),
        _cell(family="qwen3", thinking=True, prompt_variant="symmetric"),
    ]
    out = thinking_axis_observations(cells)
    assert len(out["o1"]) == 2
    variants = sorted(r["prompt_variant"] for r in out["o1"])
    assert variants == ["cooperative", "symmetric"]
