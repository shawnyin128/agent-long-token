"""load_cell_summary + compute_per_cell_analysis on fixture grid trees."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentdiet.analysis_phase.bootstrap import (
    compute_per_cell_analysis,
    load_cell_summary,
)


def _write_cell(
    grid_dir: Path, cell_dirname: str, *,
    sa_correct: list[bool], voting_correct: list[bool],
    debate_correct: list[bool],
    model: str = "Qwen/Qwen3-30B-A3B",
    family: str = "qwen3",
    dataset_name: str = "gsm8k",
    thinking: bool = False,
    prompt_variant: str = "cooperative",
    over_budget_factor: float = 1.0,
) -> None:
    cell_path = grid_dir / cell_dirname
    cell_path.mkdir(parents=True, exist_ok=True)
    cell_block = {
        "model": model,
        "model_family": family,
        "dataset_name": dataset_name,
        "thinking": thinking,
        "prompt_variant": prompt_variant,
    }

    def _record(condition: str, correct: list[bool]) -> dict:
        return {
            "condition": condition,
            "cell": cell_block,
            "questions": [
                {"qid": f"q{i}", "gold": "1",
                 "final_answer": "1" if c else "2",
                 "correct": c, "prompt_tokens": 10,
                 "completion_tokens": 5, "total_tokens": 15,
                 "meta": {}}
                for i, c in enumerate(correct)
            ],
            "n_evaluated": len(correct),
            "accuracy": sum(correct) / len(correct) if correct else 0.0,
            "total_tokens": 15 * len(correct),
            "meta": {},
        }

    (cell_path / "sa.json").write_text(json.dumps(_record("sa", sa_correct)))
    (cell_path / "voting.json").write_text(json.dumps(_record("voting", voting_correct)))
    (cell_path / "debate.json").write_text(json.dumps(_record("debate", debate_correct)))
    summary = {
        "cell": cell_block,
        "sa_accuracy": sum(sa_correct) / len(sa_correct),
        "voting_accuracy": sum(voting_correct) / len(voting_correct),
        "debate_accuracy": sum(debate_correct) / len(debate_correct),
        "sa_total_tokens": 15 * len(sa_correct),
        "voting_total_tokens": 15 * len(voting_correct),
        "debate_total_tokens": 15 * len(debate_correct),
        "delta_debate_voting": (sum(debate_correct) / len(debate_correct))
                                - (sum(voting_correct) / len(voting_correct)),
        "delta_debate_sa": 0.0,
        "calibration": {
            "N_raw": 10, "N": 10,
            "mean_debate_tokens": 30.0, "mean_sa_tokens": 30.0,
            "over_budget_factor": over_budget_factor, "floor_active": False,
            "calibration_prefix_n": 5,
        },
        "n_questions": len(sa_correct),
    }
    (cell_path / "summary.json").write_text(json.dumps(summary))


def test_load_cell_summary_reads_all_three_correctness_vectors(tmp_path):
    _write_cell(
        tmp_path, "Qwen__Qwen3-30B-A3B__gsm8k__t0",
        sa_correct=[True, False, True, False, True],
        voting_correct=[True, True, False, False, True],
        debate_correct=[True, True, True, False, True],
    )
    cell = load_cell_summary(
        tmp_path, "Qwen__Qwen3-30B-A3B__gsm8k__t0",
        n_resamples=200,
    )
    assert cell.sa_per_q_correct == [True, False, True, False, True]
    assert cell.voting_per_q_correct == [True, True, False, False, True]
    assert cell.debate_per_q_correct == [True, True, True, False, True]


def test_load_cell_summary_preserves_metadata(tmp_path):
    _write_cell(
        tmp_path, "openai__gpt-oss-20b__livecodebench__t1__pv-symmetric",
        sa_correct=[True], voting_correct=[True], debate_correct=[True],
        model="openai/gpt-oss-20b", family="gpt-oss",
        dataset_name="livecodebench", thinking=True,
        prompt_variant="symmetric", over_budget_factor=1.5,
    )
    cell = load_cell_summary(
        tmp_path, "openai__gpt-oss-20b__livecodebench__t1__pv-symmetric",
        n_resamples=100,
    )
    assert cell.model == "openai/gpt-oss-20b"
    assert cell.model_family == "gpt-oss"
    assert cell.dataset_name == "livecodebench"
    assert cell.thinking is True
    assert cell.prompt_variant == "symmetric"
    assert cell.over_budget_factor == 1.5


def test_load_cell_summary_missing_file_raises(tmp_path):
    cell_path = tmp_path / "broken_cell"
    cell_path.mkdir()
    # Only one of four files present
    (cell_path / "sa.json").write_text(json.dumps({"questions": []}))
    with pytest.raises(FileNotFoundError, match="missing artifact"):
        load_cell_summary(tmp_path, "broken_cell", n_resamples=10)


def test_load_cell_summary_question_count_disagreement_raises(tmp_path):
    _write_cell(
        tmp_path, "mismatch_cell",
        sa_correct=[True, True],
        voting_correct=[True, True, True],  # different length
        debate_correct=[True, True],
    )
    with pytest.raises(ValueError, match="question counts disagree"):
        load_cell_summary(tmp_path, "mismatch_cell", n_resamples=10)


def test_compute_per_cell_analysis_returns_one_per_cell_dir(tmp_path):
    for name in ("a_cell", "b_cell", "c_cell"):
        _write_cell(
            tmp_path, name,
            sa_correct=[True, False],
            voting_correct=[True, True],
            debate_correct=[False, True],
        )
    cells = compute_per_cell_analysis(tmp_path, n_resamples=200)
    assert len(cells) == 3
    assert sorted(c.cell_dirname for c in cells) == ["a_cell", "b_cell", "c_cell"]


def test_compute_per_cell_analysis_skips_incomplete_cells(tmp_path):
    _write_cell(tmp_path, "complete",
                sa_correct=[True], voting_correct=[True], debate_correct=[True])
    incomplete = tmp_path / "in_progress"
    incomplete.mkdir()
    (incomplete / "sa.json").write_text(json.dumps({"questions": []}))
    cells = compute_per_cell_analysis(tmp_path, n_resamples=100)
    assert [c.cell_dirname for c in cells] == ["complete"]


def test_compute_per_cell_analysis_missing_grid_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="grid_dir does not exist"):
        compute_per_cell_analysis(tmp_path / "nonexistent", n_resamples=10)


def test_load_cell_summary_delta_matches_summary(tmp_path):
    """Bootstrap delta on identical inputs equals (debate - voting)
    accuracy from summary.json (deterministic 0/1 sample)."""
    _write_cell(
        tmp_path, "delta_check",
        sa_correct=[True, False],
        voting_correct=[False, False],   # acc 0.0
        debate_correct=[True, True],     # acc 1.0
    )
    cell = load_cell_summary(tmp_path, "delta_check", n_resamples=200)
    assert cell.delta_debate_voting == pytest.approx(1.0)
    assert cell.delta_ci_low == pytest.approx(1.0)
    assert cell.delta_ci_high == pytest.approx(1.0)
