"""End-to-end fixture run + schema regression on a 4-cell synthetic grid."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentdiet.cli.analyze_phase import main


def _write_cell(
    grid_dir: Path, cell_dirname: str, *,
    sa_correct: list[bool], voting_correct: list[bool],
    debate_correct: list[bool], total_tokens_factor: int,
    family: str, dataset_name: str, thinking: bool,
    prompt_variant: str = "cooperative", over_budget_factor: float = 1.0,
    qids: list[str] | None = None,
) -> None:
    cell_path = grid_dir / cell_dirname
    cell_path.mkdir(parents=True, exist_ok=True)
    model_id = {"qwen3": "Qwen/Qwen3-30B-A3B",
                "gpt-oss": "openai/gpt-oss-20b"}[family]
    cell_block = {
        "model": model_id, "model_family": family,
        "dataset_name": dataset_name, "thinking": thinking,
        "prompt_variant": prompt_variant,
    }
    n = len(sa_correct)
    qs = qids or [f"q{i}" for i in range(n)]

    def _record(condition: str, correct: list[bool]) -> dict:
        return {
            "condition": condition, "cell": cell_block,
            "questions": [
                {"qid": qs[i], "gold": "1",
                 "final_answer": "1" if c else "9",
                 "correct": c, "prompt_tokens": 10,
                 "completion_tokens": 5,
                 "total_tokens": 15 * total_tokens_factor,
                 "meta": {}}
                for i, c in enumerate(correct)
            ],
            "n_evaluated": n,
            "accuracy": sum(correct) / n,
            "total_tokens": 15 * total_tokens_factor * n,
            "meta": {},
        }

    (cell_path / "sa.json").write_text(json.dumps(_record("sa", sa_correct)))
    (cell_path / "voting.json").write_text(json.dumps(_record("voting", voting_correct)))
    (cell_path / "debate.json").write_text(json.dumps(_record("debate", debate_correct)))
    summary = {
        "cell": cell_block,
        "sa_accuracy": sum(sa_correct) / n,
        "voting_accuracy": sum(voting_correct) / n,
        "debate_accuracy": sum(debate_correct) / n,
        "sa_total_tokens": 15 * total_tokens_factor * n,
        "voting_total_tokens": 15 * total_tokens_factor * n,
        "debate_total_tokens": 15 * total_tokens_factor * n * 9,
        "delta_debate_voting": (sum(debate_correct) - sum(voting_correct)) / n,
        "delta_debate_sa": (sum(debate_correct) - sum(sa_correct)) / n,
        "calibration": {
            "N_raw": 9, "N": 9,
            "mean_debate_tokens": 30.0 * 9, "mean_sa_tokens": 30.0,
            "over_budget_factor": over_budget_factor, "floor_active": False,
            "calibration_prefix_n": 5,
        },
        "n_questions": n,
    }
    (cell_path / "summary.json").write_text(json.dumps(summary))


@pytest.fixture
def four_cell_grid(tmp_path: Path) -> Path:
    """Synthetic 4-cell grid: Qwen3 + GSM8K x t0/t1, GPT-OSS + HumanEval+ x t0/t1."""
    grid = tmp_path / "grid"
    correctness_pattern = ([True, False] * 3 + [True, True])  # 8 questions

    _write_cell(grid, "Qwen__Qwen3-30B-A3B__gsm8k__t0",
                sa_correct=correctness_pattern,
                voting_correct=[True] * 8,
                debate_correct=[True] * 7 + [False],
                total_tokens_factor=2,
                family="qwen3", dataset_name="gsm8k", thinking=False)
    _write_cell(grid, "Qwen__Qwen3-30B-A3B__gsm8k__t1",
                sa_correct=[True] * 7 + [False],
                voting_correct=[True] * 6 + [False, False],
                debate_correct=[True] * 8,
                total_tokens_factor=4,
                family="qwen3", dataset_name="gsm8k", thinking=True,
                over_budget_factor=1.5)  # open marker
    _write_cell(grid, "openai__gpt-oss-20b__humaneval_plus__t0",
                sa_correct=[True] * 5 + [False] * 3,
                voting_correct=[True] * 6 + [False] * 2,
                debate_correct=[True] * 7 + [False],
                total_tokens_factor=3,
                family="gpt-oss", dataset_name="humaneval_plus", thinking=False)
    _write_cell(grid, "openai__gpt-oss-20b__humaneval_plus__t1",
                sa_correct=[True] * 6 + [False] * 2,
                voting_correct=[True] * 5 + [False] * 3,
                debate_correct=[True] * 8,
                total_tokens_factor=6,
                family="gpt-oss", dataset_name="humaneval_plus", thinking=True)
    return grid


def test_end_to_end_full_pipeline(tmp_path, four_cell_grid):
    """analyze.main on the 4-cell fixture writes the full output tree."""
    pytest.importorskip("matplotlib")
    out = tmp_path / "out"
    rc = main([
        "--grid-dir", str(four_cell_grid),
        "--output-dir", str(out),
        "--n-resamples", "200",
    ])
    assert rc == 0
    # Data
    assert (out / "data" / "analysis.json").is_file()
    # Tables
    for name in ("phase_summary.tex", "aime_per_year.tex",
                 "cross_model.tex", "thinking_o1.tex", "thinking_o2.tex"):
        assert (out / "tables" / name).is_file()
    # Figure
    fig = out / "figs" / "phase_diagram.pdf"
    assert fig.is_file()
    assert fig.stat().st_size > 1024


def test_analysis_json_top_level_keys(tmp_path, four_cell_grid):
    out = tmp_path / "out"
    main([
        "--grid-dir", str(four_cell_grid),
        "--output-dir", str(out),
        "--n-resamples", "200",
        "--skip-figure",
    ])
    payload = json.loads(
        (out / "data" / "analysis.json").read_text(encoding="utf-8")
    )
    expected = {"meta", "cells", "aime", "cross_model",
                "thinking_o1", "thinking_o2"}
    assert expected.issubset(payload.keys())
    assert len(payload["cells"]) == 4


def test_per_cell_row_schema(tmp_path, four_cell_grid):
    out = tmp_path / "out"
    main([
        "--grid-dir", str(four_cell_grid),
        "--output-dir", str(out),
        "--n-resamples", "200",
        "--skip-figure",
    ])
    payload = json.loads(
        (out / "data" / "analysis.json").read_text(encoding="utf-8")
    )
    for row in payload["cells"]:
        for key in ("cell_dirname", "model", "model_family",
                    "dataset_name", "thinking", "prompt_variant",
                    "sa_accuracy", "voting_accuracy", "debate_accuracy",
                    "delta_debate_voting", "delta_ci_low", "delta_ci_high",
                    "over_budget_factor", "n_questions"):
            assert key in row, f"row missing key {key!r}"


def test_delta_ci_bounds_well_formed(tmp_path, four_cell_grid):
    """For every cell, ci_low <= delta <= ci_high."""
    out = tmp_path / "out"
    main([
        "--grid-dir", str(four_cell_grid),
        "--output-dir", str(out),
        "--n-resamples", "200",
        "--skip-figure",
    ])
    payload = json.loads(
        (out / "data" / "analysis.json").read_text(encoding="utf-8")
    )
    for row in payload["cells"]:
        assert row["delta_ci_low"] <= row["delta_debate_voting"] <= row["delta_ci_high"]


def test_thinking_observations_pair_both_states(tmp_path, four_cell_grid):
    """4-cell fixture has 2 (model, dataset) pairs each with both thinking
    states -> thinking_o1 and thinking_o2 each have 2 rows."""
    out = tmp_path / "out"
    main([
        "--grid-dir", str(four_cell_grid),
        "--output-dir", str(out),
        "--n-resamples", "100",
        "--skip-figure",
    ])
    payload = json.loads(
        (out / "data" / "analysis.json").read_text(encoding="utf-8")
    )
    assert len(payload["thinking_o1"]) == 2
    assert len(payload["thinking_o2"]) == 2


def test_no_aime_cells_yields_empty_aime_block(tmp_path, four_cell_grid):
    out = tmp_path / "out"
    main([
        "--grid-dir", str(four_cell_grid),
        "--output-dir", str(out),
        "--n-resamples", "100",
        "--skip-figure",
    ])
    payload = json.loads(
        (out / "data" / "analysis.json").read_text(encoding="utf-8")
    )
    # Fixture has no AIME cells
    assert payload["aime"] == []


def test_cross_model_groups_emerge_under_matching_dataset_thinking(tmp_path):
    """Add Qwen3 + GPT-OSS on the same dataset / thinking and assert
    cross_model_agreement returns at least one row."""
    pytest.importorskip("matplotlib")
    grid = tmp_path / "grid"
    correctness_pattern = [True] * 5 + [False] * 3

    _write_cell(grid, "Qwen__Qwen3-30B-A3B__humaneval_plus__t0",
                sa_correct=correctness_pattern,
                voting_correct=[True] * 6 + [False] * 2,
                debate_correct=[True] * 7 + [False],
                total_tokens_factor=2,
                family="qwen3", dataset_name="humaneval_plus", thinking=False)
    _write_cell(grid, "openai__gpt-oss-20b__humaneval_plus__t0",
                sa_correct=correctness_pattern,
                voting_correct=[True] * 7 + [False],
                debate_correct=[True] * 8,
                total_tokens_factor=3,
                family="gpt-oss", dataset_name="humaneval_plus", thinking=False)

    out = tmp_path / "out"
    main([
        "--grid-dir", str(grid),
        "--output-dir", str(out),
        "--n-resamples", "100",
        "--skip-figure",
    ])
    payload = json.loads(
        (out / "data" / "analysis.json").read_text(encoding="utf-8")
    )
    assert len(payload["cross_model"]) == 1
    assert sorted(payload["cross_model"][0]["families"]) == ["gpt-oss", "qwen3"]
