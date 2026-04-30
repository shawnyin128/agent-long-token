"""cli/analyze_phase.py end-to-end: --grid-dir → analysis.json + tables + figure."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentdiet.cli.analyze_phase import main


def _write_cell(
    grid_dir: Path, cell_dirname: str, *,
    sa_correct: list[bool], voting_correct: list[bool],
    debate_correct: list[bool],
    model: str = "Qwen/Qwen3-30B-A3B", family: str = "qwen3",
    dataset_name: str = "gsm8k", thinking: bool = False,
    prompt_variant: str = "cooperative", over_budget_factor: float = 1.0,
    qids: list[str] | None = None,
) -> None:
    cell_path = grid_dir / cell_dirname
    cell_path.mkdir(parents=True, exist_ok=True)
    cell_block = {
        "model": model, "model_family": family,
        "dataset_name": dataset_name, "thinking": thinking,
        "prompt_variant": prompt_variant,
    }
    qs = qids or [f"q{i}" for i in range(len(sa_correct))]

    def _record(condition: str, correct: list[bool]) -> dict:
        return {
            "condition": condition, "cell": cell_block,
            "questions": [
                {"qid": qs[i], "gold": "1",
                 "final_answer": "1" if c else "9",
                 "correct": c, "prompt_tokens": 10,
                 "completion_tokens": 5, "total_tokens": 15,
                 "meta": {}}
                for i, c in enumerate(correct)
            ],
            "n_evaluated": len(correct),
            "accuracy": sum(correct) / len(correct),
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
        "delta_debate_voting": 0.0,
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


def test_main_writes_analysis_json_tables_and_figure(tmp_path):
    pytest.importorskip("matplotlib")
    grid = tmp_path / "grid"
    out = tmp_path / "out"
    _write_cell(grid, "Qwen__Qwen3-30B-A3B__gsm8k__t0",
                sa_correct=[True, False] * 5,
                voting_correct=[True] * 10,
                debate_correct=[True] * 10)
    _write_cell(grid, "Qwen__Qwen3-30B-A3B__gsm8k__t1",
                sa_correct=[True] * 10,
                voting_correct=[False, True] * 5,
                debate_correct=[True] * 10,
                thinking=True)
    _write_cell(grid, "openai__gpt-oss-20b__humaneval_plus__t0",
                sa_correct=[True] * 8 + [False] * 2,
                voting_correct=[True] * 7 + [False] * 3,
                debate_correct=[True] * 9 + [False],
                model="openai/gpt-oss-20b", family="gpt-oss",
                dataset_name="humaneval_plus")

    rc = main([
        "--grid-dir", str(grid),
        "--output-dir", str(out),
        "--n-resamples", "200",
    ])
    assert rc == 0
    # data
    data_path = out / "data" / "analysis.json"
    assert data_path.is_file()
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    for key in ("meta", "cells", "aime", "cross_model",
                "thinking_o1", "thinking_o2"):
        assert key in payload
    assert len(payload["cells"]) == 3
    # tables
    for name in ("phase_summary.tex", "aime_per_year.tex",
                 "cross_model.tex", "thinking_o1.tex", "thinking_o2.tex"):
        path = out / "tables" / name
        assert path.is_file()
        text = path.read_text(encoding="utf-8")
        assert "\\begin{tabular}" in text
        assert "\\toprule" in text
    # figure
    fig = out / "figs" / "phase_diagram.pdf"
    assert fig.is_file()
    assert fig.stat().st_size > 1024


def test_skip_figure_suppresses_pdf(tmp_path):
    grid = tmp_path / "grid"
    out = tmp_path / "out"
    _write_cell(grid, "cell1",
                sa_correct=[True, True], voting_correct=[True, False],
                debate_correct=[True, True])
    rc = main([
        "--grid-dir", str(grid), "--output-dir", str(out),
        "--n-resamples", "100", "--skip-figure",
    ])
    assert rc == 0
    assert (out / "data" / "analysis.json").is_file()
    assert not (out / "figs" / "phase_diagram.pdf").exists()


def test_main_empty_grid_returns_exit_2(tmp_path):
    grid = tmp_path / "empty_grid"
    grid.mkdir()
    rc = main([
        "--grid-dir", str(grid),
        "--output-dir", str(tmp_path / "out"),
        "--n-resamples", "10",
    ])
    assert rc == 2


def test_main_missing_grid_dir_returns_exit_2(tmp_path):
    rc = main([
        "--grid-dir", str(tmp_path / "nope"),
        "--output-dir", str(tmp_path / "out"),
        "--n-resamples", "10",
    ])
    assert rc == 2


def test_main_case_study_writes_separate_json(tmp_path):
    grid = tmp_path / "grid"
    out = tmp_path / "out"
    _write_cell(grid, "case_cell",
                sa_correct=[True, True], voting_correct=[False, True],
                debate_correct=[True, True])
    rc = main([
        "--grid-dir", str(grid), "--output-dir", str(out),
        "--n-resamples", "100", "--skip-figure",
        "--case-study-cell", "case_cell",
    ])
    assert rc == 0
    cs_path = out / "data" / "voting_wrong_debate_right.json"
    assert cs_path.is_file()
    payload = json.loads(cs_path.read_text(encoding="utf-8"))
    assert payload["cell"] == "case_cell"
    assert any(r["qid"] == "q0" for r in payload["rows"])


def test_main_aime_cell_produces_per_year_rows(tmp_path):
    grid = tmp_path / "grid"
    out = tmp_path / "out"
    qids = ([f"aime-2024-q{i}" for i in range(2)] +
            [f"aime-2025-q{i}" for i in range(2)] +
            [f"aime-2026-q{i}" for i in range(2)])
    _write_cell(grid, "aime_cell",
                sa_correct=[True] * 6,
                voting_correct=[False, False, True, True, True, True],
                debate_correct=[True] * 6,
                dataset_name="aime", qids=qids)
    rc = main([
        "--grid-dir", str(grid), "--output-dir", str(out),
        "--n-resamples", "100", "--skip-figure",
    ])
    assert rc == 0
    payload = json.loads((out / "data" / "analysis.json").read_text(encoding="utf-8"))
    years = sorted(r["year"] for r in payload["aime"])
    assert years == [2024, 2025, 2026]
