"""render_phase_diagram + short_label tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from agentdiet.analysis_phase.bootstrap import CellAnalysis
from agentdiet.analysis_phase.figure_phase_diagram import (
    render_phase_diagram,
    short_label,
)


def _cell(
    cell_dirname: str = "Qwen__Qwen3-30B-A3B__gsm8k__t0",
    *,
    family: str = "qwen3",
    dataset_name: str = "gsm8k",
    thinking: bool = False,
    prompt_variant: str = "cooperative",
    sa: float = 0.7,
    voting: float = 0.75,
    debate: float = 0.78,
    ci_low: float = 0.0,
    ci_high: float = 0.06,
    over_budget: float = 1.0,
) -> CellAnalysis:
    return CellAnalysis(
        cell_dirname=cell_dirname,
        model="Qwen/Qwen3-30B-A3B",
        model_family=family,
        dataset_name=dataset_name,
        thinking=thinking,
        prompt_variant=prompt_variant,
        sa_accuracy=sa,
        voting_accuracy=voting,
        debate_accuracy=debate,
        delta_debate_voting=debate - voting,
        delta_ci_low=ci_low,
        delta_ci_high=ci_high,
        over_budget_factor=over_budget,
        sa_per_q_correct=[True] * 10,
        voting_per_q_correct=[True] * 10,
        debate_per_q_correct=[True] * 10,
        qids=[f"q{i}" for i in range(10)],
        n_questions=10,
    )


# --- short_label ---


def test_short_label_qwen3_gsm8k_off():
    cell = _cell(family="qwen3", dataset_name="gsm8k", thinking=False)
    assert short_label(cell) == "Q3-GS"


def test_short_label_thinking_on_appends_plus():
    cell = _cell(family="qwen3", dataset_name="gsm8k", thinking=True)
    assert short_label(cell) == "Q3-GS+"


def test_short_label_gpt_oss_humaneval():
    cell = _cell(family="gpt-oss", dataset_name="humaneval_plus", thinking=False)
    assert short_label(cell) == "GO-HE"


def test_short_label_includes_variant_prefix_for_non_cooperative():
    cell = _cell(prompt_variant="symmetric")
    assert "(sym" in short_label(cell)


def test_short_label_excludes_variant_for_cooperative():
    cell = _cell(prompt_variant="cooperative")
    assert "(" not in short_label(cell)


# --- render_phase_diagram ---


def test_render_writes_pdf_and_returns_path(tmp_path):
    pytest.importorskip("matplotlib")
    output = tmp_path / "phase.pdf"
    cells = [
        _cell("c1", thinking=False),
        _cell("c2", thinking=True, dataset_name="humaneval_plus"),
        _cell("c3", thinking=False, dataset_name="livecodebench",
              over_budget=1.5),  # open marker
    ]
    returned = render_phase_diagram(cells, output)
    assert returned == output
    assert output.is_file()
    # Reasonable file size for a non-empty plot
    assert output.stat().st_size > 1024


def test_render_creates_parent_dir(tmp_path):
    pytest.importorskip("matplotlib")
    output = tmp_path / "a" / "b" / "phase.pdf"
    render_phase_diagram([_cell()], output)
    assert output.is_file()


def test_render_writes_png_when_extension_is_png(tmp_path):
    pytest.importorskip("matplotlib")
    output = tmp_path / "phase.png"
    render_phase_diagram([_cell()], output)
    assert output.is_file()
    # PNG header
    with open(output, "rb") as f:
        head = f.read(8)
    assert head == b"\x89PNG\r\n\x1a\n"


def test_render_handles_empty_cells_list(tmp_path):
    pytest.importorskip("matplotlib")
    output = tmp_path / "empty.pdf"
    render_phase_diagram([], output)
    assert output.is_file()


def test_render_with_negative_delta_and_zero_crossing(tmp_path):
    pytest.importorskip("matplotlib")
    output = tmp_path / "neg.pdf"
    cells = [
        _cell("neg1", debate=0.30, voting=0.40, ci_low=-0.20, ci_high=0.05),
        _cell("pos1", debate=0.85, voting=0.75, ci_low=0.02, ci_high=0.18),
    ]
    render_phase_diagram(cells, output)
    assert output.stat().st_size > 1024
