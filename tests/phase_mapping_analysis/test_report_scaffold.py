"""LaTeX report scaffold + Makefile target structural checks."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_TEX = REPO_ROOT / "docs" / "reports" / "final-report.tex"
MAKEFILE = REPO_ROOT / "Makefile"


def _tex() -> str:
    return REPORT_TEX.read_text(encoding="utf-8")


def _make() -> str:
    return MAKEFILE.read_text(encoding="utf-8")


def test_report_tex_exists():
    assert REPORT_TEX.is_file()


def test_report_tex_documentclass():
    text = _tex()
    assert "\\documentclass" in text


def test_report_tex_inputs_all_expected_table_files():
    text = _tex()
    expected = [
        "tables/phase_summary.tex",
        "tables/aime_per_year.tex",
        "tables/cross_model.tex",
        "tables/thinking_o1.tex",
        "tables/thinking_o2.tex",
    ]
    for path in expected:
        assert path in text, f"final-report.tex missing input for {path}"


def test_report_tex_includegraphics_phase_diagram():
    text = _tex()
    assert "figs/phase_diagram.pdf" in text
    assert "\\includegraphics" in text


def test_report_tex_uses_iffileexists_guards():
    """Tables / figures use \\IfFileExists so the .tex compiles
    even when analyze_phase has not yet run."""
    text = _tex()
    # At least 5 IfFileExists guards (one per table + figure)
    assert text.count("\\IfFileExists") >= 5


def test_report_tex_cites_du_and_wang():
    text = _tex()
    assert "du2023" in text.lower() or "Du, Y" in text
    assert "wang2023" in text.lower() or "Wang, X" in text


def test_report_tex_has_limitations_section():
    text = _tex()
    assert "Limitations" in text
    # spec §7 bullets — at least 4 named limitations
    for keyword in ("AIME contamination", "Single seed",
                    "Diversity-source", "Cross-model token"):
        assert keyword in text, f"limitations section missing '{keyword}'"


def test_report_tex_has_methodological_appendix():
    text = _tex()
    assert "Methodological Appendix" in text


def test_makefile_phase_report_target_exists():
    text = _make()
    assert "phase-report:" in text


def test_makefile_analyze_phase_target_invokes_cli_analyze_phase():
    text = _make()
    assert "analyze-phase:" in text
    assert "agentdiet.cli.analyze_phase" in text


def test_makefile_phase_report_chains_pdflatex_twice():
    text = _make()
    # Two pdflatex invocations for cross-references
    assert text.count("pdflatex -interaction=nonstopmode") == 2


def test_makefile_phase_report_depends_on_analyze_phase():
    """phase-report should run analyze-phase first."""
    text = _make()
    # Look for "phase-report: analyze-phase" prerequisite
    assert "phase-report: analyze-phase" in text


def test_makefile_phase_report_handles_missing_pdflatex():
    text = _make()
    assert "pdflatex" in text
    assert "command -v pdflatex" in text or "if command" in text
