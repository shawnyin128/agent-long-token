from __future__ import annotations

import re
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _makefile() -> str:
    return (PROJECT_ROOT / "Makefile").read_text()


def _readme() -> str:
    return (PROJECT_ROOT / "README.md").read_text()


@pytest.mark.parametrize("target", ["report", "report-clean"])
def test_makefile_declares_target(target):
    mf = _makefile()
    phony = re.search(r"\.PHONY:\s*([\s\S]*?)\n\n", mf)
    assert phony and target in phony.group(1)
    assert re.search(rf"^{re.escape(target)}:\s*\n(\t.+\n)+", mf, re.MULTILINE)


def test_readme_mentions_report_generation():
    r = _readme()
    assert re.search(r"#{1,3}\s*Report", r, re.IGNORECASE)
    assert "make report" in r
    assert "pandoc" in r.lower()


def test_template_contains_all_imrad_sections():
    md = (PROJECT_ROOT / "docs" / "reports" / "final-report.template.md").read_text()
    for heading in ("Abstract", "Introduction", "Methods", "Results",
                    "Discussion", "Limitations"):
        assert heading in md, f"missing IMRAD section {heading}"


def test_template_references_expected_figures_and_tables():
    md = (PROJECT_ROOT / "docs" / "reports" / "final-report.template.md").read_text()
    for name in ("claim_type_distribution.png", "signal_correlations.png",
                 "delta_ranking.png", "pareto.png", "baselines.tex",
                 "claim_stats.tex"):
        assert name in md, f"template missing reference to {name}"
