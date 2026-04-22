from __future__ import annotations

from pathlib import Path

import pytest


plt = pytest.importorskip("matplotlib.pyplot")


from agentdiet.report import (
    render_figure_delta_ranking,
    render_figure_pareto,
    render_figure_signal_correlations,
    render_figure_type_distribution,
)


def _check_png(path: Path):
    assert path.exists()
    assert path.stat().st_size > 200   # trivially non-empty


def test_type_distribution_figure(tmp_path):
    dist = {(0, 1, "proposal"): 4, (0, 1, "evidence"): 2,
            (1, 2, "agreement"): 3}
    out = render_figure_type_distribution(dist, tmp_path / "type.png")
    _check_png(out)


def test_signal_correlations_figure(tmp_path):
    corrs = {"novelty": 0.3, "referenced_later": -0.1, "position": 0.0}
    out = render_figure_signal_correlations(corrs, tmp_path / "sig.png")
    _check_png(out)


def test_delta_ranking_figure(tmp_path):
    ranking = [
        {"type": "evidence", "delta": 0.15, "n_used": 10,
         "acc_with": 1.0, "acc_without": 0.85},
        {"type": "agreement", "delta": -0.05, "n_used": 10,
         "acc_with": 1.0, "acc_without": 1.05},
    ]
    out = render_figure_delta_ranking(ranking, tmp_path / "delta.png")
    _check_png(out)


def test_pareto_figure(tmp_path):
    points = [
        {"method": "b1", "accuracy": 0.9, "total_tokens": 10000},
        {"method": "ours", "accuracy": 0.85, "total_tokens": 4000},
    ]
    out = render_figure_pareto(points, tmp_path / "pareto.png")
    _check_png(out)


def test_figure_closes_matplotlib_state(tmp_path):
    # Rendering repeatedly must not accumulate figures.
    import matplotlib.pyplot as plt
    before = len(plt.get_fignums())
    render_figure_pareto(
        [{"method": "b1", "accuracy": 0.9, "total_tokens": 10000}],
        tmp_path / "p1.png",
    )
    render_figure_pareto(
        [{"method": "ours", "accuracy": 0.85, "total_tokens": 4000}],
        tmp_path / "p2.png",
    )
    after = len(plt.get_fignums())
    assert after == before  # no state leak
