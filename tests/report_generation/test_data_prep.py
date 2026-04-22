from __future__ import annotations

import pytest

from agentdiet.report import (
    claim_type_distribution, delta_ranking, pareto_data,
    signal_flip_correlations,
)


def _claim(cid, t, agent=0, round_=1):
    return {"id": cid, "text": "x", "agent_id": agent, "round": round_,
            "type": t, "source_message_span": [0, 1]}


def test_claim_type_distribution_counts_correctly():
    claims = [
        _claim("c0", "proposal", agent=0, round_=1),
        _claim("c1", "evidence", agent=0, round_=1),
        _claim("c2", "proposal", agent=1, round_=1),
        _claim("c3", "agreement", agent=0, round_=2),
    ]
    dist = claim_type_distribution(claims)
    assert dist[(0, 1, "proposal")] == 1
    assert dist[(1, 1, "proposal")] == 1
    assert dist[(0, 1, "evidence")] == 1
    assert dist[(0, 2, "agreement")] == 1


def test_claim_type_distribution_empty_input():
    assert claim_type_distribution([]) == {}


def test_signal_flip_correlations_returns_per_signal_floats():
    rows = [
        {"claim_id": f"c{i}", "flip_coincidence": (i < 5),
         "novelty": float(i) / 10.0,
         "referenced_later": (i % 2 == 0),
         "position": i} for i in range(10)
    ]
    corrs = signal_flip_correlations(rows)
    assert set(corrs.keys()) == {"novelty", "referenced_later", "position"}
    for v in corrs.values():
        assert -1.0 <= v <= 1.0


def test_signal_flip_correlations_perfect_positive():
    rows = [
        {"claim_id": "c0", "flip_coincidence": True, "novelty": 1.0,
         "referenced_later": True, "position": 1},
        {"claim_id": "c1", "flip_coincidence": False, "novelty": 0.0,
         "referenced_later": False, "position": 0},
    ]
    corrs = signal_flip_correlations(rows)
    # With perfectly aligned novelty, correlation == 1.
    assert corrs["novelty"] == pytest.approx(1.0)
    assert corrs["referenced_later"] == pytest.approx(1.0)
    assert corrs["position"] == pytest.approx(1.0)


def test_signal_flip_correlations_handles_zero_variance_safely():
    # All rows have the same flip_coincidence → correlation is undefined;
    # helper should return 0.0 for that signal rather than NaN.
    rows = [
        {"claim_id": f"c{i}", "flip_coincidence": True,
         "novelty": float(i), "referenced_later": True, "position": i}
        for i in range(5)
    ]
    corrs = signal_flip_correlations(rows)
    for v in corrs.values():
        # No variance in flip_coincidence → safe fallback to 0.
        assert v == 0.0


def test_delta_ranking_sorts_by_abs_delta_desc():
    summary = {"per_type": [
        {"type": "proposal", "delta": 0.02, "n_used": 10, "acc_with": 1, "acc_without": 1},
        {"type": "evidence", "delta": -0.15, "n_used": 10, "acc_with": 1, "acc_without": 1},
        {"type": "other",   "delta": 0.10, "n_used": 10, "acc_with": 1, "acc_without": 1},
    ]}
    ranked = delta_ranking(summary)
    assert [r["type"] for r in ranked] == ["evidence", "other", "proposal"]


def test_delta_ranking_tie_break_by_type_name():
    summary = {"per_type": [
        {"type": "proposal", "delta": 0.10, "n_used": 1, "acc_with": 1, "acc_without": 1},
        {"type": "evidence", "delta": -0.10, "n_used": 1, "acc_with": 1, "acc_without": 1},
    ]}
    ranked = delta_ranking(summary)
    # Both |delta|=0.10 → alphabetical tie-break.
    assert [r["type"] for r in ranked] == ["evidence", "proposal"]


def test_pareto_data_extracts_per_method_points():
    results = {"per_method": [
        {"method": "b1", "accuracy": 0.9, "total_tokens": 10000,
         "acc_per_1k": 0.09, "n_evaluated": 100},
        {"method": "ours", "accuracy": 0.85, "total_tokens": 4000,
         "acc_per_1k": 0.2125, "n_evaluated": 100},
    ]}
    pts = pareto_data(results)
    assert len(pts) == 2
    assert {"method", "accuracy", "total_tokens"} <= set(pts[0].keys())
    # Values preserved.
    methods = {p["method"]: p for p in pts}
    assert methods["b1"]["accuracy"] == 0.9
    assert methods["ours"]["total_tokens"] == 4000


def test_pareto_data_empty_input():
    assert pareto_data({"per_method": []}) == []
