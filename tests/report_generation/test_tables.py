from __future__ import annotations

import pytest

from agentdiet.report import render_table_baselines, render_table_claim_stats


def _eval_results():
    return {"per_method": [
        {"method": "b1", "accuracy": 0.9, "total_tokens": 10000,
         "acc_per_1k": 0.09, "n_evaluated": 100},
        {"method": "b_under_score", "accuracy": 0.5, "total_tokens": 2000,
         "acc_per_1k": 0.25, "n_evaluated": 100},
    ]}


def test_baseline_table_contains_latex_tabular():
    s = render_table_baselines(_eval_results())
    assert r"\begin{tabular}" in s
    assert r"\end{tabular}" in s
    assert r"\toprule" in s and r"\bottomrule" in s


def test_baseline_table_escapes_underscores():
    s = render_table_baselines(_eval_results())
    # b_under_score should be escaped.
    assert r"b\_under\_score" in s
    assert "b_under_score &" not in s


def test_baseline_table_formats_numbers():
    s = render_table_baselines(_eval_results())
    assert "0.900" in s
    assert "10000" in s


def test_baseline_table_empty_results():
    s = render_table_baselines({"per_method": []})
    assert r"\begin{tabular}" in s
    assert r"\end{tabular}" in s


def test_claim_stats_percentages_sum_to_100():
    claims = [
        {"id": "c0", "text": "x", "agent_id": 0, "round": 1,
         "type": "proposal", "source_message_span": [0, 1]},
        {"id": "c1", "text": "x", "agent_id": 0, "round": 1,
         "type": "proposal", "source_message_span": [0, 1]},
        {"id": "c2", "text": "x", "agent_id": 0, "round": 1,
         "type": "evidence", "source_message_span": [0, 1]},
        {"id": "c3", "text": "x", "agent_id": 0, "round": 1,
         "type": "other", "source_message_span": [0, 1]},
    ]
    s = render_table_claim_stats(claims)
    # Extract the percentage column — should sum to ~100.
    import re
    percents = [float(m.group(1)) for m in
                re.finditer(r"& (\d+\.\d+) \\\\", s)]
    assert abs(sum(percents) - 100.0) < 0.5


def test_claim_stats_handles_empty_claims():
    s = render_table_claim_stats([])
    # No rows but structure intact.
    assert r"\begin{tabular}" in s
    assert r"\midrule" in s
