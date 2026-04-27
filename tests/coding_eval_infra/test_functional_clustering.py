"""Functional clustering: signature-based grouping + tie-break."""
from __future__ import annotations

import pytest

from agentdiet.eval.base import JudgeResult, TestCase
from agentdiet.eval.clustering import cluster_by_signature


class _FixedJudge:
    """Returns a fixed JudgeResult per sample, indexed by hash."""

    def __init__(self, sig_by_sample: dict[str, tuple[bool, ...]]):
        self._sigs = sig_by_sample

    def run(self, code, tests, timeout_s=10.0):
        sig = self._sigs[code]
        return JudgeResult(
            passed=sig,
            errors=tuple(None if p else "x" for p in sig),
            total=len(sig),
            n_passed=sum(sig),
        )


_TESTS = [TestCase(name="t0", script="assert True"),
          TestCase(name="t1", script="assert True")]


def test_largest_cluster_wins():
    samples = ["A", "B", "C"]
    judge = _FixedJudge({
        "A": (True, True),
        "B": (True, True),
        "C": (True, False),
    })
    res = cluster_by_signature(samples, judge, _TESTS)
    assert res.signature == (True, True)
    assert res.cluster_size == 2
    assert res.representative_index in {0, 1}
    assert res.representative_sample in {"A", "B"}


def test_tie_break_by_smallest_lex_signature():
    """3 unique signatures, all size 1. Smallest lex tuple wins."""
    samples = ["A", "B", "C"]
    judge = _FixedJudge({
        "A": (True, True),    # int (1, 1)
        "B": (False, True),   # int (0, 1) — smallest
        "C": (False, False),  # int (0, 0) — actually smaller than B
    })
    res = cluster_by_signature(samples, judge, _TESTS)
    # (False, False) -> (0,0) is smallest lex; B is (0,1)
    assert res.signature == (False, False)
    assert res.representative_sample == "C"
    assert res.cluster_size == 1


def test_single_sample_degenerate():
    samples = ["only"]
    judge = _FixedJudge({"only": (True,)})
    res = cluster_by_signature(samples, judge, [TestCase(name="t", script="assert True")])
    assert res.cluster_size == 1
    assert res.representative_sample == "only"


def test_empty_samples_raises():
    with pytest.raises(ValueError):
        cluster_by_signature([], _FixedJudge({}), _TESTS)


def test_no_public_tests_one_cluster():
    """When public_tests is empty, all samples have signature () and
    fall in one cluster."""
    samples = ["A", "B", "C"]
    judge = _FixedJudge({"A": (), "B": (), "C": ()})
    res = cluster_by_signature(samples, judge, [])
    assert res.signature == ()
    assert res.cluster_size == 3
    assert res.representative_index == 0


def test_determinism_over_repeated_runs():
    samples = ["A", "B", "C", "D", "E"]
    judge = _FixedJudge({
        "A": (True, False),
        "B": (True, False),
        "C": (False, True),
        "D": (False, True),
        "E": (True, True),
    })
    first = cluster_by_signature(samples, judge, _TESTS)
    for _ in range(50):
        again = cluster_by_signature(samples, judge, _TESTS)
        assert again.representative_index == first.representative_index
        assert again.signature == first.signature
        assert again.cluster_size == first.cluster_size


def test_all_clusters_partitions_input_indices():
    samples = ["A", "B", "C", "D"]
    judge = _FixedJudge({
        "A": (True, True),
        "B": (True, True),
        "C": (False, False),
        "D": (False, True),
    })
    res = cluster_by_signature(samples, judge, _TESTS)
    indices_collected: set[int] = set()
    for sig, indices in res.all_clusters.items():
        for idx in indices:
            assert idx not in indices_collected, "double-counted"
            indices_collected.add(idx)
    assert indices_collected == {0, 1, 2, 3}


def test_three_way_tie_break_proposer_reviewer_integrator_scenario():
    """3 different code samples, each with a different signature, but
    one matches the proposer's signature. Verify deterministic pick."""
    samples = ["proposer_code", "reviewer_code", "integrator_code"]
    judge = _FixedJudge({
        "proposer_code":   (True, True, False),   # (1,1,0)
        "reviewer_code":   (True, False, True),   # (1,0,1) — smaller lex
        "integrator_code": (False, True, True),   # (0,1,1) — smallest lex
    })
    res = cluster_by_signature(
        samples, judge,
        [TestCase(name=f"t{i}", script="assert True") for i in range(3)],
    )
    assert res.representative_sample == "integrator_code"


def test_two_clusters_both_size_two_smallest_lex_wins():
    samples = ["A", "B", "C", "D"]
    judge = _FixedJudge({
        "A": (True, True),    # (1,1)
        "B": (True, True),
        "C": (False, False),  # (0,0) — smallest
        "D": (False, False),
    })
    res = cluster_by_signature(samples, judge, _TESTS)
    assert res.signature == (False, False)
    assert res.cluster_size == 2
    assert res.representative_sample in {"C", "D"}
