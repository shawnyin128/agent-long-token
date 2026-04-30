"""paired_bootstrap_delta math + edge cases."""
from __future__ import annotations

import pytest

from agentdiet.analysis_phase.bootstrap import (
    BootstrapResult,
    paired_bootstrap_delta,
)


def test_identical_inputs_give_zero_delta_and_ci_contains_zero():
    # Same correctness vector for both — observed delta is 0 and CI
    # straddles 0.
    correct = [True, True, False, True, False]
    boot = paired_bootstrap_delta(correct, correct, n_resamples=2000)
    assert boot.delta == 0.0
    assert boot.ci_low <= 0.0 <= boot.ci_high


def test_a_always_correct_b_always_wrong_gives_delta_one():
    a = [True] * 20
    b = [False] * 20
    boot = paired_bootstrap_delta(a, b, n_resamples=2000)
    assert boot.delta == 1.0
    # All resamples yield acc_a=1, acc_b=0 -> CI is degenerate at 1.0
    assert boot.ci_low == pytest.approx(1.0)
    assert boot.ci_high == pytest.approx(1.0)


def test_b_always_correct_a_always_wrong_gives_delta_minus_one():
    a = [False] * 10
    b = [True] * 10
    boot = paired_bootstrap_delta(a, b, n_resamples=2000)
    assert boot.delta == -1.0
    assert boot.ci_low == pytest.approx(-1.0)


def test_deterministic_under_fixed_seed():
    a = [True, False, True, False, True, False, True, False]
    b = [True, True, False, False, True, True, False, False]
    boot1 = paired_bootstrap_delta(a, b, n_resamples=500, seed=42)
    boot2 = paired_bootstrap_delta(a, b, n_resamples=500, seed=42)
    assert boot1 == boot2


def test_different_seeds_give_different_ci_endpoints():
    a = [True, False] * 30
    b = [True, True, False] * 20
    boot1 = paired_bootstrap_delta(a, b, n_resamples=500, seed=42)
    boot2 = paired_bootstrap_delta(a, b, n_resamples=500, seed=99)
    # Observed delta is the same; CI bounds should differ at finite n
    assert boot1.delta == boot2.delta
    assert (boot1.ci_low, boot1.ci_high) != (boot2.ci_low, boot2.ci_high)


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        paired_bootstrap_delta([True, False], [True], n_resamples=10)


def test_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        paired_bootstrap_delta([], [], n_resamples=10)


def test_invalid_ci_raises():
    with pytest.raises(ValueError, match="ci must be"):
        paired_bootstrap_delta([True], [True], ci=1.5)


def test_invalid_n_resamples_raises():
    with pytest.raises(ValueError, match="n_resamples"):
        paired_bootstrap_delta([True], [True], n_resamples=0)


def test_result_is_a_frozen_dataclass():
    boot = paired_bootstrap_delta([True], [False], n_resamples=10)
    assert isinstance(boot, BootstrapResult)
    with pytest.raises(Exception):  # frozen
        boot.delta = 0.5  # type: ignore[misc]
