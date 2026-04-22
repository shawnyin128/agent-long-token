from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentdiet.compress import Policy, load_policy


def test_b1_minimal():
    p = Policy(mode="b1")
    assert p.mode == "b1"


def test_b2_minimal():
    p = Policy(mode="b2")
    assert p.mode == "b2"


def test_b3_requires_last_k_ge_1():
    p = Policy(mode="b3", last_k=1)
    assert p.last_k == 1
    with pytest.raises(ValidationError):
        Policy(mode="b3", last_k=0)


def test_b3_default_last_k_is_1():
    # last_k omitted → defaults applied via pre-validator or assigned at use site.
    p = Policy(mode="b3")
    assert p.last_k == 1


def test_b5_drop_rate_in_unit_interval():
    Policy(mode="b5", drop_rate=0.5)
    with pytest.raises(ValidationError):
        Policy(mode="b5", drop_rate=-0.01)
    with pytest.raises(ValidationError):
        Policy(mode="b5", drop_rate=1.01)


def test_b5_default_drop_rate():
    p = Policy(mode="b5")
    assert p.drop_rate == 0.3


def test_ours_requires_at_least_one_filter():
    with pytest.raises(ValidationError):
        Policy(mode="ours")


def test_ours_with_drop_types():
    p = Policy(mode="ours", drop_types=["agreement"])
    assert p.drop_types == ["agreement"]


def test_ours_with_novelty_threshold():
    p = Policy(mode="ours", drop_low_novelty=0.3)
    assert p.drop_low_novelty == 0.3


def test_ours_with_drop_unreferenced():
    p = Policy(mode="ours", drop_unreferenced=True)
    assert p.drop_unreferenced is True


def test_invalid_mode_rejected():
    with pytest.raises(ValidationError):
        Policy(mode="b4")


def test_load_policy_round_trip(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"mode": "ours", "drop_types": ["other"]}))
    p = load_policy(path)
    assert p.mode == "ours"
    assert p.drop_types == ["other"]


def test_load_policy_raises_on_invalid(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"mode": "ours"}')   # no filter specified
    with pytest.raises(ValidationError):
        load_policy(path)


def test_drop_types_only_accepts_valid_claim_types():
    with pytest.raises(ValidationError):
        Policy(mode="ours", drop_types=["not_a_type"])


def test_drop_low_novelty_in_unit_interval():
    Policy(mode="ours", drop_low_novelty=0.5)
    with pytest.raises(ValidationError):
        Policy(mode="ours", drop_low_novelty=1.5)
