from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentdiet.types import FlipEvent


def test_flip_event_happy_path():
    fe = FlipEvent(
        question_id="q1",
        round=2,
        triggering_claim_id="q1_r2_a0_c0",
        pre_flip_answers={0: "5", 1: "4", 2: None},
        post_flip_answers={0: "5", 1: "5", 2: "5"},
    )
    assert fe.round == 2
    assert fe.triggering_claim_id.startswith("q1_")


def test_flip_event_rejects_empty_triggering_claim_id():
    with pytest.raises(ValidationError):
        FlipEvent(
            question_id="q1", round=2, triggering_claim_id="",
            pre_flip_answers={}, post_flip_answers={},
        )


def test_flip_event_rejects_round_lt_2():
    # round 1 has no prior round to flip from
    with pytest.raises(ValidationError):
        FlipEvent(
            question_id="q1", round=1, triggering_claim_id="qx",
            pre_flip_answers={}, post_flip_answers={},
        )
