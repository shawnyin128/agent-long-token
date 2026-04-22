from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentdiet.types import Claim, CLAIM_TYPES


def test_claim_accepts_valid_fields():
    c = Claim(
        id="q5_r1_a0_c0",
        text="x = 12",
        agent_id=0,
        round=1,
        type="evidence",
        source_message_span=(0, 6),
    )
    assert c.type == "evidence"
    assert c.source_message_span == (0, 6)


@pytest.mark.parametrize("t", list(CLAIM_TYPES))
def test_claim_accepts_every_enum_value(t):
    Claim(
        id="qx_r1_a0_c0", text="t", agent_id=0, round=1,
        type=t, source_message_span=(0, 1),
    )


def test_claim_rejects_unknown_type():
    with pytest.raises(ValidationError):
        Claim(
            id="qx_r1_a0_c0", text="t", agent_id=0, round=1,
            type="bogus", source_message_span=(0, 1),
        )


def test_claim_rejects_span_start_ge_end():
    with pytest.raises(ValidationError):
        Claim(
            id="qx_r1_a0_c0", text="t", agent_id=0, round=1,
            type="proposal", source_message_span=(5, 5),
        )
    with pytest.raises(ValidationError):
        Claim(
            id="qx_r1_a0_c0", text="t", agent_id=0, round=1,
            type="proposal", source_message_span=(7, 4),
        )


def test_claim_rejects_negative_span():
    with pytest.raises(ValidationError):
        Claim(
            id="qx_r1_a0_c0", text="t", agent_id=0, round=1,
            type="proposal", source_message_span=(-1, 3),
        )


def test_claim_types_has_all_six():
    assert set(CLAIM_TYPES) == {
        "proposal", "evidence", "correction",
        "agreement", "question", "other",
    }
