from __future__ import annotations

import pytest

from agentdiet.compress import Policy, apply, format_history
from agentdiet.types import Dialogue, Message


def _dialogue() -> Dialogue:
    return Dialogue(
        question_id="q", question="Q", gold_answer="7",
        messages=[
            Message(agent_id=0, round=1, text="turn 0-1"),
            Message(agent_id=1, round=1, text="turn 1-1"),
            Message(agent_id=0, round=2, text="turn 0-2"),
            Message(agent_id=1, round=2, text="turn 1-2"),
            Message(agent_id=0, round=3, text="turn 0-3"),
            Message(agent_id=1, round=3, text="turn 1-3"),
        ],
    )


def _claims(dialogue: Dialogue) -> dict:
    claims = []
    for i, m in enumerate(dialogue.messages):
        claims.append({
            "id": f"c{i}", "text": m.text, "agent_id": m.agent_id,
            "round": m.round, "type": "proposal",
            "source_message_span": [0, len(m.text)],
        })
    return {"qid": dialogue.question_id, "claims": claims,
            "per_message_status": [], "extraction_failed": False}


def test_b1_returns_full_history():
    d = _dialogue()
    out = apply(d, Policy(mode="b1"))
    expected = format_history(list(d.messages))
    assert out == expected


def test_b2_returns_only_round1_agent0():
    d = _dialogue()
    out = apply(d, Policy(mode="b2"))
    assert "turn 0-1" in out
    assert "turn 1-1" not in out
    assert "turn 0-2" not in out
    assert "turn 0-3" not in out


def test_b3_last_k_1_is_final_round():
    d = _dialogue()
    out = apply(d, Policy(mode="b3", last_k=1))
    assert "turn 0-3" in out and "turn 1-3" in out
    assert "turn 0-1" not in out
    assert "turn 0-2" not in out


def test_b3_last_k_2_is_last_two_rounds():
    d = _dialogue()
    out = apply(d, Policy(mode="b3", last_k=2))
    assert "turn 0-2" in out and "turn 1-2" in out
    assert "turn 0-3" in out and "turn 1-3" in out
    assert "turn 0-1" not in out


def test_b3_last_k_larger_than_rounds_keeps_all():
    d = _dialogue()
    out = apply(d, Policy(mode="b3", last_k=99))
    for m in d.messages:
        assert m.text in out


def test_b5_drop_rate_0_equivalent_to_full_under_claims():
    d = _dialogue()
    cd = _claims(d)
    out = apply(d, Policy(mode="b5", drop_rate=0.0, random_seed=42),
                claims_doc=cd)
    # All messages present.
    for m in d.messages:
        assert m.text in out


def test_b5_drop_rate_1_removes_every_claim_span():
    d = _dialogue()
    cd = _claims(d)
    out = apply(d, Policy(mode="b5", drop_rate=1.0, random_seed=42),
                claims_doc=cd)
    # Each claim spanned the full message text → all text masked.
    for m in d.messages:
        assert m.text not in out


def test_b5_seeded_deterministic():
    d = _dialogue()
    cd = _claims(d)
    p = Policy(mode="b5", drop_rate=0.5, random_seed=42)
    a = apply(d, p, claims_doc=cd)
    b = apply(d, p, claims_doc=cd)
    assert a == b


def test_b5_different_seeds_can_produce_different_outputs():
    d = _dialogue()
    cd = _claims(d)
    a = apply(d, Policy(mode="b5", drop_rate=0.5, random_seed=1), claims_doc=cd)
    b = apply(d, Policy(mode="b5", drop_rate=0.5, random_seed=999), claims_doc=cd)
    # Not guaranteed to differ, but with 6 claims and drop_rate=0.5 two very
    # different seeds almost certainly produce different mask sets.
    assert a != b


def test_b5_requires_claims_doc():
    d = _dialogue()
    with pytest.raises(ValueError):
        apply(d, Policy(mode="b5", drop_rate=0.3))


def test_b1_b2_b3_do_not_require_claims_doc():
    d = _dialogue()
    apply(d, Policy(mode="b1"))
    apply(d, Policy(mode="b2"))
    apply(d, Policy(mode="b3", last_k=1))
