"""Red-team tests for compression-policy."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentdiet.compress import Policy, apply, format_history
from agentdiet.types import Dialogue, Message


def test_empty_dialogue_produces_empty_history():
    d = Dialogue(question_id="q", question="Q", gold_answer="7", messages=[])
    assert apply(d, Policy(mode="b1")) == ""
    assert apply(d, Policy(mode="b2")) == ""
    assert apply(d, Policy(mode="b3", last_k=1)) == ""


def test_b5_random_seed_fallback_from_random_seed_arg():
    """random_seed in Policy takes precedence; if None, random_seed arg to apply() is used."""
    d = Dialogue(
        question_id="q", question="Q", gold_answer="7",
        messages=[Message(agent_id=0, round=1, text="hello world text")],
    )
    cd = {"qid": "q", "claims": [
        {"id": "c0", "text": "x", "agent_id": 0, "round": 1,
         "type": "proposal", "source_message_span": [0, 5]}
    ]}
    a = apply(d, Policy(mode="b5", drop_rate=0.5),
              claims_doc=cd, random_seed=1)
    b = apply(d, Policy(mode="b5", drop_rate=0.5),
              claims_doc=cd, random_seed=1)
    c = apply(d, Policy(mode="b5", drop_rate=0.5, random_seed=999),
              claims_doc=cd, random_seed=1)
    # Same random_seed → same output.
    assert a == b
    # Policy.random_seed wins over apply's random_seed.
    # With 1 claim, seeds 1 and 999 produce at most different draws;
    # the Policy seed path is verified regardless via determinism of c.
    d2 = apply(d, Policy(mode="b5", drop_rate=0.5, random_seed=999),
               claims_doc=cd, random_seed=42)
    assert c == d2


def test_drop_unreferenced_false_is_not_counted_as_filter():
    """drop_unreferenced=False is explicit opt-OUT; policy still needs another filter."""
    with pytest.raises(ValidationError):
        Policy(mode="ours", drop_unreferenced=False)


def test_drop_types_empty_list_is_not_counted_as_filter():
    with pytest.raises(ValidationError):
        Policy(mode="ours", drop_types=[])


def test_b3_zero_rounds_dialogue_returns_empty():
    d = Dialogue(question_id="q", question="Q", gold_answer="7", messages=[])
    assert apply(d, Policy(mode="b3", last_k=5)) == ""


def test_format_history_preserves_order():
    msgs = [
        Message(agent_id=1, round=2, text="second"),
        Message(agent_id=0, round=1, text="first"),
    ]
    out = format_history(msgs)
    # format_history preserves input ordering (does not sort).
    assert out.index("second") < out.index("first")


def test_b1_output_format_has_agent_round_headers():
    d = Dialogue(
        question_id="q", question="Q", gold_answer="7",
        messages=[Message(agent_id=3, round=2, text="body")],
    )
    out = apply(d, Policy(mode="b1"))
    assert "Agent 3 (round 2)" in out
    assert "body" in out


def test_ours_mode_ignores_claim_ids_not_in_signal_scores():
    """If signal_scores lacks a claim id, the claim is neither
    novelty-filtered nor referenced-filtered — it survives the
    signal-based filters (absence of row = no signal to decide on)."""
    d = Dialogue(
        question_id="q", question="Q", gold_answer="7",
        messages=[Message(agent_id=0, round=1, text="important content here")],
    )
    cd = {"qid": "q", "claims": [
        {"id": "c0", "text": "x", "agent_id": 0, "round": 1,
         "type": "evidence", "source_message_span": [0, 9]},
    ]}
    # signal_scores missing c0 entirely.
    out = apply(d, Policy(mode="ours", drop_low_novelty=0.99),
                claims_doc=cd, signal_scores=[])
    # Nothing dropped: c0 has no signal row.
    assert "important" in out


def test_drop_types_and_signal_filters_compose():
    """Union semantics: a claim matching EITHER drop_types OR signal filter is dropped."""
    d = Dialogue(
        question_id="q", question="Q", gold_answer="7",
        messages=[Message(agent_id=0, round=1,
                          text="TYPED_CLAIM and LOW_NOV_CLAIM here")],
    )
    t = d.messages[0].text
    cd = {"claims": [
        {"id": "c_typed", "text": "t", "agent_id": 0, "round": 1,
         "type": "agreement",
         "source_message_span": [t.find("TYPED_CLAIM"),
                                 t.find("TYPED_CLAIM") + len("TYPED_CLAIM")]},
        {"id": "c_low", "text": "l", "agent_id": 0, "round": 1,
         "type": "evidence",
         "source_message_span": [t.find("LOW_NOV_CLAIM"),
                                 t.find("LOW_NOV_CLAIM") + len("LOW_NOV_CLAIM")]},
    ]}
    sig = [
        {"claim_id": "c_typed", "novelty": 0.9, "referenced_later": True, "position": 1},
        {"claim_id": "c_low",   "novelty": 0.05, "referenced_later": True, "position": 1},
    ]
    out = apply(d,
                Policy(mode="ours", drop_types=["agreement"], drop_low_novelty=0.1),
                claims_doc=cd, signal_scores=sig)
    assert "TYPED_CLAIM" not in out    # removed by drop_types
    assert "LOW_NOV_CLAIM" not in out  # removed by drop_low_novelty
