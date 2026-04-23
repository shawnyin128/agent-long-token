from __future__ import annotations

import pytest

from agentdiet.analysis.ablate import reconstruct_masked_history
from agentdiet.types import Dialogue, Message


def _dialogue() -> Dialogue:
    return Dialogue(
        question_id="q", question="Q", gold_answer="7",
        messages=[
            Message(agent_id=0, round=1, text="agent 0 round 1 reasoning chain here"),
            Message(agent_id=1, round=1, text="agent 1 round 1 I agree"),
            Message(agent_id=0, round=2, text="agent 0 round 2 reply"),
            Message(agent_id=1, round=2, text="agent 1 round 2 conclusion"),
        ],
    )


def _claims_doc_with_agreement_in_r1_a1() -> dict:
    # Only agent 1's round-1 message has an agreement claim.
    return {
        "qid": "q",
        "claims": [
            {
                "id": "q_r1_a1_c0", "text": "agrees",
                "agent_id": 1, "round": 1, "type": "agreement",
                "source_message_span": [18, 25],   # "I agree"
            },
        ],
    }


def test_default_granularity_is_message():
    d = _dialogue()
    cd = _claims_doc_with_agreement_in_r1_a1()
    masked = reconstruct_masked_history(
        d, cd, drop_type="agreement", up_to_round=1,
    )
    # Default should blank the whole agent-1 round-1 message, not just the span.
    m_a1_r1 = [m for m in masked if m.agent_id == 1 and m.round == 1][0]
    assert m_a1_r1.text == ""


def test_message_level_blanks_full_text_on_any_matching_claim():
    d = _dialogue()
    cd = _claims_doc_with_agreement_in_r1_a1()
    masked = reconstruct_masked_history(
        d, cd, drop_type="agreement", up_to_round=1, granularity="message",
    )
    m_a1_r1 = [m for m in masked if m.agent_id == 1 and m.round == 1][0]
    assert m_a1_r1.text == ""
    # Messages without a matching claim stay intact.
    m_a0_r1 = [m for m in masked if m.agent_id == 0 and m.round == 1][0]
    assert m_a0_r1.text == "agent 0 round 1 reasoning chain here"


def test_message_level_preserves_indices_and_count():
    d = _dialogue()
    cd = _claims_doc_with_agreement_in_r1_a1()
    masked = reconstruct_masked_history(
        d, cd, drop_type="agreement", up_to_round=1, granularity="message",
    )
    assert len(masked) == len(d.messages)
    keys = [(m.agent_id, m.round) for m in masked]
    assert keys == [(0, 1), (1, 1), (0, 2), (1, 2)]


def test_span_level_mode_unchanged():
    d = _dialogue()
    cd = _claims_doc_with_agreement_in_r1_a1()
    masked = reconstruct_masked_history(
        d, cd, drop_type="agreement", up_to_round=1, granularity="span",
    )
    # Span-level should delete only [18:25], leaving the rest.
    m_a1_r1 = [m for m in masked if m.agent_id == 1 and m.round == 1][0]
    assert m_a1_r1.text != ""
    assert "I agree" not in m_a1_r1.text
    assert "agent 1 round 1 " in m_a1_r1.text


def test_rounds_after_up_to_round_untouched_either_mode():
    d = _dialogue()
    # Claim in round 2 — but up_to_round=1 so it should NOT affect masking.
    cd = {
        "qid": "q",
        "claims": [
            {"id": "q_r2_a0_c0", "text": "x", "agent_id": 0, "round": 2,
             "type": "agreement", "source_message_span": [0, 10]},
        ],
    }
    for gran in ("message", "span"):
        masked = reconstruct_masked_history(
            d, cd, drop_type="agreement", up_to_round=1, granularity=gran,
        )
        # Round-2 message preserved in both modes (out of scope).
        m_r2 = [m for m in masked if m.round == 2][0]
        assert m_r2.text == d.messages[2].text


def test_invalid_granularity_raises():
    d = _dialogue()
    cd = _claims_doc_with_agreement_in_r1_a1()
    with pytest.raises(ValueError):
        reconstruct_masked_history(
            d, cd, drop_type="agreement", up_to_round=1, granularity="letter",
        )


def test_message_level_multiple_matching_claims_in_one_message_still_blanks():
    # Two agreement claims in the same message; message-level blanks it once.
    d = Dialogue(
        question_id="q", question="Q", gold_answer="7",
        messages=[Message(agent_id=0, round=1, text="a much longer message")],
    )
    cd = {
        "qid": "q",
        "claims": [
            {"id": "c0", "text": "x", "agent_id": 0, "round": 1,
             "type": "agreement", "source_message_span": [0, 5]},
            {"id": "c1", "text": "y", "agent_id": 0, "round": 1,
             "type": "agreement", "source_message_span": [10, 17]},
        ],
    }
    masked = reconstruct_masked_history(
        d, cd, drop_type="agreement", up_to_round=1, granularity="message",
    )
    assert masked[0].text == ""


def test_message_level_different_type_leaves_message_intact():
    d = _dialogue()
    cd = _claims_doc_with_agreement_in_r1_a1()
    masked = reconstruct_masked_history(
        d, cd, drop_type="evidence", up_to_round=1, granularity="message",
    )
    # No evidence claims in cd → every message should stay unchanged.
    for orig, m in zip(d.messages, masked):
        assert m.text == orig.text
