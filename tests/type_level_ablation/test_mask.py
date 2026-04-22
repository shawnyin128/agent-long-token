from __future__ import annotations

import pytest

from agentdiet.analysis.ablate import (
    mask_message_text,
    reconstruct_masked_history,
)
from agentdiet.types import Dialogue, Message


def test_mask_removes_single_span():
    s = "hello world foo bar"
    out = mask_message_text(s, [(6, 11)])  # remove "world"
    assert out == "hello  foo bar"


def test_mask_with_empty_spans_returns_original():
    s = "unchanged"
    assert mask_message_text(s, []) == s


def test_mask_merges_overlapping_spans():
    s = "abcdefghij"
    out = mask_message_text(s, [(2, 6), (4, 8)])  # merge to (2, 8)
    assert out == "abij"


def test_mask_sorts_unordered_spans():
    s = "abcdefghij"
    out = mask_message_text(s, [(7, 9), (1, 3)])
    assert out == "adefgj"


def test_mask_adjacent_spans_collapse():
    s = "abcdefghij"
    out = mask_message_text(s, [(2, 5), (5, 8)])  # touching
    assert out == "abij"


def test_mask_entire_text_returns_empty():
    s = "gone"
    out = mask_message_text(s, [(0, 4)])
    assert out == ""


def test_mask_ignores_spans_with_start_ge_end():
    s = "abcdef"
    out = mask_message_text(s, [(2, 2), (3, 1)])  # both degenerate
    assert out == s


def _dialogue_with_claims() -> tuple[Dialogue, dict]:
    d = Dialogue(
        question_id="q", question="Q", gold_answer="7",
        messages=[
            Message(agent_id=0, round=1, text="First I compute 3+4=7. Then I verify."),
            Message(agent_id=1, round=1, text="Agree with agent 0. #### 7"),
            Message(agent_id=0, round=2, text="Final answer: 7"),
        ],
    )
    claims = [
        {"id": "q_r1_a0_c0", "text": "compute 3+4=7",
         "agent_id": 0, "round": 1, "type": "evidence",
         "source_message_span": [16, 23]},
        {"id": "q_r1_a1_c0", "text": "agreement",
         "agent_id": 1, "round": 1, "type": "agreement",
         "source_message_span": [0, 19]},
        {"id": "q_r2_a0_c0", "text": "final answer",
         "agent_id": 0, "round": 2, "type": "proposal",
         "source_message_span": [0, 15]},
    ]
    cd = {"qid": "q", "claims": claims, "per_message_status": [],
          "extraction_failed": False}
    return d, cd


def test_reconstruct_preserves_message_count():
    d, cd = _dialogue_with_claims()
    new = reconstruct_masked_history(d, cd, drop_type="agreement", up_to_round=1)
    assert len(new) == len(d.messages)


def test_reconstruct_masks_drop_type_only():
    d, cd = _dialogue_with_claims()
    new = reconstruct_masked_history(d, cd, drop_type="agreement", up_to_round=1)
    # Agent 1 round 1 had the only agreement claim — its span [0,19] removed.
    m0_r1 = [m for m in new if m.agent_id == 0 and m.round == 1][0]
    m1_r1 = [m for m in new if m.agent_id == 1 and m.round == 1][0]
    # Agent 0's evidence claim untouched.
    assert "3+4=7" in m0_r1.text
    # Agent 1's message should lose the agreement portion.
    assert "Agree with agent 0" not in m1_r1.text


def test_reconstruct_does_not_touch_rounds_beyond_up_to_round():
    d, cd = _dialogue_with_claims()
    new = reconstruct_masked_history(d, cd, drop_type="proposal", up_to_round=1)
    # Round 2 unchanged because up_to_round=1.
    m_r2 = [m for m in new if m.round == 2][0]
    assert m_r2.text == "Final answer: 7"


def test_reconstruct_emits_empty_string_for_fully_masked_message():
    d, cd = _dialogue_with_claims()
    new = reconstruct_masked_history(d, cd, drop_type="agreement", up_to_round=1)
    m1_r1 = [m for m in new if m.agent_id == 1 and m.round == 1][0]
    assert m1_r1.text == " #### 7"  # only non-span chars remain
    # message count stays the same (not dropped).
    assert len(new) == len(d.messages)


def test_reconstruct_with_nonexistent_drop_type_returns_identical_texts():
    d, cd = _dialogue_with_claims()
    new = reconstruct_masked_history(d, cd, drop_type="question", up_to_round=2)
    for orig, masked in zip(d.messages, new):
        assert orig.text == masked.text
