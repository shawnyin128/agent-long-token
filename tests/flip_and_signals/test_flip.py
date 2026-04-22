from __future__ import annotations

import pytest

from agentdiet.analysis.flip import round_majority, locate_flips
from agentdiet.types import Dialogue, Message


def _dialogue(per_round_per_agent_answers: list[list[str]], gold: str = "7") -> Dialogue:
    """per_round_per_agent_answers[r][a] = stringified numeric answer for round r+1, agent a."""
    messages = []
    for r_idx, round_ans in enumerate(per_round_per_agent_answers, 1):
        for a_idx, ans in enumerate(round_ans):
            messages.append(Message(
                agent_id=a_idx, round=r_idx,
                text=f"My answer: {ans}. #### {ans}",
            ))
    return Dialogue(
        question_id="qt", question="Q", gold_answer=gold,
        messages=messages, final_answer=None,
    )


def _claims_doc(dialogue: Dialogue, per_msg_claim_types=None) -> dict:
    claims = []
    idx = 0
    for m in dialogue.messages:
        t = "proposal"
        if per_msg_claim_types:
            t = per_msg_claim_types.get((m.round, m.agent_id), "proposal")
        claims.append({
            "id": f"qt_r{m.round}_a{m.agent_id}_c0",
            "text": f"answer {m.text[-4:].strip()}",
            "agent_id": m.agent_id, "round": m.round,
            "type": t, "source_message_span": [0, 3],
        })
        idx += 1
    return {
        "qid": dialogue.question_id, "claims": claims,
        "per_message_status": [], "extraction_failed": False,
    }


def test_round_majority_returns_none_for_tie():
    d = _dialogue([["7", "4", None]])
    # Round 1: {7:1, 4:1, None:1} → tie → None
    assert round_majority(d, round_idx=1) is None


def test_round_majority_handles_clear_winner():
    d = _dialogue([["7", "7", "4"]])
    assert round_majority(d, round_idx=1) == "7"


def test_no_flip_when_all_rounds_correct():
    d = _dialogue([["7", "7", "7"], ["7", "7", "7"]], gold="7")
    cd = _claims_doc(d)
    events = locate_flips(d, cd)
    assert events == []


def test_no_flip_when_all_rounds_wrong():
    d = _dialogue([["3", "3", "3"], ["3", "3", "3"]], gold="7")
    cd = _claims_doc(d)
    events = locate_flips(d, cd)
    assert events == []


def test_flip_detected_wrong_to_right():
    d = _dialogue([["3", "3", "4"], ["7", "7", "7"]], gold="7")
    cd = _claims_doc(d)
    events = locate_flips(d, cd)
    assert len(events) == 1
    fe = events[0]
    assert fe.round == 2
    assert fe.question_id == "qt"
    assert fe.triggering_claim_id.startswith("qt_r2_")
    assert fe.pre_flip_answers == {0: "3", 1: "3", 2: "4"}
    assert fe.post_flip_answers == {0: "7", 1: "7", 2: "7"}


def test_multiple_flips_detected():
    # wrong -> right -> wrong -> right
    d = _dialogue(
        [["3", "3", "4"], ["7", "7", "7"], ["3", "3", "3"], ["7", "7", "7"]],
        gold="7",
    )
    cd = _claims_doc(d)
    events = locate_flips(d, cd)
    # Flip happens at r=2 (wrong→right). Flip at r=4 would be wrong→right again.
    assert len(events) == 2
    assert [e.round for e in events] == [2, 4]


def test_no_flip_right_to_wrong():
    d = _dialogue([["7", "7", "7"], ["3", "3", "3"]], gold="7")
    cd = _claims_doc(d)
    events = locate_flips(d, cd)
    assert events == []


def test_triggering_claim_references_real_claim():
    d = _dialogue([["3", "3", "3"], ["7", "7", "7"]], gold="7")
    cd = _claims_doc(d)
    events = locate_flips(d, cd)
    claim_ids = {c["id"] for c in cd["claims"]}
    assert all(e.triggering_claim_id in claim_ids for e in events)


def test_triggering_claim_prefers_proposal_or_correction():
    """If flip round has a 'correction' claim, prefer it over 'other'."""
    d = _dialogue([["3", "3", "3"], ["7", "7", "7"]], gold="7")
    cd = _claims_doc(d, per_msg_claim_types={
        (2, 0): "other",
        (2, 1): "correction",
        (2, 2): "other",
    })
    events = locate_flips(d, cd)
    assert len(events) == 1
    # The correction claim is a1's — prefer it.
    assert "a1" in events[0].triggering_claim_id


def test_triggering_claim_falls_back_to_first_when_no_proposal_correction():
    d = _dialogue([["3", "3", "3"], ["7", "7", "7"]], gold="7")
    cd = _claims_doc(d, per_msg_claim_types={
        (2, 0): "other", (2, 1): "agreement", (2, 2): "other",
    })
    events = locate_flips(d, cd)
    assert len(events) == 1
    # First claim in round-2 sort order: agent 0, c0.
    assert events[0].triggering_claim_id == "qt_r2_a0_c0"


def test_empty_claims_doc_raises_invariant():
    d = _dialogue([["3", "3", "3"], ["7", "7", "7"]], gold="7")
    # claims list empty → no triggering_claim_id possible → AssertionError
    cd = {"qid": "qt", "claims": [], "per_message_status": [],
          "extraction_failed": False}
    with pytest.raises(AssertionError):
        locate_flips(d, cd)
