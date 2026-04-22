from __future__ import annotations

import pytest

from agentdiet.compress import Policy, apply
from agentdiet.types import Dialogue, Message


def _dialogue_with_typed_claims() -> tuple[Dialogue, dict, list[dict]]:
    d = Dialogue(
        question_id="q", question="Q", gold_answer="7",
        messages=[
            Message(agent_id=0, round=1,
                    text="PROPOSAL_SPAN | EVIDENCE_SPAN | AGREEMENT_SPAN"),
            Message(agent_id=1, round=1, text="QUESTION_SPAN trailing"),
        ],
    )
    t = d.messages[0].text
    claims = [
        {"id": "c_prop", "text": "p", "agent_id": 0, "round": 1,
         "type": "proposal",
         "source_message_span": [t.find("PROPOSAL_SPAN"),
                                 t.find("PROPOSAL_SPAN") + len("PROPOSAL_SPAN")]},
        {"id": "c_evid", "text": "e", "agent_id": 0, "round": 1,
         "type": "evidence",
         "source_message_span": [t.find("EVIDENCE_SPAN"),
                                 t.find("EVIDENCE_SPAN") + len("EVIDENCE_SPAN")]},
        {"id": "c_agr", "text": "a", "agent_id": 0, "round": 1,
         "type": "agreement",
         "source_message_span": [t.find("AGREEMENT_SPAN"),
                                 t.find("AGREEMENT_SPAN") + len("AGREEMENT_SPAN")]},
        {"id": "c_q", "text": "q", "agent_id": 1, "round": 1,
         "type": "question",
         "source_message_span": [0, len("QUESTION_SPAN")]},
    ]
    cd = {"qid": "q", "claims": claims, "per_message_status": [],
          "extraction_failed": False}
    signal_scores = [
        {"claim_id": "c_prop", "flip_coincidence": False, "novelty": 0.9,
         "referenced_later": True, "position": 1},
        {"claim_id": "c_evid", "flip_coincidence": False, "novelty": 0.2,
         "referenced_later": True, "position": 1},
        {"claim_id": "c_agr", "flip_coincidence": False, "novelty": 0.1,
         "referenced_later": False, "position": 1},
        {"claim_id": "c_q", "flip_coincidence": False, "novelty": 0.4,
         "referenced_later": False, "position": 1},
    ]
    return d, cd, signal_scores


def test_ours_drop_types_removes_only_that_type():
    d, cd, _ = _dialogue_with_typed_claims()
    out = apply(d, Policy(mode="ours", drop_types=["agreement"]), claims_doc=cd)
    assert "AGREEMENT_SPAN" not in out
    assert "PROPOSAL_SPAN" in out
    assert "EVIDENCE_SPAN" in out


def test_ours_drop_types_multiple():
    d, cd, _ = _dialogue_with_typed_claims()
    out = apply(d, Policy(mode="ours", drop_types=["agreement", "question"]),
                claims_doc=cd)
    assert "AGREEMENT_SPAN" not in out
    assert "QUESTION_SPAN" not in out
    assert "PROPOSAL_SPAN" in out


def test_ours_drop_low_novelty_removes_below_threshold():
    d, cd, sig = _dialogue_with_typed_claims()
    # c_evid novelty=0.2, c_agr novelty=0.1, c_q novelty=0.4 → drop at 0.3 removes evid + agr.
    out = apply(d, Policy(mode="ours", drop_low_novelty=0.3),
                claims_doc=cd, signal_scores=sig)
    assert "EVIDENCE_SPAN" not in out
    assert "AGREEMENT_SPAN" not in out
    assert "PROPOSAL_SPAN" in out
    assert "QUESTION_SPAN" in out


def test_ours_drop_unreferenced_removes_those_with_False():
    d, cd, sig = _dialogue_with_typed_claims()
    # c_agr and c_q have referenced_later=False
    out = apply(d, Policy(mode="ours", drop_unreferenced=True),
                claims_doc=cd, signal_scores=sig)
    assert "AGREEMENT_SPAN" not in out
    assert "QUESTION_SPAN" not in out
    assert "PROPOSAL_SPAN" in out
    assert "EVIDENCE_SPAN" in out


def test_ours_union_of_filters():
    d, cd, sig = _dialogue_with_typed_claims()
    # drop_types=[question] AND drop_low_novelty=0.15 removes q + agreement (novelty 0.1).
    out = apply(d,
                Policy(mode="ours", drop_types=["question"], drop_low_novelty=0.15),
                claims_doc=cd, signal_scores=sig)
    assert "QUESTION_SPAN" not in out
    assert "AGREEMENT_SPAN" not in out
    # proposal (novelty 0.9) and evidence (novelty 0.2, above 0.15) survive.
    assert "PROPOSAL_SPAN" in out
    assert "EVIDENCE_SPAN" in out


def test_ours_with_no_claims_doc_raises():
    d, _, _ = _dialogue_with_typed_claims()
    with pytest.raises(ValueError):
        apply(d, Policy(mode="ours", drop_types=["agreement"]))


def test_ours_needing_signals_but_none_raises():
    d, cd, _ = _dialogue_with_typed_claims()
    with pytest.raises(ValueError):
        apply(d, Policy(mode="ours", drop_low_novelty=0.5), claims_doc=cd)
    with pytest.raises(ValueError):
        apply(d, Policy(mode="ours", drop_unreferenced=True), claims_doc=cd)


def test_ours_accepts_signal_scores_as_dict_by_id():
    d, cd, sig_list = _dialogue_with_typed_claims()
    sig_dict = {r["claim_id"]: r for r in sig_list}
    out = apply(d, Policy(mode="ours", drop_low_novelty=0.3),
                claims_doc=cd, signal_scores=sig_dict)
    assert "EVIDENCE_SPAN" not in out


def test_ours_drop_types_only_does_not_need_signal_scores():
    d, cd, _ = _dialogue_with_typed_claims()
    # No signal_scores argument — should work.
    apply(d, Policy(mode="ours", drop_types=["agreement"]), claims_doc=cd)


def test_ours_keeps_all_when_no_claim_matches_filter():
    d, cd, _ = _dialogue_with_typed_claims()
    out = apply(d, Policy(mode="ours", drop_types=["correction"]),
                claims_doc=cd)
    # No "correction" claims → every span survives.
    assert "PROPOSAL_SPAN" in out
    assert "EVIDENCE_SPAN" in out
    assert "AGREEMENT_SPAN" in out
    assert "QUESTION_SPAN" in out
