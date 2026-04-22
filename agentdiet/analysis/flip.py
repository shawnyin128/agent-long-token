"""Flip-point localization.

A "flip" is a round ``r >= 2`` where the agent majority at round r-1
was wrong (does not match the dialogue's ``gold_answer``) and the
majority at round r is right. This module emits one ``FlipEvent`` per
such boundary, with a ``triggering_claim_id`` chosen as the first
proposal/correction-type claim in round r that mentions the post-flip
answer (fallback: first claim of round r by (agent_id, c-index)).

The triggering claim is a pointer for downstream analysis, NOT a
causal claim — the ablation feature does the actual intervention.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

from agentdiet.aggregate import majority_vote
from agentdiet.types import Dialogue, FlipEvent, Message


_PREFERRED_TYPES = ("proposal", "correction")


def round_majority(dialogue: Dialogue, round_idx: int) -> Optional[str]:
    msgs = [m for m in dialogue.messages if m.round == round_idx]
    if not msgs:
        return None
    winner, _ = majority_vote(msgs)
    return winner


def _per_agent_answers(dialogue: Dialogue, round_idx: int) -> dict[int, Optional[str]]:
    msgs = [m for m in dialogue.messages if m.round == round_idx]
    _, per_agent = majority_vote(msgs)
    return per_agent


def _rounds_in_dialogue(dialogue: Dialogue) -> list[int]:
    return sorted({m.round for m in dialogue.messages})


def _claim_sort_key(c: dict[str, Any]) -> tuple[int, str]:
    # Stable order: by agent_id, then by claim id (which encodes c-index).
    return (int(c["agent_id"]), c["id"])


def _pick_triggering_claim(
    round_claims: list[dict[str, Any]], post_flip_answer: str
) -> dict[str, Any]:
    """Prefer proposal/correction claim mentioning post_flip_answer,
    then any proposal/correction, then the first claim in the round."""
    ordered = sorted(round_claims, key=_claim_sort_key)
    for c in ordered:
        if c["type"] in _PREFERRED_TYPES and post_flip_answer in (
            (c.get("text") or "") + " " + _quote_text(c)
        ):
            return c
    for c in ordered:
        if c["type"] in _PREFERRED_TYPES:
            return c
    return ordered[0]


def _quote_text(claim: dict[str, Any]) -> str:
    """Returned for lookup of the verbatim substring if we want it;
    here we don't have the message text so return empty. Used only to
    stabilize the 'mentions post_flip_answer' check."""
    return ""


def locate_flips(
    dialogue: Dialogue, claims_doc: dict[str, Any]
) -> list[FlipEvent]:
    gold = str(dialogue.gold_answer).strip()
    rounds = _rounds_in_dialogue(dialogue)
    if len(rounds) < 2:
        return []

    claims_by_round: dict[int, list[dict[str, Any]]] = {}
    for c in claims_doc.get("claims", []):
        claims_by_round.setdefault(int(c["round"]), []).append(c)

    events: list[FlipEvent] = []
    for i in range(1, len(rounds)):
        r_prev = rounds[i - 1]
        r_cur = rounds[i]
        maj_prev = round_majority(dialogue, r_prev)
        maj_cur = round_majority(dialogue, r_cur)
        if maj_prev == gold:
            continue  # already correct, not a flip-TO-right event
        if maj_cur != gold:
            continue  # still wrong, no flip
        round_claims = claims_by_round.get(r_cur, [])
        assert round_claims, (
            f"invariant: round {r_cur} has no claims but is a flip round "
            f"for qid={dialogue.question_id}; cannot select triggering_claim_id"
        )
        trig = _pick_triggering_claim(round_claims, post_flip_answer=gold)
        events.append(FlipEvent(
            question_id=dialogue.question_id,
            round=r_cur,
            triggering_claim_id=trig["id"],
            pre_flip_answers=_per_agent_answers(dialogue, r_prev),
            post_flip_answers=_per_agent_answers(dialogue, r_cur),
        ))

        # Invariant: triggering_claim_id must exist in claims_doc.
        all_ids = {c["id"] for c in claims_doc.get("claims", [])}
        assert trig["id"] in all_ids, \
            f"invariant: triggering_claim_id {trig['id']} not in claims"

    return events
