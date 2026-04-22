"""Compressed-history policy engine.

Single entry point ``compress.apply(dialogue, policy, ...) -> str``
supports 5 policy modes (spec §4.2):

  * ``b1`` — full history (upper bound)
  * ``b2`` — round 1, agent 0 only (lower bound / single-agent)
  * ``b3`` — last ``last_k`` rounds (sliding-window)
  * ``b5`` — uniform random claim drop at ``drop_rate`` (seeded)
  * ``ours`` — data-driven rule: any union of
    - ``drop_types`` (by claim type)
    - ``drop_low_novelty`` (threshold, needs signal_scores)
    - ``drop_unreferenced`` (bool, needs signal_scores)

All modes return a canonical history string formatted the same way
so evaluation-sweep can compute token counts apples-to-apples.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from agentdiet.analysis.ablate import mask_message_text
from agentdiet.types import CLAIM_TYPES, ClaimType, Dialogue, Message


PolicyMode = Literal["b1", "b2", "b3", "b5", "ours"]


class Policy(BaseModel):
    mode: PolicyMode
    last_k: Optional[int] = Field(default=None, ge=1)
    drop_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    random_seed: Optional[int] = None
    drop_types: Optional[list[ClaimType]] = None
    drop_low_novelty: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    drop_unreferenced: Optional[bool] = None

    @model_validator(mode="after")
    def _apply_defaults_and_validate(self) -> "Policy":
        if self.mode == "b3" and self.last_k is None:
            object.__setattr__(self, "last_k", 1)
        if self.mode == "b5" and self.drop_rate is None:
            object.__setattr__(self, "drop_rate", 0.3)
        if self.mode == "ours":
            has_filter = (
                (self.drop_types is not None and len(self.drop_types) > 0)
                or self.drop_low_novelty is not None
                or self.drop_unreferenced is True
            )
            if not has_filter:
                raise ValueError(
                    "'ours' mode requires at least one of drop_types, "
                    "drop_low_novelty, drop_unreferenced"
                )
        return self


def load_policy(path: Path) -> Policy:
    return Policy.model_validate_json(Path(path).read_text(encoding="utf-8"))


def format_history(messages: list[Message]) -> str:
    blocks = [
        f"--- Agent {m.agent_id} (round {m.round}) ---\n{m.text}"
        for m in messages
    ]
    return "\n\n".join(blocks)


def _mask_claims_by_predicate(
    dialogue: Dialogue, claims: list[dict], predicate: Callable[[dict], bool],
) -> list[Message]:
    spans_by_key: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for c in claims:
        if not predicate(c):
            continue
        key = (int(c["agent_id"]), int(c["round"]))
        a, b = c["source_message_span"]
        spans_by_key.setdefault(key, []).append((int(a), int(b)))

    new_messages: list[Message] = []
    for m in dialogue.messages:
        key = (m.agent_id, m.round)
        if key in spans_by_key:
            masked = mask_message_text(m.text, spans_by_key[key])
            new_messages.append(Message(
                agent_id=m.agent_id, round=m.round, text=masked,
            ))
        else:
            new_messages.append(m)
    return new_messages


def _apply_b1(dialogue: Dialogue) -> str:
    return format_history(list(dialogue.messages))


def _apply_b2(dialogue: Dialogue) -> str:
    kept = [m for m in dialogue.messages if m.round == 1 and m.agent_id == 0]
    return format_history(kept)


def _apply_b3(dialogue: Dialogue, last_k: int) -> str:
    rounds = sorted({m.round for m in dialogue.messages})
    keep = set(rounds[-last_k:]) if last_k <= len(rounds) else set(rounds)
    kept = [m for m in dialogue.messages if m.round in keep]
    return format_history(kept)


def _apply_b5(
    dialogue: Dialogue, drop_rate: float, claims: list[dict], seed: int,
) -> str:
    if not claims:
        return format_history(list(dialogue.messages))
    rng = random.Random(seed)
    dropped_ids: set[str] = set()
    for c in claims:
        if rng.random() < drop_rate:
            dropped_ids.add(c["id"])
    masked = _mask_claims_by_predicate(
        dialogue, claims, predicate=lambda c: c["id"] in dropped_ids,
    )
    return format_history(masked)


def _signal_scores_by_id(signal_scores: Any) -> dict[str, dict]:
    if signal_scores is None:
        return {}
    if isinstance(signal_scores, dict):
        return signal_scores
    return {row["claim_id"]: row for row in signal_scores}


def _apply_ours(
    dialogue: Dialogue, policy: Policy, claims: list[dict],
    signal_scores: Any = None,
) -> str:
    sig_by_id = _signal_scores_by_id(signal_scores)
    needs_signals = (
        policy.drop_low_novelty is not None or policy.drop_unreferenced is True
    )
    if needs_signals and signal_scores is None:
        raise ValueError("'ours' mode with novelty/unreferenced filter requires signal_scores")

    drop_type_set = set(policy.drop_types or [])
    novelty_thresh = policy.drop_low_novelty
    drop_unref = bool(policy.drop_unreferenced)

    def should_drop(c: dict) -> bool:
        if drop_type_set and c["type"] in drop_type_set:
            return True
        row = sig_by_id.get(c["id"])
        if row is not None:
            if novelty_thresh is not None and float(row.get("novelty", 1.0)) < novelty_thresh:
                return True
            if drop_unref and not bool(row.get("referenced_later", True)):
                return True
        return False

    masked = _mask_claims_by_predicate(dialogue, claims, predicate=should_drop)
    return format_history(masked)


def apply(
    dialogue: Dialogue, policy: Policy, *,
    claims_doc: Optional[dict] = None,
    signal_scores: Any = None,
    random_seed: Optional[int] = None,
) -> str:
    if policy.mode == "b1":
        return _apply_b1(dialogue)
    if policy.mode == "b2":
        return _apply_b2(dialogue)
    if policy.mode == "b3":
        assert policy.last_k is not None
        return _apply_b3(dialogue, policy.last_k)
    if policy.mode == "b5":
        if claims_doc is None:
            raise ValueError("'b5' mode requires claims_doc")
        assert policy.drop_rate is not None
        seed = policy.random_seed if policy.random_seed is not None else (
            random_seed if random_seed is not None else 42
        )
        return _apply_b5(dialogue, policy.drop_rate,
                          claims_doc.get("claims", []), seed)
    if policy.mode == "ours":
        if claims_doc is None:
            raise ValueError("'ours' mode requires claims_doc")
        return _apply_ours(dialogue, policy, claims_doc.get("claims", []),
                            signal_scores=signal_scores)
    raise ValueError(f"unknown policy mode: {policy.mode}")
