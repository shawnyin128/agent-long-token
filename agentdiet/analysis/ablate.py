"""Type-level ablation (spec §5 + §7 Gate 2).

For each of the 6 claim types t, remove all claims of that type from
debate history and replay the final round; compute Δ_t = acc(with) −
acc(without) on the ``single_wrong ∧ debate_right`` subset.

A hard ``MAX_NEW_LLM_CALLS=500`` cap (spec §6.1) is enforced against
``LLMClient.call_count`` — cache hits are free.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Optional

from agentdiet.agents import make_default_agents
from agentdiet.aggregate import majority_vote
from agentdiet.config import Config
from agentdiet.dataset import parse_answer
from agentdiet.debate import INITIAL_USER_TEMPLATE, LATER_ROUND_TEMPLATE
from agentdiet.llm_client import LLMClient
from agentdiet.types import CLAIM_TYPES, Dialogue, Message


log = logging.getLogger(__name__)


MAX_NEW_LLM_CALLS_DEFAULT = 500


def load_dialogue_and_claims(cfg: Config, qid: str) -> tuple[Dialogue, dict]:
    dpath = cfg.dialogues_dir / f"{qid}.json"
    cpath = cfg.claims_dir / f"{qid}.json"
    if not dpath.exists():
        raise FileNotFoundError(f"dialogue artifact missing: {dpath}")
    if not cpath.exists():
        raise FileNotFoundError(f"claims artifact missing: {cpath}")
    d = Dialogue.model_validate_json(dpath.read_text(encoding="utf-8"))
    cd = json.loads(cpath.read_text(encoding="utf-8"))
    return d, cd


def _pilot_single_path(cfg: Config, qid: str) -> Path:
    return cfg.artifacts_dir / "pilot" / "single" / cfg.model_slug / f"{qid}.json"


def _load_single_doc(cfg: Config, qid: str) -> Optional[dict]:
    path = _pilot_single_path(cfg, qid)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def is_single_wrong_debate_right(
    single_doc: dict, debate_dialogue: Dialogue, gold: str
) -> bool:
    s_final = single_doc.get("final_answer")
    d_final = debate_dialogue.final_answer
    if s_final is None or d_final is None:
        return False
    return str(s_final).strip() != str(gold).strip() and \
        str(d_final).strip() == str(gold).strip()


def select_subset(cfg: Config, target_size: int) -> list[str]:
    dialogue_qids = {p.stem for p in cfg.dialogues_dir.glob("*.json")}
    claim_qids = {p.stem for p in cfg.claims_dir.glob("*.json")}
    both = sorted(dialogue_qids & claim_qids)

    eligible: list[str] = []
    for qid in both:
        single = _load_single_doc(cfg, qid)
        if single is None:
            continue
        try:
            dialogue = Dialogue.model_validate_json(
                (cfg.dialogues_dir / f"{qid}.json").read_text(encoding="utf-8")
            )
        except Exception:  # noqa: BLE001
            continue
        if is_single_wrong_debate_right(single, dialogue, dialogue.gold_answer):
            eligible.append(qid)

    if target_size >= len(eligible):
        return eligible
    rng = random.Random(cfg.seed)
    return sorted(rng.sample(eligible, target_size))


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    valid = [(a, b) for a, b in spans if a < b]
    if not valid:
        return []
    valid.sort()
    merged: list[tuple[int, int]] = [valid[0]]
    for a, b in valid[1:]:
        pa, pb = merged[-1]
        if a <= pb:
            merged[-1] = (pa, max(pb, b))
        else:
            merged.append((a, b))
    return merged


def mask_message_text(text: str, spans: list[tuple[int, int]]) -> str:
    """Delete character ranges from ``text`` (inclusive start, exclusive
    end). Overlapping/adjacent spans are merged before deletion.
    Degenerate spans (``start >= end``) are silently dropped."""
    merged = _merge_spans(spans)
    if not merged:
        return text
    out: list[str] = []
    cursor = 0
    for a, b in merged:
        a = max(0, a)
        b = min(len(text), b)
        if a > cursor:
            out.append(text[cursor:a])
        cursor = b
    if cursor < len(text):
        out.append(text[cursor:])
    return "".join(out)


def _agent_ids_in_dialogue(dialogue: Dialogue) -> list[int]:
    return sorted({m.agent_id for m in dialogue.messages})


def _format_other_responses(prior_messages: list[Message], self_id: int) -> str:
    parts = []
    for m in prior_messages:
        if m.agent_id == self_id:
            continue
        parts.append(f"--- Agent {m.agent_id} (round {m.round}) ---\n{m.text}")
    return "\n\n".join(parts)


def _build_agent_api_messages(
    system_prompt: str, dialogue: Dialogue, masked_history: list[Message],
    agent_id: int, final_round: int,
) -> list[dict]:
    """Rebuild the api-messages array the agent would send for the final
    round, given a history whose texts have been span-masked."""
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for r in range(1, final_round):
        round_msgs = [m for m in masked_history if m.round == r]
        if r == 1:
            user = INITIAL_USER_TEMPLATE.format(question=dialogue.question)
        else:
            prior = [m for m in masked_history if m.round == r - 1]
            user = LATER_ROUND_TEMPLATE.format(
                other_responses=_format_other_responses(prior, agent_id)
            )
        self_resp_list = [m for m in round_msgs if m.agent_id == agent_id]
        messages.append({"role": "user", "content": user})
        if self_resp_list:
            messages.append({"role": "assistant", "content": self_resp_list[0].text})
        else:
            messages.append({"role": "assistant", "content": ""})
    # Final-round user turn (ask for fresh answer given masked history).
    prior = [m for m in masked_history if m.round == final_round - 1]
    final_user = LATER_ROUND_TEMPLATE.format(
        other_responses=_format_other_responses(prior, agent_id)
    )
    messages.append({"role": "user", "content": final_user})
    return messages


def replay_final_round(
    *,
    dialogue: Dialogue, claims_doc: dict, drop_type: str,
    llm_client: LLMClient, model: str, temperature: float = 0.0,
) -> dict:
    rounds = sorted({m.round for m in dialogue.messages})
    if not rounds:
        raise ValueError("dialogue has no messages")
    final_round = rounds[-1]
    if final_round < 2:
        raise ValueError("final round must be >= 2 for ablation replay")

    masked = reconstruct_masked_history(
        dialogue, claims_doc, drop_type=drop_type, up_to_round=final_round - 1,
    )
    agent_ids = _agent_ids_in_dialogue(dialogue)
    agents = make_default_agents(len(agent_ids))

    new_final_round: list[Message] = []
    for agent, agent_id in zip(agents, agent_ids):
        api_msgs = _build_agent_api_messages(
            agent.system_prompt, dialogue, masked, agent_id, final_round,
        )
        response = llm_client.chat(api_msgs, model=model, temperature=temperature)
        new_final_round.append(Message(
            agent_id=agent_id, round=final_round, text=response,
        ))

    post_final, _ = majority_vote(new_final_round)
    pre_final = dialogue.final_answer
    gold = str(dialogue.gold_answer).strip()

    def _eq(a: Optional[str]) -> bool:
        return a is not None and str(a).strip() == gold

    return {
        "qid": dialogue.question_id,
        "drop_type": drop_type,
        "pre_final": pre_final,
        "post_final": post_final,
        "gold": gold,
        "correct_with": _eq(pre_final),
        "correct_without": _eq(post_final),
        "skipped": False,
    }


def run_ablation(
    *,
    cfg: Config, qids: list[str], llm_client: LLMClient,
    max_new_llm_calls: int = MAX_NEW_LLM_CALLS_DEFAULT,
    types: Optional[list[str]] = None,
) -> list[dict]:
    type_list = list(types or CLAIM_TYPES)
    rows: list[dict] = []
    baseline_calls = llm_client.call_count

    for qid in qids:
        try:
            dialogue, claims_doc = load_dialogue_and_claims(cfg, qid)
        except FileNotFoundError as e:
            for t in type_list:
                rows.append({
                    "qid": qid, "drop_type": t, "skipped": True,
                    "skip_reason": f"missing artifact: {e}",
                })
            continue

        for t in type_list:
            used = llm_client.call_count - baseline_calls
            if used >= max_new_llm_calls:
                rows.append({
                    "qid": qid, "drop_type": t, "skipped": True,
                    "skip_reason": f"budget exceeded ({used}/{max_new_llm_calls})",
                })
                continue
            try:
                row = replay_final_round(
                    dialogue=dialogue, claims_doc=claims_doc, drop_type=t,
                    llm_client=llm_client, model=cfg.model,
                    temperature=cfg.temperature,
                )
            except Exception as exc:  # noqa: BLE001
                rows.append({
                    "qid": qid, "drop_type": t, "skipped": True,
                    "skip_reason": f"{type(exc).__name__}: {exc}",
                })
                continue
            rows.append(row)
    return rows


def reconstruct_masked_history(
    dialogue: Dialogue, claims_doc: dict, *, drop_type: str, up_to_round: int,
) -> list[Message]:
    """Return a new list of Messages mirroring ``dialogue.messages`` where
    rounds ``1..up_to_round`` have all ``drop_type`` claims span-masked
    out. Message count and (agent_id, round) indices are preserved so
    downstream replay can rebuild the transcript."""
    spans_by_key: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for c in claims_doc.get("claims", []):
        if c.get("type") != drop_type:
            continue
        r = int(c["round"])
        if r > up_to_round:
            continue
        key = (int(c["agent_id"]), r)
        a, b = c["source_message_span"]
        spans_by_key.setdefault(key, []).append((int(a), int(b)))

    new_messages: list[Message] = []
    for m in dialogue.messages:
        if m.round > up_to_round or (m.agent_id, m.round) not in spans_by_key:
            new_messages.append(Message(
                agent_id=m.agent_id, round=m.round, text=m.text,
            ))
            continue
        masked = mask_message_text(m.text, spans_by_key[(m.agent_id, m.round)])
        new_messages.append(Message(
            agent_id=m.agent_id, round=m.round, text=masked,
        ))
    return new_messages
