"""Schema-guided claim extraction.

The extractor turns one debate message into a list of ``Claim`` records
(6-type taxonomy, spec §3.4). Prompts live here as module constants so
they are diff-able, unit-testable, and easy to audit.

Design notes (see ``claim-extraction.plan.yaml``):

  * One LLM call per message (D1).
  * LLM returns a verbatim ``quote`` substring and we compute
    ``source_message_span`` via ``str.index`` (D2) — avoids off-by-N.
  * Prompt-only JSON + pydantic validation + one strict retry (D3).
  * Failures after the strict retry are logged to
    ``artifacts/failures/claim_extraction/`` and the message is marked
    ``extraction_failed``; the pipeline does not abort.
"""
from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from agentdiet.llm_client import LLMClient
from agentdiet.types import CLAIM_TYPES, Claim, Dialogue, Message


log = logging.getLogger(__name__)


TYPE_DEFINITIONS: dict[str, str] = {
    "proposal": (
        "Introduces a new candidate answer or a new reasoning path that "
        "has not yet been stated in the debate."
    ),
    "evidence": (
        "Supports a position with an explicit computation, derivation, "
        "citation, or numeric step-by-step calculation."
    ),
    "correction": (
        "Identifies a concrete error in a previous claim and proposes a "
        "fix (e.g. wrong arithmetic, missed unit, misread condition)."
    ),
    "agreement": (
        "Affirms a previous claim without adding new information — "
        "a bare 'I agree with agent K' with no fresh reasoning."
    ),
    "question": (
        "A direct question aimed at another agent (not rhetorical)."
    ),
    "other": (
        "Pleasantries, meta-discourse, off-topic content, or filler "
        "that does not fit any of the above."
    ),
}


def _format_type_block() -> str:
    lines = []
    for t in CLAIM_TYPES:
        lines.append(f"  - {t}: {TYPE_DEFINITIONS[t]}")
    return "\n".join(lines)


SYSTEM_PROMPT = f"""You extract structured claims from a single message produced \
by one agent during a multi-agent math debate.

A CLAIM is a minimal, atomic assertion the agent is making. Split the message \
into the smallest self-contained claims. Each claim must be classified into \
exactly one of these 6 types:

{_format_type_block()}

You MUST output a single JSON array. Each element is an object with keys:
  - "type": one of {list(CLAIM_TYPES)}
  - "text": your paraphrase of the claim, short and self-contained (<= 30 words)
  - "quote": a VERBATIM substring of the message text that best supports this \
claim. It MUST appear in the message character-for-character (copy-paste). Do \
not paraphrase inside "quote". Do not add ellipses.

Output ONLY the JSON array. No prose before or after. No markdown fences."""


FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    {
        "message_text": (
            "Let me compute: Janet has 16 eggs per day. She eats 3 and bakes "
            "with 4, leaving 16 - 3 - 4 = 9 eggs to sell. At $2 each, that is "
            "9 * 2 = 18 dollars. #### 18"
        ),
        "claims_json": json.dumps([
            {
                "type": "proposal",
                "text": "Final answer is 18 dollars.",
                "quote": "#### 18",
            },
            {
                "type": "evidence",
                "text": "16 - 3 - 4 = 9 eggs remain to sell.",
                "quote": "16 - 3 - 4 = 9 eggs to sell",
            },
            {
                "type": "evidence",
                "text": "9 eggs at $2 each equals $18.",
                "quote": "9 * 2 = 18 dollars",
            },
        ]),
    },
    {
        "message_text": (
            "I agree with agent 0 that she has 9 eggs left, but the price is "
            "$2 per egg not per dozen, so agent 1's calculation of $1.50 is "
            "wrong — the correct total is 9 * 2 = 18. #### 18"
        ),
        "claims_json": json.dumps([
            {
                "type": "agreement",
                "text": "Agrees with agent 0 that 9 eggs remain.",
                "quote": "I agree with agent 0 that she has 9 eggs left",
            },
            {
                "type": "correction",
                "text": "Agent 1's $1.50 figure is wrong; price is per egg.",
                "quote": "agent 1's calculation of $1.50 is wrong",
            },
            {
                "type": "proposal",
                "text": "Correct total is $18.",
                "quote": "#### 18",
            },
        ]),
    },
    {
        "message_text": (
            "Agent 2, can you double-check whether the 4 eggs for baking are "
            "per day or per week? Great discussion so far! #### 18"
        ),
        "claims_json": json.dumps([
            {
                "type": "question",
                "text": "Asks agent 2 to verify whether 4 eggs is daily or weekly.",
                "quote": "Agent 2, can you double-check whether the 4 eggs for baking are per day or per week?",
            },
            {
                "type": "other",
                "text": "Social pleasantry.",
                "quote": "Great discussion so far!",
            },
            {
                "type": "proposal",
                "text": "Final answer is 18.",
                "quote": "#### 18",
            },
        ]),
    },
]


def _format_few_shot() -> str:
    blocks = []
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        blocks.append(
            f"Example {i}:\n"
            f"MESSAGE:\n{ex['message_text']}\n\n"
            f"OUTPUT:\n{ex['claims_json']}"
        )
    return "\n\n---\n\n".join(blocks)


FEW_SHOT_BLOCK = _format_few_shot()


def build_user_prompt(
    question: str, message_text: str, agent_id: int, round: int
) -> str:
    return (
        f"Here are three complete examples showing how to extract claims.\n\n"
        f"{FEW_SHOT_BLOCK}\n\n---\n\n"
        f"Now extract claims from the following message.\n\n"
        f"Original problem (context only, do NOT extract claims from it):\n"
        f"{question}\n\n"
        f"MESSAGE (from agent {agent_id}, round {round}):\n"
        f"{message_text}\n\n"
        f"OUTPUT:"
    )


def _build_retry_prompt(
    original_user: str, prior_response: str, reason: str
) -> str:
    return (
        f"{original_user}\n\n"
        f"Your previous output was rejected. Reason: {reason}\n"
        f"Previous output was:\n{prior_response}\n\n"
        f"Output ONLY a valid JSON array. Every element MUST have keys "
        f"'type' (one of {list(CLAIM_TYPES)}), 'text', and 'quote' where "
        f"'quote' is a verbatim substring of the MESSAGE. No markdown, "
        f"no prose."
    )


def _strip_json_fences(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        # remove ```json or ``` opening fence
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1 :]
        if s.endswith("```"):
            s = s[: -3]
        s = s.strip()
    return s


_JSON_LEGAL_ESCAPE = set('"\\/bfnrtu')


def _fix_json_escapes(raw: str) -> str:
    """Convert invalid '\\X' sequences (LaTeX '\\(' '\\frac' etc.) to
    '\\\\X' so JSON string parsing accepts them as literal backslash + X.
    Leaves legal escapes ('\\"' '\\n' '\\\\' '\\uXXXX') intact. Handles
    consecutive backslashes correctly (odd count adds one, even stays)."""
    out: list[str] = []
    i = 0
    n = len(raw)
    while i < n:
        if raw[i] != "\\":
            out.append(raw[i])
            i += 1
            continue
        j = i
        while j < n and raw[j] == "\\":
            j += 1
        run = j - i
        nxt = raw[j] if j < n else ""
        if run % 2 == 1 and nxt and nxt not in _JSON_LEGAL_ESCAPE:
            out.append("\\" * (run + 1))
        else:
            out.append("\\" * run)
        i = j
    return "".join(out)


def _parse_claims_payload(
    raw_response: str,
    message_text: str,
    qid: str,
    agent_id: int,
    round: int,
    start_idx: int,
) -> tuple[list[Claim], str | None]:
    """Parse + validate JSON payload. Return (claims, error) — error None on success."""
    stripped = _fix_json_escapes(_strip_json_fences(raw_response))
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as e:
        return [], f"json parse error: {e.msg}"
    if not isinstance(payload, list):
        return [], f"expected JSON array, got {type(payload).__name__}"

    claims: list[Claim] = []
    dropped: list[str] = []
    for i, item in enumerate(payload):
        if not isinstance(item, dict):
            dropped.append(f"item {i}: not an object")
            continue
        t = item.get("type")
        text = item.get("text")
        quote = item.get("quote")
        if t not in CLAIM_TYPES:
            dropped.append(f"item {i}: bad type {t!r}")
            continue
        if not isinstance(text, str) or not isinstance(quote, str):
            dropped.append(f"item {i}: missing text or quote")
            continue
        if not quote:
            dropped.append(f"item {i}: empty quote")
            continue
        idx = message_text.find(quote)
        if idx < 0:
            dropped.append(f"item {i}: quote not in message")
            continue
        span = (idx, idx + len(quote))
        try:
            c = Claim(
                id=f"{qid}_r{round}_a{agent_id}_c{start_idx + len(claims)}",
                text=text,
                agent_id=agent_id,
                round=round,
                type=t,
                source_message_span=span,
            )
        except ValidationError as e:
            dropped.append(f"item {i}: {e.errors()[0].get('msg', 'validation error')}")
            continue
        # defensive invariant: span within message text
        assert 0 <= c.source_message_span[0] < c.source_message_span[1] <= len(message_text), \
            f"span {c.source_message_span} out of range for len {len(message_text)}"
        claims.append(c)

    if not claims and dropped:
        return [], f"no valid claims (dropped {len(dropped)}): {dropped[0]}"
    return claims, None


def _failure_path(
    failures_dir: Path, qid: str, round: int, agent_id: int
) -> Path:
    return failures_dir / "claim_extraction" / f"{qid}_r{round}_a{agent_id}.json"


def _log_failure(
    failures_dir: Path,
    qid: str,
    round: int,
    agent_id: int,
    user_prompt: str,
    raw_response: str,
    reason: str,
) -> None:
    path = _failure_path(failures_dir, qid, round, agent_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "qid": qid,
        "round": round,
        "agent_id": agent_id,
        "reason": reason,
        "user_prompt": user_prompt,
        "raw_response": raw_response,
        "traceback": traceback.format_exc() if traceback.format_exc().strip() != "NoneType: None" else None,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    import os
    os.replace(tmp, path)


def extract_claims_from_message(
    *,
    message: Message,
    question: str,
    qid: str,
    llm_client: LLMClient,
    model: str,
    temperature: float = 0.0,
    failures_dir: Path,
    start_idx: int = 0,
) -> tuple[list[Claim], bool]:
    """Extract claims from one message.

    Returns (claims, extraction_failed). On success returns
    (claims, False). On final failure after retry returns ([], True)
    and writes the raw context to failures_dir/claim_extraction/.
    """
    user_prompt = build_user_prompt(
        question=question,
        message_text=message.text,
        agent_id=message.agent_id,
        round=message.round,
    )
    api_msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    raw1 = llm_client.chat(api_msgs, model=model, temperature=temperature)
    claims, err1 = _parse_claims_payload(
        raw1, message.text, qid, message.agent_id, message.round, start_idx
    )
    if err1 is None:
        return claims, False

    # Strict retry once.
    retry_user = _build_retry_prompt(user_prompt, raw1, err1)
    retry_msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": retry_user},
    ]
    raw2 = llm_client.chat(retry_msgs, model=model, temperature=temperature)
    claims, err2 = _parse_claims_payload(
        raw2, message.text, qid, message.agent_id, message.round, start_idx
    )
    if err2 is None:
        return claims, False

    _log_failure(
        failures_dir=failures_dir,
        qid=qid,
        round=message.round,
        agent_id=message.agent_id,
        user_prompt=retry_user,
        raw_response=raw2,
        reason=f"first: {err1} | retry: {err2}",
    )
    return [], True


def extract_claims_for_dialogue(
    *,
    dialogue: Dialogue,
    llm_client: LLMClient,
    model: str,
    temperature: float = 0.0,
    failures_dir: Path,
) -> dict[str, Any]:
    """Run extraction across every message in a dialogue.

    Returns a dict with: qid, claims (list[dict]), per_message_status
    (list of {agent_id, round, extraction_failed, n_claims}),
    extraction_failed (bool: any message failed).
    """
    all_claims: list[Claim] = []
    per_message: list[dict[str, Any]] = []
    any_failed = False
    for m in dialogue.messages:
        msg_claims, failed = extract_claims_from_message(
            message=m,
            question=dialogue.question,
            qid=dialogue.question_id,
            llm_client=llm_client,
            model=model,
            temperature=temperature,
            failures_dir=failures_dir,
            start_idx=0,
        )
        all_claims.extend(msg_claims)
        per_message.append({
            "agent_id": m.agent_id,
            "round": m.round,
            "extraction_failed": failed,
            "n_claims": len(msg_claims),
        })
        if failed:
            any_failed = True
    return {
        "qid": dialogue.question_id,
        "claims": [c.model_dump() for c in all_claims],
        "per_message_status": per_message,
        "extraction_failed": any_failed,
    }
