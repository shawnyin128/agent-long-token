from __future__ import annotations

from collections import Counter
from typing import Callable, Optional

from agentdiet.dataset import parse_answer
from agentdiet.types import Message


def majority_vote(
    last_round_messages: list[Message],
    parse_fn: Callable[[str], Optional[str]] = parse_answer,
) -> tuple[Optional[str], dict[int, Optional[str]]]:
    """Majority-vote aggregator with per-user-decision D1 tie handling.

    Returns (winner_or_None, per_agent_answers).
    Ties (no unique plurality) -> winner is None.
    """
    per_agent: dict[int, Optional[str]] = {}
    for m in last_round_messages:
        per_agent[m.agent_id] = parse_fn(m.text)

    parsed = [a for a in per_agent.values() if a is not None]
    if not parsed:
        return None, per_agent

    counts = Counter(parsed)
    top = counts.most_common()
    # If top-1 is strictly greater than top-2 (or there is no top-2), we have a winner.
    if len(top) == 1:
        return top[0][0], per_agent
    if top[0][1] > top[1][1]:
        return top[0][0], per_agent
    return None, per_agent
