"""Token-matched majority-voting (self-consistency) baseline (spec §5.3).

calibrate_n() picks the per-cell N that gives total-token parity with
3x3 debate, with a floor of N=3 to keep voting semantics meaningful.
run_voting() executes N independent SA samples on a question and
majority-votes the parsed answers.

Both functions are pure (no filesystem). The thinking-axis-grid CLI
serializes CalibrationResult to artifacts/sc_calibration_{cell}.json.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

from agentdiet.aggregate import majority_vote
from agentdiet.dataset import parse_answer
from agentdiet.llm_client import LLMClient
from agentdiet.types import Message


@dataclass(frozen=True)
class CalibrationResult:
    N_raw: int
    N: int
    mean_debate_tokens: float
    mean_sa_tokens: float
    over_budget_factor: float
    floor_active: bool


@dataclass(frozen=True)
class VotingResult:
    samples: list[str]
    parsed_answers: list[Optional[str]]
    final_answer: Optional[str]
    total_tokens: int


N_FLOOR = 3


def calibrate_n(
    debate_token_counts: list[int],
    sa_token_counts: list[int],
) -> CalibrationResult:
    """Total-token match with N>=3 floor (spec §5.3).

    N_raw = ceil(mean(debate) / mean(sa)).
    N = max(N_raw, 3).
    over_budget_factor = (N * mean_sa) / mean_debate.

    Raises ValueError on empty input or any non-positive count.
    """
    if not debate_token_counts:
        raise ValueError("debate_token_counts is empty")
    if not sa_token_counts:
        raise ValueError("sa_token_counts is empty")
    if any(c <= 0 for c in debate_token_counts):
        raise ValueError("debate_token_counts contains a non-positive count")
    if any(c <= 0 for c in sa_token_counts):
        raise ValueError("sa_token_counts contains a non-positive count")

    mean_debate = sum(debate_token_counts) / len(debate_token_counts)
    mean_sa = sum(sa_token_counts) / len(sa_token_counts)
    n_raw = math.ceil(mean_debate / mean_sa)
    n = max(n_raw, N_FLOOR)
    over_budget_factor = (n * mean_sa) / mean_debate
    return CalibrationResult(
        N_raw=n_raw,
        N=n,
        mean_debate_tokens=mean_debate,
        mean_sa_tokens=mean_sa,
        over_budget_factor=over_budget_factor,
        floor_active=(n_raw < N_FLOOR),
    )


def run_voting(
    *,
    question: str,
    n_samples: int,
    llm_client: LLMClient,
    model: str,
    system_prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.95,
    thinking: bool = False,
    parser: Callable[[str], Optional[str]] = parse_answer,
) -> VotingResult:
    """Execute n_samples independent SA calls and majority-vote.

    To force deterministically-different responses while reusing the
    cache, each sample's user message is prefixed with "Sample id: K"
    so cache keys differ per sample (D5 in plan).
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    samples: list[str] = []
    total_tokens = 0
    for k in range(n_samples):
        user_content = f"Sample id: {k}\n\n{question}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        result = llm_client.chat_full(
            messages, model, temperature,
            thinking=thinking, top_p=top_p,
        )
        samples.append(result.response)
        prompt_tokens = result.prompt_tokens or 0
        completion_tokens = result.completion_tokens or 0
        total_tokens += prompt_tokens + completion_tokens

    parsed_answers: list[Optional[str]] = [parser(s) for s in samples]
    pseudo_messages = [
        Message(agent_id=k, round=1, text=samples[k])
        for k in range(n_samples)
    ]
    final, _ = majority_vote(pseudo_messages, parse_fn=parser)

    return VotingResult(
        samples=samples,
        parsed_answers=parsed_answers,
        final_answer=final,
        total_tokens=total_tokens,
    )
