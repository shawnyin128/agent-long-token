"""Code-domain debate protocol (spec §4.3b).

Three roles — proposer, reviewer, integrator — produce a uniform
"## Notes / ## Code" message schema each round. Aggregation happens
later via functional clustering on the round-3 code blocks.

Thinking traces are NOT part of the visible message — each agent's
internal reasoning stays private; only the (Notes, Code) pair is
broadcast to the next round.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from agentdiet.eval.base import CodeQuestion
from agentdiet.llm_client import LLMClient


CodeRole = Literal["proposer", "reviewer", "integrator"]


_FORMAT_BLOCK = (
    "Format your output EXACTLY as:\n"
    "## Notes\n"
    "<your role-specific notes, at most 200 words>\n\n"
    "## Code\n"
    "```python\n"
    "<your full Python solution>\n"
    "```\n"
    "Do not include any text outside these two sections."
)

PROPOSER_PROMPT = (
    "You are a code proposer. Each round, write a complete Python "
    "solution that you can defend. Refine your approach across rounds "
    "based on the other agents' work. " + _FORMAT_BLOCK
)

REVIEWER_PROMPT = (
    "You are a code reviewer. Read the other agents' code from the "
    "previous round, identify concrete issues (bugs, missing edge cases, "
    "complexity problems), AND produce your own code that addresses "
    "what you found. " + _FORMAT_BLOCK
)

INTEGRATOR_PROMPT = (
    "You are a code integrator. Synthesize the prior agents' outputs. "
    "Name which prior ideas you kept and which you dropped. Produce "
    "the best code you can. " + _FORMAT_BLOCK
)


CODE_ROLE_PROMPTS: dict[CodeRole, str] = {
    "proposer": PROPOSER_PROMPT,
    "reviewer": REVIEWER_PROMPT,
    "integrator": INTEGRATOR_PROMPT,
}

DEFAULT_CODE_ROLE_ORDER: tuple[CodeRole, ...] = (
    "proposer", "reviewer", "integrator",
)


# ---------------------------------------------------------------------------
# Schema parser
# ---------------------------------------------------------------------------


_NOTES_RE = re.compile(
    r"##\s*Notes\s*\n(?P<notes>.*?)(?=##\s*Code|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_CODE_RE = re.compile(
    r"##\s*Code\s*\n```python\s*\n(?P<code>.*?)```",
    re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class CodeMessage:
    agent_id: int
    round: int
    role: CodeRole
    notes: str
    code: str
    raw: str


def parse_code_message(text: str) -> tuple[str, str]:
    """Extract (notes, code) substrings from a model response.

    Missing or malformed sections return empty strings; callers must
    treat empty `code` as a parse failure.
    """
    notes = ""
    code = ""
    m_notes = _NOTES_RE.search(text)
    if m_notes:
        notes = m_notes.group("notes").strip()
    m_code = _CODE_RE.search(text)
    if m_code:
        code = m_code.group("code").rstrip()
    return notes, code


# ---------------------------------------------------------------------------
# Agents + Dialogue
# ---------------------------------------------------------------------------


@dataclass
class CodeAgent:
    id: int
    role: CodeRole
    system_prompt: str

    @classmethod
    def make(cls, agent_id: int, role: CodeRole) -> "CodeAgent":
        return cls(id=agent_id, role=role, system_prompt=CODE_ROLE_PROMPTS[role])


def make_default_code_agents(n_agents: int = 3) -> list[CodeAgent]:
    if n_agents > len(DEFAULT_CODE_ROLE_ORDER):
        raise ValueError(
            f"Only {len(DEFAULT_CODE_ROLE_ORDER)} default code roles "
            f"defined; requested {n_agents}"
        )
    return [
        CodeAgent.make(i, DEFAULT_CODE_ROLE_ORDER[i])
        for i in range(n_agents)
    ]


@dataclass
class CodeDialogue:
    question_id: str
    prompt: str
    messages: list[CodeMessage]
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


INITIAL_CODE_USER_TEMPLATE = (
    "Solve the following coding problem:\n\n{prompt}"
)

LATER_CODE_USER_TEMPLATE = (
    "Coding problem:\n\n{prompt}\n\n"
    "Other agents' previous-round outputs:\n\n{other_outputs}\n\n"
    "Now produce your own (Notes, Code) for this round, following the "
    "schema in your role description. Do not refer to other agents by "
    "name in code; only in Notes."
)


def _format_other_outputs(
    others: list[CodeMessage],
) -> str:
    parts: list[str] = []
    for m in others:
        parts.append(
            f"--- Agent {m.agent_id} ({m.role}, round {m.round}) ---\n"
            f"## Notes\n{m.notes}\n\n"
            f"## Code\n```python\n{m.code}\n```"
        )
    return "\n\n".join(parts) if parts else "(no prior outputs)"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run_code_debate(
    *,
    question: CodeQuestion,
    llm_client: LLMClient,
    model: str,
    n_agents: int = 3,
    n_rounds: int = 3,
    temperature: float = 0.0,
    thinking: bool = False,
    agents: Optional[list[CodeAgent]] = None,
) -> CodeDialogue:
    """Run an N-agent x R-round code debate.

    Returns a CodeDialogue with messages in (round, agent_id) order.
    Token totals are tracked via LLMClient.chat_full and are accessible
    via dialogue.meta["total_prompt_tokens"] / ["total_completion_tokens"].
    """
    if agents is None:
        agents = make_default_code_agents(n_agents)
    elif len(agents) != n_agents:
        raise ValueError(f"n_agents={n_agents} but got {len(agents)} agents")

    messages: list[CodeMessage] = []
    total_prompt = 0
    total_completion = 0

    for r in range(1, n_rounds + 1):
        if r == 1:
            user_for_all = INITIAL_CODE_USER_TEMPLATE.format(prompt=question.prompt)
        round_outputs: list[CodeMessage] = []
        prior = [m for m in messages if m.round == r - 1] if r > 1 else []
        for agent in agents:
            if r == 1:
                user_content = user_for_all
            else:
                others = [m for m in prior if m.agent_id != agent.id]
                user_content = LATER_CODE_USER_TEMPLATE.format(
                    prompt=question.prompt,
                    other_outputs=_format_other_outputs(others),
                )
            api_msgs = [
                {"role": "system", "content": agent.system_prompt},
                {"role": "user", "content": user_content},
            ]
            result = llm_client.chat_full(
                api_msgs, model, temperature, thinking=thinking,
            )
            notes, code = parse_code_message(result.response)
            round_outputs.append(CodeMessage(
                agent_id=agent.id, round=r, role=agent.role,
                notes=notes, code=code, raw=result.response,
            ))
            total_prompt += result.prompt_tokens or 0
            total_completion += result.completion_tokens or 0
        messages.extend(round_outputs)

    expected = n_agents * n_rounds
    if len(messages) != expected:
        raise AssertionError(
            f"CodeDialogue invariant violated: {len(messages)} messages, "
            f"expected {expected}"
        )

    return CodeDialogue(
        question_id=question.qid,
        prompt=question.prompt,
        messages=messages,
        meta={
            "model": model,
            "temperature": temperature,
            "n_agents": n_agents,
            "n_rounds": n_rounds,
            "thinking": thinking,
            "roles": [a.role for a in agents],
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "timestamp": time.time(),
        },
    )
