from __future__ import annotations

import re
import time
from typing import Optional

from agentdiet.agents import FORMAT_INSTR, Agent, make_default_agents
from agentdiet.aggregate import majority_vote
from agentdiet.dataset import Question
from agentdiet.llm_client import LLMClient
from agentdiet.types import Dialogue, Message


INITIAL_USER_TEMPLATE = "Solve the following problem:\n\n{question}"

LATER_ROUND_TEMPLATE = (
    "Here are the solutions from other agents in the previous round.\n"
    "{other_responses}\n\n"
    "Based on these and your own previous reasoning, provide an updated solution. "
    + FORMAT_INSTR + "."
)


_THINKING_TRACE_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def strip_thinking_trace(text: str) -> str:
    """Remove <think>...</think> blocks before broadcasting to peers.

    Per spec §4.3 condition 3: thinking traces are internal to each
    agent and MUST NOT be shared across rounds. Qwen3 with
    enable_thinking=True wraps reasoning in <think>...</think>; we
    strip those before showing the message to other agents.

    No-op on responses that don't contain a thinking block, so the
    same code path is safe for thinking-off and for non-Qwen3 models.
    """
    return _THINKING_TRACE_RE.sub("", text).strip()


def format_other_responses(messages_this_round: list[Message], self_id: int) -> str:
    parts = []
    for m in messages_this_round:
        if m.agent_id == self_id:
            continue
        parts.append(
            f"--- Agent {m.agent_id} (round {m.round}) ---\n"
            f"{strip_thinking_trace(m.text)}"
        )
    return "\n\n".join(parts)


def run_debate(
    question: Question,
    llm_client: LLMClient,
    model: str,
    n_agents: int = 3,
    n_rounds: int = 3,
    temperature: float = 0.0,
    agents: Optional[list[Agent]] = None,
    seed: Optional[int] = None,
    thinking: bool = False,
    prompt_variant: str = "cooperative",
) -> Dialogue:
    if agents is None:
        if prompt_variant == "cooperative":
            agents = make_default_agents(n_agents)
        else:
            from agentdiet.prompts import get_variant
            prompts = get_variant(prompt_variant)
            agents = make_default_agents(n_agents, prompts=prompts)
    elif len(agents) != n_agents:
        raise ValueError(f"n_agents={n_agents} but got {len(agents)} agents")

    messages: list[Message] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for r in range(1, n_rounds + 1):
        if r == 1:
            user_content_for_all = INITIAL_USER_TEMPLATE.format(question=question.question)
            round_messages: list[Message] = []
            for agent in agents:
                api_msgs = agent.build_api_messages(user_content_for_all)
                result = llm_client.chat_full(
                    api_msgs, model=model, temperature=temperature,
                    thinking=thinking,
                )
                agent.record_turn(user_content_for_all, result.response)
                round_messages.append(Message(agent_id=agent.id, round=r, text=result.response))
                total_prompt_tokens += result.prompt_tokens or 0
                total_completion_tokens += result.completion_tokens or 0
            messages.extend(round_messages)
        else:
            prior_round = [m for m in messages if m.round == r - 1]
            round_messages = []
            for agent in agents:
                user_content = LATER_ROUND_TEMPLATE.format(
                    other_responses=format_other_responses(prior_round, agent.id)
                )
                api_msgs = agent.build_api_messages(user_content)
                result = llm_client.chat_full(
                    api_msgs, model=model, temperature=temperature,
                    thinking=thinking,
                )
                agent.record_turn(user_content, result.response)
                round_messages.append(Message(agent_id=agent.id, round=r, text=result.response))
                total_prompt_tokens += result.prompt_tokens or 0
                total_completion_tokens += result.completion_tokens or 0
            messages.extend(round_messages)

    expected = n_agents * n_rounds
    if len(messages) != expected:
        raise AssertionError(
            f"Dialogue invariant violated: {len(messages)} messages, expected {expected}"
        )

    last_round = [m for m in messages if m.round == n_rounds]
    final, per_agent = majority_vote(last_round)

    return Dialogue(
        question_id=question.qid,
        question=question.question,
        gold_answer=question.gold_answer,
        messages=messages,
        final_answer=final,
        meta={
            "model": model,
            "temperature": temperature,
            "n_agents": n_agents,
            "n_rounds": n_rounds,
            "seed": seed,
            "thinking": thinking,
            "prompt_variant": prompt_variant,
            "roles": [a.role for a in agents],
            "per_agent_final_answers": {str(k): v for k, v in per_agent.items()},
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "timestamp": time.time(),
        },
    )
