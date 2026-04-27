from __future__ import annotations

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


def format_other_responses(messages_this_round: list[Message], self_id: int) -> str:
    parts = []
    for m in messages_this_round:
        if m.agent_id == self_id:
            continue
        parts.append(f"--- Agent {m.agent_id} (round {m.round}) ---\n{m.text}")
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
) -> Dialogue:
    if agents is None:
        agents = make_default_agents(n_agents)
    elif len(agents) != n_agents:
        raise ValueError(f"n_agents={n_agents} but got {len(agents)} agents")

    messages: list[Message] = []
    for r in range(1, n_rounds + 1):
        if r == 1:
            user_content_for_all = INITIAL_USER_TEMPLATE.format(question=question.question)
            round_messages: list[Message] = []
            for agent in agents:
                api_msgs = agent.build_api_messages(user_content_for_all)
                response = llm_client.chat(api_msgs, model=model, temperature=temperature)
                agent.record_turn(user_content_for_all, response)
                round_messages.append(Message(agent_id=agent.id, round=r, text=response))
            messages.extend(round_messages)
        else:
            prior_round = [m for m in messages if m.round == r - 1]
            round_messages = []
            for agent in agents:
                user_content = LATER_ROUND_TEMPLATE.format(
                    other_responses=format_other_responses(prior_round, agent.id)
                )
                api_msgs = agent.build_api_messages(user_content)
                response = llm_client.chat(api_msgs, model=model, temperature=temperature)
                agent.record_turn(user_content, response)
                round_messages.append(Message(agent_id=agent.id, round=r, text=response))
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
            "roles": [a.role for a in agents],
            "per_agent_final_answers": {str(k): v for k, v in per_agent.items()},
            "timestamp": time.time(),
        },
    )
