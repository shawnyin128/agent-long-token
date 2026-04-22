from __future__ import annotations

import time

from agentdiet.agents import SOLVER_PROMPT
from agentdiet.dataset import Question, parse_answer
from agentdiet.debate import INITIAL_USER_TEMPLATE
from agentdiet.llm_client import LLMClient
from agentdiet.types import Dialogue, Message


def run_single_agent(
    question: Question,
    llm_client: LLMClient,
    model: str,
    temperature: float = 0.0,
    system_prompt: str = SOLVER_PROMPT,
) -> Dialogue:
    """Baseline: one solver-agent turn on the question. Returns a Dialogue
    with exactly one Message (agent_id=0, round=1) so it stays schema-
    compatible with run_debate outputs."""
    user_content = INITIAL_USER_TEMPLATE.format(question=question.question)
    api_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    response = llm_client.chat(api_messages, model=model, temperature=temperature)
    message = Message(agent_id=0, round=1, text=response)
    final = parse_answer(response)
    return Dialogue(
        question_id=question.qid,
        question=question.question,
        gold_answer=question.gold_answer,
        messages=[message],
        final_answer=final,
        meta={
            "method": "single",
            "model": model,
            "temperature": temperature,
            "timestamp": time.time(),
        },
    )
