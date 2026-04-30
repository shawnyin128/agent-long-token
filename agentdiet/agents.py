from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Role = Literal["solver", "skeptic", "synthesizer"]


FORMAT_INSTR = (
    "End your response with a line containing exactly '#### <answer>' "
    "(for example '#### 42')"
)

SOLVER_PROMPT = (
    "You are a careful math problem solver. Work through the problem step "
    "by step, showing your calculation clearly. " + FORMAT_INSTR + "."
)

SKEPTIC_PROMPT = (
    "You are a skeptical math reviewer. Read the solutions from other "
    "agents carefully. Check each step for computational errors, missed "
    "conditions, or unit mistakes. Provide your own corrected solution "
    "with clear reasoning. " + FORMAT_INSTR + "."
)

SYNTHESIZER_PROMPT = (
    "You are a synthesizer. Compare the other agents' solutions. Identify "
    "where they agree and where they disagree. Produce a final reasoned "
    "answer based on the strongest argument. " + FORMAT_INSTR + "."
)


SYSTEM_PROMPTS: dict[Role, str] = {
    "solver": SOLVER_PROMPT,
    "skeptic": SKEPTIC_PROMPT,
    "synthesizer": SYNTHESIZER_PROMPT,
}

DEFAULT_ROLE_ORDER: tuple[Role, ...] = ("solver", "skeptic", "synthesizer")


@dataclass
class Agent:
    id: int
    role: Role
    system_prompt: str
    history: list[dict] = field(default_factory=list)

    @classmethod
    def make(cls, agent_id: int, role: Role) -> "Agent":
        return cls(id=agent_id, role=role, system_prompt=SYSTEM_PROMPTS[role])

    def build_api_messages(self, round_user_content: str) -> list[dict]:
        """Messages to send this turn: system + accumulated history + new user turn."""
        return (
            [{"role": "system", "content": self.system_prompt}]
            + self.history
            + [{"role": "user", "content": round_user_content}]
        )

    def record_turn(self, user_content: str, assistant_content: str) -> None:
        self.history.append({"role": "user", "content": user_content})
        self.history.append({"role": "assistant", "content": assistant_content})


def make_default_agents(
    n_agents: int = 3,
    prompts: list[str] | None = None,
) -> list[Agent]:
    if n_agents > len(DEFAULT_ROLE_ORDER):
        raise ValueError(
            f"Only {len(DEFAULT_ROLE_ORDER)} default roles defined; requested {n_agents}"
        )
    if prompts is None:
        return [Agent.make(i, DEFAULT_ROLE_ORDER[i]) for i in range(n_agents)]
    if len(prompts) != n_agents:
        raise ValueError(
            f"prompts list has {len(prompts)} entries; requested {n_agents} agents"
        )
    return [
        Agent(
            id=i,
            role=DEFAULT_ROLE_ORDER[i],
            system_prompt=prompts[i],
        )
        for i in range(n_agents)
    ]
