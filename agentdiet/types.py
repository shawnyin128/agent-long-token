from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt


class Message(BaseModel):
    agent_id: NonNegativeInt
    round: PositiveInt
    text: str


class Dialogue(BaseModel):
    question_id: str
    question: str
    gold_answer: str
    messages: list[Message] = Field(default_factory=list)
    final_answer: Optional[str] = None
    meta: dict[str, Any] = Field(default_factory=dict)
