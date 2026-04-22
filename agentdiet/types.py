from __future__ import annotations

from typing import Any, Literal, Optional, get_args

from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt, model_validator


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


ClaimType = Literal[
    "proposal", "evidence", "correction", "agreement", "question", "other"
]
CLAIM_TYPES: tuple[str, ...] = get_args(ClaimType)


class Claim(BaseModel):
    id: str
    text: str
    agent_id: NonNegativeInt
    round: PositiveInt
    type: ClaimType
    source_message_span: tuple[int, int]

    @model_validator(mode="after")
    def _validate_span(self) -> "Claim":
        start, end = self.source_message_span
        if start < 0 or end < 0:
            raise ValueError(f"span bounds must be non-negative, got {(start, end)}")
        if start >= end:
            raise ValueError(f"span start must be < end, got {(start, end)}")
        return self
