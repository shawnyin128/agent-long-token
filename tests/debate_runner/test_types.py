from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from agentdiet.types import Dialogue, Message


def test_message_valid():
    m = Message(agent_id=0, round=1, text="hello")
    assert m.agent_id == 0
    assert m.round == 1


def test_message_rejects_negative_agent():
    with pytest.raises(ValidationError):
        Message(agent_id=-1, round=1, text="x")


def test_message_rejects_zero_round():
    with pytest.raises(ValidationError):
        Message(agent_id=0, round=0, text="x")


def test_dialogue_defaults():
    d = Dialogue(question_id="q1", question="Q?", gold_answer="1")
    assert d.messages == []
    assert d.final_answer is None
    assert d.meta == {}


def test_dialogue_roundtrip_json():
    d = Dialogue(
        question_id="q1",
        question="Q?",
        gold_answer="42",
        messages=[Message(agent_id=0, round=1, text="hi")],
        final_answer="42",
        meta={"model": "m", "temperature": 0.0},
    )
    s = d.model_dump_json()
    d2 = Dialogue.model_validate_json(s)
    assert d2 == d
    assert json.loads(s)["meta"]["model"] == "m"
