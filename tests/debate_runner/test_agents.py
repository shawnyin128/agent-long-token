from __future__ import annotations

import pytest

from agentdiet.agents import (
    DEFAULT_ROLE_ORDER,
    SYSTEM_PROMPTS,
    Agent,
    make_default_agents,
)


def test_three_prompts_all_distinct():
    vals = list(SYSTEM_PROMPTS.values())
    assert len(set(vals)) == 3


def test_prompts_all_mention_hash_format():
    for role, p in SYSTEM_PROMPTS.items():
        assert "####" in p, f"role {role} prompt missing '####' instruction"


def test_make_default_agents_yields_differentiated_roles():
    agents = make_default_agents(3)
    assert [a.role for a in agents] == list(DEFAULT_ROLE_ORDER)
    assert len({a.system_prompt for a in agents}) == 3


def test_make_default_agents_rejects_excess():
    with pytest.raises(ValueError):
        make_default_agents(4)


def test_build_api_messages_shape():
    a = Agent.make(0, "solver")
    msgs = a.build_api_messages("hello")
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == SYSTEM_PROMPTS["solver"]
    assert msgs[-1] == {"role": "user", "content": "hello"}


def test_record_turn_accumulates_history():
    a = Agent.make(0, "solver")
    a.record_turn("q", "a1")
    a.record_turn("followup", "a2")
    assert a.history == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "followup"},
        {"role": "assistant", "content": "a2"},
    ]


def test_build_api_messages_includes_history():
    a = Agent.make(0, "solver")
    a.record_turn("q", "a1")
    msgs = a.build_api_messages("next")
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "user"]
    assert msgs[-1]["content"] == "next"
