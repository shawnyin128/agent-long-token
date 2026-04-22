from __future__ import annotations

from agentdiet.aggregate import majority_vote
from agentdiet.types import Message


def _msgs(agent_texts: list[tuple[int, str]]) -> list[Message]:
    return [Message(agent_id=i, round=3, text=t) for i, t in agent_texts]


def test_all_agree():
    w, per = majority_vote(_msgs([(0, "#### 42"), (1, "#### 42"), (2, "#### 42")]))
    assert w == "42"
    assert per == {0: "42", 1: "42", 2: "42"}


def test_2_of_3():
    w, _ = majority_vote(_msgs([(0, "#### 42"), (1, "#### 42"), (2, "#### 7")]))
    assert w == "42"


def test_all_distinct_returns_none():
    w, per = majority_vote(_msgs([(0, "#### 1"), (1, "#### 2"), (2, "#### 3")]))
    assert w is None
    assert per == {0: "1", 1: "2", 2: "3"}


def test_two_unparseable_one_answer():
    w, per = majority_vote(_msgs([(0, "no number here"), (1, "blah"), (2, "#### 7")]))
    assert w == "7"
    assert per == {0: None, 1: None, 2: "7"}


def test_all_unparseable():
    w, per = majority_vote(_msgs([(0, "x"), (1, "y"), (2, "z")]))
    assert w is None
    assert per == {0: None, 1: None, 2: None}


def test_numeric_equivalence_42_vs_42_point_0():
    w, _ = majority_vote(_msgs([(0, "#### 42"), (1, "#### 42.0"), (2, "#### 7")]))
    assert w == "42"


def test_even_split_2_vs_2_is_none():
    w, _ = majority_vote(
        _msgs([(0, "#### 1"), (1, "#### 1"), (2, "#### 2"), (3, "#### 2")])
    )
    assert w is None


def test_empty_messages_returns_none():
    w, per = majority_vote([])
    assert w is None
    assert per == {}
