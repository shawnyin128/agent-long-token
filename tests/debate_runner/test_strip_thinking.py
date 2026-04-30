"""strip_thinking_trace removes <think>...</think> before broadcast."""
from __future__ import annotations

from agentdiet.debate import format_other_responses, strip_thinking_trace
from agentdiet.types import Message


def test_strips_simple_block():
    text = "<think>\nLet me think...\n</think>\n\nThe answer is 18.\n#### 18"
    cleaned = strip_thinking_trace(text)
    assert "<think>" not in cleaned
    assert "</think>" not in cleaned
    assert "Let me think" not in cleaned
    assert "#### 18" in cleaned


def test_strips_multiple_blocks():
    text = (
        "<think>first</think>\n"
        "Step 1.\n"
        "<think>second</think>\n"
        "Step 2.\n#### 7"
    )
    cleaned = strip_thinking_trace(text)
    assert "first" not in cleaned
    assert "second" not in cleaned
    assert "Step 1" in cleaned
    assert "#### 7" in cleaned


def test_no_block_is_no_op():
    text = "Plain answer.\n#### 42"
    assert strip_thinking_trace(text) == text


def test_empty_input():
    assert strip_thinking_trace("") == ""


def test_case_insensitive_tag():
    text = "<THINK>upper case</THINK>\nresult\n#### 3"
    cleaned = strip_thinking_trace(text)
    assert "upper case" not in cleaned
    assert "result" in cleaned


def test_multiline_block_with_special_chars():
    text = (
        "<think>\n"
        "Hmm, the equation is x + 7 = 19.\n"
        "Solving: x = 19 - 7 = 12.\n"
        "Wait, let me double-check.\n"
        "</think>\n\n"
        "x = 12\n#### 12"
    )
    cleaned = strip_thinking_trace(text)
    assert "double-check" not in cleaned
    assert "x = 12" in cleaned


def test_format_other_responses_strips_thinking():
    """Per spec §4.3, agents' thinking traces are NOT broadcast across rounds."""
    msgs = [
        Message(agent_id=1, round=1,
                text="<think>my private reasoning</think>\nMy answer: 18.\n#### 18"),
        Message(agent_id=2, round=1,
                text="<think>another private</think>\n#### 9"),
    ]
    formatted = format_other_responses(msgs, self_id=0)
    assert "private reasoning" not in formatted
    assert "another private" not in formatted
    assert "<think>" not in formatted
    assert "</think>" not in formatted
    assert "My answer: 18" in formatted
    assert "#### 18" in formatted
    assert "#### 9" in formatted


def test_format_other_responses_excludes_self():
    msgs = [
        Message(agent_id=0, round=1, text="my own response"),
        Message(agent_id=1, round=1, text="peer response"),
    ]
    formatted = format_other_responses(msgs, self_id=0)
    assert "my own response" not in formatted
    assert "peer response" in formatted
