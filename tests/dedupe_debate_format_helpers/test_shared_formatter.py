from __future__ import annotations

import pytest

from agentdiet.types import Message


def test_format_other_responses_importable_from_debate():
    from agentdiet.debate import format_other_responses
    assert callable(format_other_responses)


def test_no_local_definition_in_ablate():
    import agentdiet.analysis.ablate as ablate
    # Private duplicate must be gone.
    assert not hasattr(ablate, "_format_other_responses"), \
        "ablate must not redefine _format_other_responses; import from debate"


def test_ablate_reuses_debate_formatter():
    from agentdiet.debate import format_other_responses
    import agentdiet.analysis.ablate as ablate
    # If ablate uses the shared one, referencing format_other_responses
    # via its module attribute should resolve to debate's function.
    assert getattr(ablate, "format_other_responses", None) is format_other_responses


def test_filters_self_id_and_joins_with_blank_lines():
    from agentdiet.debate import format_other_responses
    msgs = [
        Message(agent_id=0, round=1, text="alpha"),
        Message(agent_id=1, round=1, text="beta"),
        Message(agent_id=2, round=1, text="gamma"),
    ]
    out = format_other_responses(msgs, self_id=1)
    # Self excluded; others included.
    assert "alpha" in out and "gamma" in out
    assert "beta" not in out
    # Two blocks joined by blank line.
    assert "\n\n" in out
    # Header format.
    assert "--- Agent 0 (round 1) ---" in out
    assert "--- Agent 2 (round 1) ---" in out


def test_empty_input_returns_empty_string():
    from agentdiet.debate import format_other_responses
    assert format_other_responses([], self_id=0) == ""


def test_all_filtered_returns_empty_string():
    from agentdiet.debate import format_other_responses
    msgs = [Message(agent_id=0, round=1, text="only self")]
    assert format_other_responses(msgs, self_id=0) == ""
