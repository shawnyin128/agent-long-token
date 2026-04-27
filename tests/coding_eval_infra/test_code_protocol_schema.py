"""Notes/Code schema parser + role-prompt invariants."""
from __future__ import annotations

from agentdiet.debate.code_protocol import (
    CODE_ROLE_PROMPTS,
    INTEGRATOR_PROMPT,
    PROPOSER_PROMPT,
    REVIEWER_PROMPT,
    parse_code_message,
)


def test_parse_well_formed_message():
    text = (
        "## Notes\n"
        "Approach: use a hash set.\n\n"
        "## Code\n"
        "```python\n"
        "def f(x): return x * 2\n"
        "```\n"
    )
    notes, code = parse_code_message(text)
    assert "hash set" in notes
    assert code == "def f(x): return x * 2"


def test_parse_missing_notes_yields_empty_notes():
    text = (
        "## Code\n"
        "```python\n"
        "def f(): pass\n"
        "```\n"
    )
    notes, code = parse_code_message(text)
    assert notes == ""
    assert code == "def f(): pass"


def test_parse_missing_code_yields_empty_code():
    text = "## Notes\nNo code yet.\n"
    notes, code = parse_code_message(text)
    assert notes == "No code yet."
    assert code == ""


def test_parse_unterminated_code_fence_returns_empty_code():
    text = (
        "## Notes\nstart\n\n"
        "## Code\n```python\n"
        "def f(): pass\n"
        # missing closing ```
    )
    notes, code = parse_code_message(text)
    assert notes == "start"
    assert code == ""


def test_parse_handles_extra_text_inside_notes():
    text = (
        "## Notes\n"
        "I considered approach A then B because A is O(n^2).\n"
        "## Code\n"
        "```python\n"
        "x = 1\n"
        "```\n"
    )
    notes, code = parse_code_message(text)
    assert "approach A" in notes
    assert code == "x = 1"


def test_parse_case_insensitive_section_headers():
    text = (
        "## NOTES\n"
        "ok\n\n"
        "## code\n"
        "```python\n"
        "y = 2\n"
        "```\n"
    )
    notes, code = parse_code_message(text)
    assert notes == "ok"
    assert code == "y = 2"


def test_role_prompts_contain_schema_markers():
    for prompt in (PROPOSER_PROMPT, REVIEWER_PROMPT, INTEGRATOR_PROMPT):
        assert "## Notes" in prompt
        assert "## Code" in prompt


def test_role_prompts_dict_has_three_roles():
    assert set(CODE_ROLE_PROMPTS.keys()) == {"proposer", "reviewer", "integrator"}


def test_proposer_reviewer_integrator_have_distinct_intent():
    """Each role's prompt mentions a role-distinguishing keyword."""
    assert "proposer" in PROPOSER_PROMPT.lower() or "propose" in PROPOSER_PROMPT.lower()
    assert "reviewer" in REVIEWER_PROMPT.lower() or "critic" in REVIEWER_PROMPT.lower()
    assert "integrator" in INTEGRATOR_PROMPT.lower() or "synthes" in INTEGRATOR_PROMPT.lower()
