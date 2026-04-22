"""Agent prompts must not teach the model a literal 'N' placeholder."""
from __future__ import annotations

import pytest

from agentdiet.agents import (
    SOLVER_PROMPT, SKEPTIC_PROMPT, SYNTHESIZER_PROMPT, SYSTEM_PROMPTS,
)


@pytest.mark.parametrize("prompt", [SOLVER_PROMPT, SKEPTIC_PROMPT, SYNTHESIZER_PROMPT])
def test_prompt_no_longer_uses_literal_N(prompt):
    # This is the literal the model was learning before the fix.
    assert "'#### N'" not in prompt
    assert "#### N " not in prompt
    # Defensive: also reject '#### N\n' or '#### N"' shapes.
    import re
    assert not re.search(r"####\s*N\b", prompt), \
        f"prompt still contains literal '#### N' — model will copy it"


@pytest.mark.parametrize("prompt", [SOLVER_PROMPT, SKEPTIC_PROMPT, SYNTHESIZER_PROMPT])
def test_prompt_teaches_hash_format(prompt):
    assert "####" in prompt
    # Example should be a concrete numeric template.
    assert "42" in prompt or "<answer>" in prompt or "final numeric" in prompt


def test_all_three_roles_use_same_format_instruction():
    # The format-instruction sentence should be identical across roles so
    # there is one canonical spec to parse against.
    def _extract_hash_sentence(p: str) -> str:
        for line in p.split(". "):
            if "####" in line:
                return line.strip()
        return ""
    s_solver = _extract_hash_sentence(SOLVER_PROMPT)
    s_skeptic = _extract_hash_sentence(SKEPTIC_PROMPT)
    s_synth = _extract_hash_sentence(SYNTHESIZER_PROMPT)
    assert s_solver and s_skeptic and s_synth
    assert s_solver == s_skeptic == s_synth, \
        "format instruction drifted across roles"


def test_system_prompts_dict_includes_all_three():
    assert set(SYSTEM_PROMPTS.keys()) == {"solver", "skeptic", "synthesizer"}
