"""Prompt-variant registry: keys, content invariants, lookup behavior."""
from __future__ import annotations

import pytest

from agentdiet.agents import (
    FORMAT_INSTR,
    SKEPTIC_PROMPT,
    SOLVER_PROMPT,
    SYNTHESIZER_PROMPT,
)
from agentdiet.prompts import PROMPT_VARIANTS, get_variant


EXPECTED_VARIANTS = {"cooperative", "adversarial-strict", "symmetric"}


def test_registry_has_exactly_three_keys():
    assert set(PROMPT_VARIANTS.keys()) == EXPECTED_VARIANTS


def test_cooperative_is_byte_identical_to_existing_role_prompts():
    cooperative = PROMPT_VARIANTS["cooperative"]
    assert cooperative == [SOLVER_PROMPT, SKEPTIC_PROMPT, SYNTHESIZER_PROMPT]


def test_get_variant_returns_three_prompts_per_variant():
    for name in EXPECTED_VARIANTS:
        prompts = get_variant(name)
        assert len(prompts) == 3
        assert all(isinstance(p, str) and p for p in prompts)


def test_get_variant_returns_a_copy_not_the_internal_list():
    """Mutations on the returned list must not corrupt the registry."""
    a = get_variant("cooperative")
    a.clear()
    b = get_variant("cooperative")
    assert len(b) == 3


def test_adversarial_strict_skeptic_has_disagreement_obligation():
    prompts = get_variant("adversarial-strict")
    skeptic = prompts[1]
    assert "disagreement" in skeptic.lower()
    assert "wrong" in skeptic.lower()
    # Solver slot in adversarial-strict matches plain SOLVER_PROMPT
    assert prompts[0] == SOLVER_PROMPT


def test_adversarial_strict_synthesizer_enumerates_disagreements():
    synth = get_variant("adversarial-strict")[2]
    assert "enumerate" in synth.lower() or "list every disagreement" in synth.lower()


def test_symmetric_three_identical_strings():
    prompts = get_variant("symmetric")
    assert prompts[0] == prompts[1] == prompts[2]


def test_symmetric_uses_solver_prompt():
    prompts = get_variant("symmetric")
    assert prompts[0] == SOLVER_PROMPT


def test_get_variant_unknown_name_raises_with_message():
    with pytest.raises(KeyError, match="unknown prompt variant"):
        get_variant("totally-invalid-name")


def test_every_variant_carries_format_instr():
    """Parser stability: each prompt across every variant ends with the
    shared FORMAT_INSTR sentence so the parser keeps working."""
    for name in EXPECTED_VARIANTS:
        for p in get_variant(name):
            assert FORMAT_INSTR in p, (
                f"variant {name!r}: prompt missing FORMAT_INSTR"
            )
