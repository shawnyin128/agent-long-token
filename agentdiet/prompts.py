"""Prompt-variant registry for the prompt-robustness sub-grid (spec §4.5).

Three variants of the 3-agent debate role triplet:
  - cooperative:        re-export of the existing solver/skeptic/synthesizer
                        prompts (RQ0 baseline; main grid uses this).
  - adversarial-strict: same role triplet, with stricter obligations on
                        the skeptic (must produce a concrete disagreement)
                        and synthesizer (must enumerate disagreement points
                        before resolving).
  - symmetric:          three identical SOLVER_PROMPT instances, no role
                        differentiation. The "diversity comes only from
                        sampling noise" baseline.

Voting and SA always use SOLVER_PROMPT regardless of variant — the variant
only changes what the 3 debate agents see.
"""
from __future__ import annotations

from agentdiet.agents import (
    FORMAT_INSTR,
    SKEPTIC_PROMPT,
    SOLVER_PROMPT,
    SYNTHESIZER_PROMPT,
)


VariantName = str


# --- adversarial-strict ---------------------------------------------------


_ADVERSARIAL_SOLVER = SOLVER_PROMPT  # solver role unchanged

_ADVERSARIAL_SKEPTIC = (
    "You are a skeptical math reviewer. You MUST identify at least one "
    "concrete disagreement with the prior agents' work and explain why "
    "their step was wrong before proposing your own solution. Read the "
    "solutions from other agents carefully. Check each step for "
    "computational errors, missed conditions, or unit mistakes. Provide "
    "your own corrected solution with clear reasoning. " + FORMAT_INSTR + "."
)

_ADVERSARIAL_SYNTHESIZER = (
    "You are a synthesizer. Before resolving, list every disagreement "
    "point you observed between the prior agents and state which side "
    "you favor on each with a one-line reason. Then produce a final "
    "reasoned answer based on the strongest argument. " + FORMAT_INSTR + "."
)


# --- symmetric -------------------------------------------------------------


_SYMMETRIC_PROMPT = SOLVER_PROMPT  # all three agents share this


# --- registry --------------------------------------------------------------


PROMPT_VARIANTS: dict[VariantName, list[str]] = {
    "cooperative": [SOLVER_PROMPT, SKEPTIC_PROMPT, SYNTHESIZER_PROMPT],
    "adversarial-strict": [
        _ADVERSARIAL_SOLVER,
        _ADVERSARIAL_SKEPTIC,
        _ADVERSARIAL_SYNTHESIZER,
    ],
    "symmetric": [_SYMMETRIC_PROMPT, _SYMMETRIC_PROMPT, _SYMMETRIC_PROMPT],
}


def get_variant(name: VariantName) -> list[str]:
    """Return the three system prompts for the given variant.

    Raises KeyError with a helpful message listing valid variant names.
    """
    if name not in PROMPT_VARIANTS:
        raise KeyError(
            f"unknown prompt variant {name!r}; "
            f"choose one of {sorted(PROMPT_VARIANTS.keys())}"
        )
    return list(PROMPT_VARIANTS[name])
