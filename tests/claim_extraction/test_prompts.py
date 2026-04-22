from __future__ import annotations

import json

from agentdiet.extract_claims import (
    TYPE_DEFINITIONS,
    FEW_SHOT_EXAMPLES,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from agentdiet.types import CLAIM_TYPES


def test_type_definitions_cover_all_six():
    assert set(TYPE_DEFINITIONS.keys()) == set(CLAIM_TYPES)
    for k, v in TYPE_DEFINITIONS.items():
        assert isinstance(v, str) and len(v) > 5, f"definition too short for {k}"


def test_few_shot_examples_are_parseable_json():
    assert len(FEW_SHOT_EXAMPLES) == 3
    for ex in FEW_SHOT_EXAMPLES:
        assert "message_text" in ex
        assert "claims_json" in ex
        claims = json.loads(ex["claims_json"])
        assert isinstance(claims, list)
        for c in claims:
            assert c["type"] in CLAIM_TYPES
            assert "quote" in c
            assert "text" in c
            assert c["quote"] in ex["message_text"], \
                f"few-shot quote not found verbatim in example text: {c['quote']!r}"


def test_system_prompt_mentions_every_type():
    for t in CLAIM_TYPES:
        assert t in SYSTEM_PROMPT, f"{t} missing from SYSTEM_PROMPT"


def test_build_user_prompt_includes_message_text_and_context():
    prompt = build_user_prompt(
        question="What is 2+2?",
        message_text="I think the answer is 4.",
        agent_id=0,
        round=1,
    )
    assert "What is 2+2?" in prompt
    assert "I think the answer is 4." in prompt
    assert "agent 0" in prompt.lower() or "agent_id" in prompt.lower()
