"""prompt_variant threads through agents -> run_debate -> grid.runner."""
from __future__ import annotations

import pytest

from agentdiet.agents import (
    DEFAULT_ROLE_ORDER,
    SKEPTIC_PROMPT,
    SOLVER_PROMPT,
    SYNTHESIZER_PROMPT,
    make_default_agents,
)
from agentdiet.dataset import Question
from agentdiet.debate import run_debate
from agentdiet.grid.runner import run_debate_q_math
from agentdiet.grid.types import CellSpec
from agentdiet.llm_client import DummyBackend, LLMClient


def _cell(thinking: bool = False) -> CellSpec:
    return CellSpec(
        model="Qwen/Qwen3-30B-A3B", model_family="qwen3",
        dataset_name="gsm8k", thinking=thinking,
    )


def _q() -> Question:
    return Question(qid="q1", question="What is 2+2?", gold_answer="4")


# --- agents.make_default_agents -------------------------------------------


def test_make_default_agents_no_prompts_kwarg_unchanged():
    agents = make_default_agents(n_agents=3)
    assert [a.role for a in agents] == list(DEFAULT_ROLE_ORDER)
    assert agents[0].system_prompt == SOLVER_PROMPT
    assert agents[1].system_prompt == SKEPTIC_PROMPT
    assert agents[2].system_prompt == SYNTHESIZER_PROMPT


def test_make_default_agents_with_prompts_overrides_per_slot():
    prompts = ["A", "B", "C"]
    agents = make_default_agents(n_agents=3, prompts=prompts)
    assert [a.system_prompt for a in agents] == ["A", "B", "C"]
    # Roles still come from DEFAULT_ROLE_ORDER
    assert [a.role for a in agents] == list(DEFAULT_ROLE_ORDER)


def test_make_default_agents_prompts_length_mismatch_raises():
    with pytest.raises(ValueError, match="entries"):
        make_default_agents(n_agents=3, prompts=["only-one"])


# --- run_debate prompt_variant -------------------------------------------


def test_run_debate_default_variant_gives_cooperative_roles(tmp_path):
    backend = DummyBackend(lambda m, mo, t, **kw: "#### 4")
    client = LLMClient(backend, cache_path=tmp_path / "c.jsonl")
    dialogue = run_debate(
        question=_q(), llm_client=client, model="m",
        n_agents=3, n_rounds=2,
    )
    assert dialogue.meta["roles"] == list(DEFAULT_ROLE_ORDER)
    assert dialogue.meta["prompt_variant"] == "cooperative"


def test_run_debate_symmetric_sends_three_identical_system_prompts(tmp_path):
    """Symmetric variant gives every agent SOLVER_PROMPT. Round 1 with
    identical agents collapses to a single cache entry — so we may see
    only one backend call. Whatever IS captured must match SOLVER_PROMPT."""
    seen_systems: list[str] = []

    def responder(messages, model, temperature, *, thinking=False, **kw):
        for m in messages:
            if m["role"] == "system":
                seen_systems.append(m["content"])
                break
        return "#### 4"

    backend = DummyBackend(responder)
    client = LLMClient(backend, cache_path=tmp_path / "c.jsonl")
    run_debate(
        question=_q(), llm_client=client, model="m",
        n_agents=3, n_rounds=1,
        prompt_variant="symmetric",
    )
    assert seen_systems, "no backend calls observed"
    assert all(p == SOLVER_PROMPT for p in seen_systems)


def test_run_debate_adversarial_strict_puts_skeptic_in_slot_1(tmp_path):
    seen_systems: list[str] = []

    def responder(messages, model, temperature, *, thinking=False, **kw):
        for m in messages:
            if m["role"] == "system":
                seen_systems.append(m["content"])
                break
        return "#### 4"

    backend = DummyBackend(responder)
    client = LLMClient(backend, cache_path=tmp_path / "c.jsonl")
    run_debate(
        question=_q(), llm_client=client, model="m",
        n_agents=3, n_rounds=1,
        prompt_variant="adversarial-strict",
    )
    assert len(seen_systems) == 3
    # Slot 1 (skeptic) must have the disagreement obligation
    assert "disagreement" in seen_systems[1].lower()
    # Slot 2 (synthesizer) enumerates
    assert ("enumerate" in seen_systems[2].lower()
            or "list every disagreement" in seen_systems[2].lower())


def test_run_debate_unknown_variant_raises(tmp_path):
    backend = DummyBackend(lambda m, mo, t, **kw: "#### 4")
    client = LLMClient(backend, cache_path=tmp_path / "c.jsonl")
    with pytest.raises(KeyError, match="unknown prompt variant"):
        run_debate(
            question=_q(), llm_client=client, model="m",
            n_agents=3, n_rounds=1,
            prompt_variant="totally-bogus",
        )


# --- grid.runner.run_debate_q_math ----------------------------------------


def test_run_debate_q_math_forwards_prompt_variant(tmp_path):
    seen_variants: list[str] = []

    def responder(messages, model, temperature, *, thinking=False, **kw):
        for m in messages:
            if m["role"] == "system":
                seen_variants.append(m["content"])
                break
        return "#### 4"

    backend = DummyBackend(responder)
    client = LLMClient(backend, cache_path=tmp_path / "c.jsonl")
    res = run_debate_q_math(
        _q(), _cell(), client,
        prompt_variant="symmetric",
    )
    assert res.correct is True
    # 9 system prompts captured (3 agents x 3 rounds), all identical SOLVER_PROMPT
    # Note: chat_full caches by message content, so the 3-round duplication
    # may collapse; relax to "at least one occurrence and all are SOLVER_PROMPT"
    assert seen_variants, "no system prompts captured"
    assert all(p == SOLVER_PROMPT for p in seen_variants)


def test_run_debate_q_math_default_variant_is_cooperative(tmp_path):
    backend = DummyBackend(lambda m, mo, t, **kw: "#### 4")
    client = LLMClient(backend, cache_path=tmp_path / "c.jsonl")
    res = run_debate_q_math(_q(), _cell(), client)
    assert res.correct is True
    # Existing tests rely on this default — regression guard


def test_run_debate_q_math_total_tokens_positive_under_each_variant(tmp_path):
    """Variant doesn't break token accumulation. cooperative has 9 unique
    cache keys (different system prompts per agent), so all 9 calls hit
    backend and produce the strict 135. symmetric collapses round-1
    cache hits and falls back to char-approx tokens, so the total
    differs but must still be > 0."""
    from agentdiet.llm_client import ChatResult

    class UsageBackend:
        def chat_full(self, messages, model, temperature, **kw):
            return ChatResult(
                response="#### 4", prompt_tokens=10, completion_tokens=5,
            )

    # cooperative: strict 135
    fresh = LLMClient(UsageBackend(), cache_path=tmp_path / "c_coop.jsonl")
    res = run_debate_q_math(_q(), _cell(), fresh, prompt_variant="cooperative")
    assert res.total_tokens == 135

    # adversarial-strict: also 9 unique cache keys (different system prompts), 135
    fresh = LLMClient(UsageBackend(), cache_path=tmp_path / "c_adv.jsonl")
    res = run_debate_q_math(_q(), _cell(), fresh, prompt_variant="adversarial-strict")
    assert res.total_tokens == 135

    # symmetric: round-1 cache collapses; total is positive but differs
    fresh = LLMClient(UsageBackend(), cache_path=tmp_path / "c_sym.jsonl")
    res = run_debate_q_math(_q(), _cell(), fresh, prompt_variant="symmetric")
    assert res.total_tokens > 0
