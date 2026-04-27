"""run_code_debate end-to-end on DummyBackend."""
from __future__ import annotations

from pathlib import Path

import pytest

from agentdiet.debate.code_protocol import (
    DEFAULT_CODE_ROLE_ORDER,
    CodeAgent,
    make_default_code_agents,
    run_code_debate,
)
from agentdiet.eval.base import CodeQuestion, TestCase
from agentdiet.llm_client import DummyBackend, LLMClient


def _question() -> CodeQuestion:
    return CodeQuestion(
        qid="q1",
        prompt="Write add(a, b) returning a+b.",
        entry_point="add",
        public_tests=[TestCase(name="t", script="assert add(1,2) == 3")],
        hidden_tests=[],
    )


def _well_formed(answer_code: str) -> str:
    return (
        "## Notes\nApproach noted.\n\n"
        "## Code\n"
        "```python\n"
        f"{answer_code}\n"
        "```\n"
    )


def test_dialogue_has_n_agents_times_n_rounds_messages(tmp_path):
    backend = DummyBackend(
        lambda msgs, m, t, **kw: _well_formed("def add(a,b): return a+b")
    )
    client = LLMClient(backend, cache_path=tmp_path / "c.jsonl")
    dialogue = run_code_debate(
        question=_question(),
        llm_client=client,
        model="m",
        n_agents=3,
        n_rounds=3,
    )
    assert len(dialogue.messages) == 9
    rounds = sorted({m.round for m in dialogue.messages})
    assert rounds == [1, 2, 3]
    agents = sorted({m.agent_id for m in dialogue.messages})
    assert agents == [0, 1, 2]


def test_role_assignment_follows_default_order(tmp_path):
    backend = DummyBackend(
        lambda msgs, m, t, **kw: _well_formed("pass")
    )
    client = LLMClient(backend, cache_path=tmp_path / "c.jsonl")
    dialogue = run_code_debate(
        question=_question(),
        llm_client=client,
        model="m",
    )
    by_id = {}
    for m in dialogue.messages:
        by_id.setdefault(m.agent_id, set()).add(m.role)
    assert by_id[0] == {"proposer"}
    assert by_id[1] == {"reviewer"}
    assert by_id[2] == {"integrator"}


def test_round1_messages_dont_include_other_agent_outputs(tmp_path):
    """Round 1's user message should NOT include peer outputs (none yet)."""
    captured_user_msgs: list[str] = []

    def responder(msgs, model, temp, **kw):
        for m in msgs:
            if m["role"] == "user":
                captured_user_msgs.append(m["content"])
        return _well_formed("pass")

    backend = DummyBackend(responder)
    client = LLMClient(backend, cache_path=tmp_path / "c.jsonl")
    run_code_debate(
        question=_question(),
        llm_client=client, model="m",
        n_agents=3, n_rounds=1,
    )
    # First round: each user message contains only the prompt
    for um in captured_user_msgs:
        assert "Other agents' previous-round outputs" not in um
        assert "Solve the following coding problem" in um


def test_round2_user_messages_contain_peer_outputs(tmp_path):
    captured: list[str] = []

    def responder(msgs, model, temp, **kw):
        for m in msgs:
            if m["role"] == "user":
                captured.append(m["content"])
        return _well_formed("def add(a,b): return a+b")

    backend = DummyBackend(responder)
    client = LLMClient(backend, cache_path=tmp_path / "c.jsonl")
    dialogue = run_code_debate(
        question=_question(),
        llm_client=client, model="m",
        n_agents=3, n_rounds=2,
    )
    # Round 2 user prompts (last 3 captured) must reference others
    round2_prompts = captured[-3:]
    for prompt in round2_prompts:
        assert "Other agents' previous-round outputs" in prompt
        assert "## Code" in prompt  # peer's code blocks rendered
    # Round 2 captures 3 prompts; each agent should NOT see itself
    # — verify by counting how many "Agent N" tokens appear per prompt
    for i, prompt in enumerate(round2_prompts):
        assert f"Agent {i} (" not in prompt


def test_dialogue_meta_token_totals_present(tmp_path):
    """Token totals are aggregated when backend reports usage."""
    from agentdiet.llm_client import ChatResult

    class UsageBackend:
        def chat_full(self, messages, model, temperature, **kw):
            return ChatResult(
                response=_well_formed("pass"),
                prompt_tokens=20,
                completion_tokens=10,
            )

    client = LLMClient(UsageBackend(), cache_path=tmp_path / "c.jsonl")
    dialogue = run_code_debate(
        question=_question(),
        llm_client=client, model="m",
        n_agents=3, n_rounds=3,
    )
    assert dialogue.meta["total_prompt_tokens"] == 9 * 20
    assert dialogue.meta["total_completion_tokens"] == 9 * 10


def test_make_default_code_agents_too_many_raises():
    with pytest.raises(ValueError):
        make_default_code_agents(n_agents=5)


def test_custom_agents_must_match_n_agents(tmp_path):
    backend = DummyBackend(lambda m, mo, t, **kw: _well_formed("x = 1"))
    client = LLMClient(backend, cache_path=tmp_path / "c.jsonl")
    bad_agents = [CodeAgent.make(0, "proposer")]
    with pytest.raises(ValueError):
        run_code_debate(
            question=_question(),
            llm_client=client, model="m",
            n_agents=3, agents=bad_agents,
        )


def test_dialogue_carries_thinking_meta(tmp_path):
    backend = DummyBackend(lambda m, mo, t, **kw: _well_formed("pass"))
    client = LLMClient(backend, cache_path=tmp_path / "c.jsonl")
    dialogue = run_code_debate(
        question=_question(),
        llm_client=client, model="m",
        thinking=True,
    )
    assert dialogue.meta["thinking"] is True
