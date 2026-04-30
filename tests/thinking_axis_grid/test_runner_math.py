"""Per-condition runners on math cells (synthetic backends)."""
from __future__ import annotations

from agentdiet.dataset import Question
from agentdiet.grid.runner import (
    aggregate_condition,
    default_sa_system_prompt,
    run_debate_q_math,
    run_sa_math,
    run_voting_q_math,
)
from agentdiet.grid.types import CellSpec, ConditionRecord
from agentdiet.llm_client import ChatResult, DummyBackend, LLMClient


def _cell(thinking: bool = False) -> CellSpec:
    return CellSpec(
        model="Qwen/Qwen3-30B-A3B", model_family="qwen3",
        dataset_name="gsm8k", thinking=thinking,
    )


def _q(qid: str, gold: str = "42") -> Question:
    return Question(qid=qid, question=f"Solve: {qid}", gold_answer=gold)


def test_run_sa_math_correct(tmp_path):
    backend = DummyBackend(lambda m, mo, t, **kw: "step by step\n#### 42")
    client = LLMClient(backend, cache_path=tmp_path / "c.jsonl")
    res = run_sa_math(_q("q1", gold="42"), _cell(), client)
    assert res.correct is True
    assert res.final_answer == "42"
    assert res.qid == "q1"


def test_run_sa_math_wrong(tmp_path):
    backend = DummyBackend(lambda m, mo, t, **kw: "guess\n#### 7")
    client = LLMClient(backend, cache_path=tmp_path / "c.jsonl")
    res = run_sa_math(_q("q1", gold="42"), _cell(), client)
    assert res.correct is False
    assert res.final_answer == "7"


def test_run_sa_math_thinking_propagates(tmp_path):
    seen = []

    def responder(messages, model, temperature, *, thinking=False, **kw):
        seen.append(thinking)
        return "#### 1"

    backend = DummyBackend(responder)
    client = LLMClient(backend, cache_path=tmp_path / "c.jsonl")
    run_sa_math(_q("q1", gold="1"), _cell(thinking=True), client)
    assert seen == [True]


def test_run_voting_q_math_majority(tmp_path):
    answers = ["7", "5", "7", "9", "7"]

    def responder(msgs, mo, t, **kw):
        user = msgs[-1]["content"]
        k = int(user.split("\n", 1)[0].split(":")[1].strip())
        return f"#### {answers[k]}"

    backend = DummyBackend(responder)
    client = LLMClient(backend, cache_path=tmp_path / "c.jsonl")
    res = run_voting_q_math(_q("q1", gold="7"), _cell(), client, n_samples=5)
    assert res.correct is True
    assert res.final_answer == "7"


def test_run_debate_q_math_total_tokens_summed(tmp_path):
    """Debate sums prompt+completion across all 9 calls."""

    class UsageBackend:
        def chat_full(self, messages, model, temperature, **kw):
            return ChatResult(
                response="reasoning\n#### 42",
                prompt_tokens=10, completion_tokens=5,
            )

    client = LLMClient(UsageBackend(), cache_path=tmp_path / "c.jsonl")
    res = run_debate_q_math(_q("q1", gold="42"), _cell(), client)
    # 9 calls × (10 + 5) = 135
    assert res.total_tokens == 135
    assert res.prompt_tokens == 90
    assert res.completion_tokens == 45
    assert res.correct is True


def test_run_debate_q_math_thinking_propagates(tmp_path):
    seen = []

    def responder(messages, model, temperature, *, thinking=False, **kw):
        seen.append(thinking)
        return "#### 9"

    backend = DummyBackend(responder)
    client = LLMClient(backend, cache_path=tmp_path / "c.jsonl")
    run_debate_q_math(_q("q1", gold="9"), _cell(thinking=True), client)
    # 9 calls — all should have thinking=True
    assert seen == [True] * 9


def test_default_sa_system_prompt_math_is_solver_prompt():
    from agentdiet.agents import SOLVER_PROMPT
    assert default_sa_system_prompt("math") == SOLVER_PROMPT


def test_aggregate_condition_computes_accuracy():
    cell = _cell()
    from agentdiet.grid.types import QuestionResult
    questions = [
        QuestionResult(qid=f"q{i}", gold="1", final_answer="1" if i < 3 else "2",
                       correct=(i < 3), prompt_tokens=10, completion_tokens=5,
                       total_tokens=15)
        for i in range(5)
    ]
    record = aggregate_condition(questions, cell, condition="sa")
    assert record.n_evaluated == 5
    assert record.accuracy == 0.6
    assert record.total_tokens == 75
    assert record.condition == "sa"
