"""Per-condition runners on code cells (DummyBackend + SubprocessJudge)."""
from __future__ import annotations

import pytest

from agentdiet.eval.base import CodeQuestion, TestCase
from agentdiet.eval.judges import SubprocessJudge
from agentdiet.grid.runner import (
    _extract_code,
    default_sa_system_prompt,
    run_debate_q_code,
    run_sa_code,
    run_voting_q_code,
)
from agentdiet.grid.types import CellSpec
from agentdiet.llm_client import DummyBackend, LLMClient


def _code_cell(thinking: bool = False) -> CellSpec:
    return CellSpec(
        model="Qwen/Qwen3-30B-A3B", model_family="qwen3",
        dataset_name="humaneval_plus", thinking=thinking,
    )


def _add_question() -> CodeQuestion:
    return CodeQuestion(
        qid="add",
        prompt="def add(a, b):\n    '''Return a + b.\n    >>> add(1, 2)\n    3\n    '''",
        entry_point="add",
        public_tests=[TestCase(name="ex1", script="assert add(1, 2) == 3")],
        hidden_tests=[
            TestCase(name="h1", script="assert add(2, 3) == 5"),
            TestCase(name="h2", script="assert add(0, 0) == 0"),
        ],
    )


def _well_formed(code: str) -> str:
    return (
        "## Notes\nthinking through.\n\n"
        "## Code\n```python\n" + code + "\n```\n"
    )


def test_extract_code_from_notes_code_schema():
    text = _well_formed("def f(): return 1")
    assert _extract_code(text) == "def f(): return 1"


def test_extract_code_from_bare_python_fence():
    text = "Here is my answer:\n```python\ndef f(): pass\n```\nThanks."
    assert _extract_code(text) == "def f(): pass"


def test_extract_code_returns_whole_text_if_no_fence():
    text = "def f(): return 99"
    assert _extract_code(text) == "def f(): return 99"


def test_run_sa_code_correct(tmp_path):
    backend = DummyBackend(
        lambda m, mo, t, **kw: _well_formed("def add(a, b): return a + b")
    )
    client = LLMClient(backend, cache_path=tmp_path / "c.jsonl")
    judge = SubprocessJudge()
    res = run_sa_code(_add_question(), _code_cell(), client, judge)
    assert res.correct is True
    assert res.qid == "add"
    # final_answer is the code
    assert "return a + b" in res.final_answer


def test_run_sa_code_wrong_solution(tmp_path):
    backend = DummyBackend(
        lambda m, mo, t, **kw: _well_formed("def add(a, b): return a - b")
    )
    client = LLMClient(backend, cache_path=tmp_path / "c.jsonl")
    judge = SubprocessJudge()
    res = run_sa_code(_add_question(), _code_cell(), client, judge)
    assert res.correct is False


def test_run_voting_q_code_clusters_and_picks_correct(tmp_path):
    """5 samples: 3 correct, 2 wrong. Public-test cluster picks the
    correct cluster; hidden-test judging then confirms correct."""
    answers_by_k = {
        0: "def add(a, b): return a + b",
        1: "def add(a, b): return a - b",   # wrong
        2: "def add(a, b): return a + b",
        3: "def add(a, b): return a + b",
        4: "def add(a, b): return a * b",   # wrong
    }

    def responder(msgs, mo, t, **kw):
        user = msgs[-1]["content"]
        k = int(user.split("\n", 1)[0].split(":")[1].strip())
        return _well_formed(answers_by_k[k])

    backend = DummyBackend(responder)
    client = LLMClient(backend, cache_path=tmp_path / "c.jsonl")
    judge = SubprocessJudge()
    res = run_voting_q_code(_add_question(), _code_cell(), client,
                            n_samples=5, judge=judge)
    assert res.correct is True
    assert "return a + b" in res.final_answer


@pytest.mark.timeout(60)
def test_run_debate_q_code_clusters_round_3(tmp_path):
    """All 3 round-3 codes correct -> cluster size 3 -> representative correct."""
    backend = DummyBackend(
        lambda m, mo, t, **kw: _well_formed("def add(a, b): return a + b")
    )
    client = LLMClient(backend, cache_path=tmp_path / "c.jsonl")
    judge = SubprocessJudge()
    res = run_debate_q_code(_add_question(), _code_cell(), client, judge)
    assert res.correct is True


def test_default_sa_system_prompt_code_has_schema_markers():
    p = default_sa_system_prompt("code")
    assert "## Notes" in p
    assert "## Code" in p


def test_default_sa_system_prompt_unknown_domain_raises():
    with pytest.raises(ValueError, match="unknown domain"):
        default_sa_system_prompt("invalid")  # type: ignore[arg-type]
