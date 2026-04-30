"""End-to-end grid pipeline + resumability + bookkeeping integration."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from agentdiet.dataset import Question
from agentdiet.eval.base import CodeQuestion, TestCase
from agentdiet.eval.judges import SubprocessJudge
from agentdiet.grid.orchestrator import run_cell
from agentdiet.grid.types import CellSpec, cell_dir, load_record, load_summary
from agentdiet.llm_client import DummyBackend, LLMClient


def _math_responder(messages, model, temperature, *, thinking=False, **kw):
    text = messages[-1]["content"]
    m = re.search(r"item (\d+)", text)
    if m:
        return f"#### {m.group(1)}"
    m = re.search(r"#### (\d+)", text)
    if m:
        return f"#### {m.group(1)}"
    return "#### 0"


def _make_math_questions(n: int) -> list[Question]:
    return [
        Question(qid=f"q{i}", question=f"Compute item {i}", gold_answer=str(i))
        for i in range(n)
    ]


def _math_cell(thinking: bool = False) -> CellSpec:
    return CellSpec(
        model="Qwen/Qwen3-30B-A3B", model_family="qwen3",
        dataset_name="gsm8k", thinking=thinking,
    )


def test_e2e_math_cell_writes_four_artifacts(tmp_path):
    backend = DummyBackend(_math_responder)
    client = LLMClient(backend, cache_path=tmp_path / "cache.jsonl")
    cell = _math_cell()
    output_dir = tmp_path / "out"

    run_cell(
        cell=cell, llm_client=client,
        questions=_make_math_questions(5),
        output_dir=output_dir, calibration_prefix=3,
    )

    cdir = output_dir / cell_dir(cell)
    artifacts = sorted(p.name for p in cdir.iterdir())
    assert artifacts == [
        "debate.json", "sa.json", "sc_calibration.json",
        "summary.json", "voting.json",
    ]


def test_e2e_resume_no_extra_backend_calls(tmp_path):
    backend = DummyBackend(_math_responder)
    client = LLMClient(backend, cache_path=tmp_path / "cache.jsonl")
    cell = _math_cell()
    output_dir = tmp_path / "out"
    qs = _make_math_questions(5)

    run_cell(cell=cell, llm_client=client, questions=qs,
             output_dir=output_dir, calibration_prefix=3)
    first_calls = backend.call_count

    # Re-run without force: artifacts already exist; backend not touched
    run_cell(cell=cell, llm_client=client, questions=qs,
             output_dir=output_dir, calibration_prefix=3)
    assert backend.call_count == first_calls


def test_e2e_token_totals_match_aggregation(tmp_path):
    backend = DummyBackend(_math_responder)
    client = LLMClient(backend, cache_path=tmp_path / "cache.jsonl")
    cell = _math_cell()
    output_dir = tmp_path / "out"

    summary = run_cell(
        cell=cell, llm_client=client,
        questions=_make_math_questions(5),
        output_dir=output_dir, calibration_prefix=3,
    )

    sa = load_record(output_dir / cell_dir(cell) / "sa.json")
    voting = load_record(output_dir / cell_dir(cell) / "voting.json")
    debate = load_record(output_dir / cell_dir(cell) / "debate.json")

    assert summary.sa_total_tokens == sa.total_tokens
    assert summary.voting_total_tokens == voting.total_tokens
    assert summary.debate_total_tokens == debate.total_tokens
    # Per-condition: sum of per-question total_tokens equals record total
    for record in (sa, voting, debate):
        per_q_sum = sum(q.total_tokens for q in record.questions)
        assert per_q_sum == record.total_tokens


def test_e2e_cell_dir_is_deterministic(tmp_path):
    backend = DummyBackend(_math_responder)
    client = LLMClient(backend, cache_path=tmp_path / "cache.jsonl")
    cell_a = _math_cell(thinking=False)
    cell_b = _math_cell(thinking=False)

    out = tmp_path / "out"
    run_cell(cell=cell_a, llm_client=client, questions=_make_math_questions(3),
             output_dir=out, calibration_prefix=2)

    expected = out / cell_dir(cell_b)
    assert expected.is_dir()
    assert (expected / "summary.json").is_file()


def _code_responder(code: str):
    def responder(messages, model, temperature, *, thinking=False, **kw):
        return (
            "## Notes\nstraightforward.\n\n"
            "## Code\n```python\n" + code + "\n```\n"
        )
    return responder


@pytest.mark.timeout(120)
def test_e2e_code_cell_summary_realistic(tmp_path):
    backend = DummyBackend(_code_responder("def add(a, b): return a + b"))
    client = LLMClient(backend, cache_path=tmp_path / "cache.jsonl")
    judge = SubprocessJudge()
    cell = CellSpec(
        model="Qwen/Qwen3-30B-A3B", model_family="qwen3",
        dataset_name="humaneval_plus", thinking=False,
    )
    questions = [
        CodeQuestion(
            qid=f"add{i}",
            prompt=("def add(a, b):\n    '''Return a + b.\n"
                    "    >>> add(1, 2)\n    3\n    '''"),
            entry_point="add",
            public_tests=[TestCase(name="ex", script="assert add(1, 2) == 3")],
            hidden_tests=[
                TestCase(name="h1", script="assert add(2, 3) == 5"),
                TestCase(name="h2", script="assert add(0, 0) == 0"),
            ],
        )
        for i in range(3)
    ]

    summary = run_cell(
        cell=cell, llm_client=client, questions=questions,
        output_dir=tmp_path / "out", calibration_prefix=2, judge=judge,
    )
    assert summary.sa_accuracy == 1.0
    assert summary.voting_accuracy == 1.0
    assert summary.debate_accuracy == 1.0
    assert summary.delta_debate_voting == 0.0
    assert summary.delta_debate_sa == 0.0
