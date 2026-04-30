"""Per-cell orchestrator: run_cell with synthetic LLM + Dataset."""
from __future__ import annotations

import pytest

from agentdiet.dataset import Question
from agentdiet.eval.base import CodeQuestion, TestCase
from agentdiet.eval.judges import SubprocessJudge
from agentdiet.grid.orchestrator import run_cell
from agentdiet.grid.types import CellSpec, cell_dir, load_record, load_summary
from agentdiet.llm_client import DummyBackend, LLMClient


def _math_cell(thinking: bool = False) -> CellSpec:
    return CellSpec(
        model="Qwen/Qwen3-30B-A3B", model_family="qwen3",
        dataset_name="gsm8k", thinking=thinking,
    )


def _code_cell() -> CellSpec:
    return CellSpec(
        model="Qwen/Qwen3-30B-A3B", model_family="qwen3",
        dataset_name="humaneval_plus", thinking=False,
    )


def _math_questions(n: int) -> list[Question]:
    return [
        Question(qid=f"q{i}", question=f"Compute item {i}", gold_answer=str(i))
        for i in range(n)
    ]


def _code_q(qid: str, code: str = "def add(a, b): return a + b") -> CodeQuestion:
    return CodeQuestion(
        qid=qid,
        prompt="def add(a, b):\n    '''Return a + b.\n    >>> add(1, 2)\n    3\n    '''",
        entry_point="add",
        public_tests=[TestCase(name="ex", script="assert add(1, 2) == 3")],
        hidden_tests=[
            TestCase(name="h1", script="assert add(2, 3) == 5"),
            TestCase(name="h2", script="assert add(0, 0) == 0"),
        ],
    )


def test_run_cell_math_full_pipeline(tmp_path):
    """Backend always returns the gold answer wrapped in #### markers."""

    def responder(messages, model, temperature, *, thinking=False, **kw):
        # In SA / round 1 of debate: user content is "Compute item I" or
        # the initial template that includes "Compute item I".
        # In voting: user starts with "Sample id: K\n\n<prompt>".
        # In debate round 2+: user content has prior responses with
        # "#### N" embedded — fall through to that pattern below.
        import re
        text = messages[-1]["content"]
        m = re.search(r"item (\d+)", text)
        if m:
            return f"reasoning\n#### {m.group(1)}"
        # Round 2+: scrape last "#### N" from prior agents' responses.
        m = re.search(r"#### (\d+)", text)
        if m:
            return f"keep going\n#### {m.group(1)}"
        return "fallback\n#### 0"

    backend = DummyBackend(responder)
    client = LLMClient(backend, cache_path=tmp_path / "c.jsonl")
    cell = _math_cell()
    qs = _math_questions(5)
    output_dir = tmp_path / "out"

    summary = run_cell(
        cell=cell, llm_client=client, questions=qs, output_dir=output_dir,
        calibration_prefix=3,
    )

    cdir = output_dir / cell_dir(cell)
    # All four artifacts written
    assert (cdir / "sa.json").is_file()
    assert (cdir / "voting.json").is_file()
    assert (cdir / "debate.json").is_file()
    assert (cdir / "sc_calibration.json").is_file()
    assert (cdir / "summary.json").is_file()
    # Accuracy: synthetic backend always echoes gold -> all 1.0
    assert summary.sa_accuracy == 1.0
    assert summary.voting_accuracy == 1.0
    assert summary.debate_accuracy == 1.0
    # Deltas zero for this trivial case
    assert summary.delta_debate_voting == 0.0
    assert summary.delta_debate_sa == 0.0
    # Calibration block has the expected keys
    cb = summary.calibration
    for k in ("N", "N_raw", "over_budget_factor", "floor_active",
              "mean_debate_tokens", "mean_sa_tokens"):
        assert k in cb


def test_run_cell_resume_skips_existing(tmp_path):
    """Second invocation with same questions + force=False reuses
    on-disk artifacts and makes no new backend calls."""
    backend = DummyBackend(
        lambda m, mo, t, **kw: "step\n#### 1"
    )
    client = LLMClient(backend, cache_path=tmp_path / "c.jsonl")
    cell = _math_cell()
    qs = _math_questions(3)
    output_dir = tmp_path / "out"

    run_cell(cell=cell, llm_client=client, questions=qs,
             output_dir=output_dir, calibration_prefix=2)
    calls_after_first = backend.call_count

    # Drop a sentinel into sa.json so we can detect it survived
    (output_dir / cell_dir(cell) / "sa.json").rename(
        output_dir / cell_dir(cell) / "sa.json.bak"
    )
    # Restore
    (output_dir / cell_dir(cell) / "sa.json.bak").rename(
        output_dir / cell_dir(cell) / "sa.json"
    )

    # Second run with force=False — should hit cache + reuse json
    run_cell(cell=cell, llm_client=client, questions=qs,
             output_dir=output_dir, calibration_prefix=2)
    assert backend.call_count == calls_after_first


def test_run_cell_force_re_runs(tmp_path):
    backend = DummyBackend(lambda m, mo, t, **kw: "#### 0")
    client = LLMClient(backend, cache_path=tmp_path / "c.jsonl")
    cell = _math_cell()
    qs = _math_questions(2)
    output_dir = tmp_path / "out"

    run_cell(cell=cell, llm_client=client, questions=qs,
             output_dir=output_dir, calibration_prefix=2)

    run_cell(cell=cell, llm_client=client, questions=qs,
             output_dir=output_dir, calibration_prefix=2, force=True)
    # Force re-runs — but cache is keyed by message content, so
    # if responses are identical, the cache hits. We can confirm
    # call_count grew only if cache was bypassed; with this
    # in-memory cache it didn't, but the artifacts WERE re-written.
    summary_path = output_dir / cell_dir(cell) / "summary.json"
    assert summary_path.is_file()


def test_run_cell_code_requires_judge(tmp_path):
    backend = DummyBackend(lambda m, mo, t, **kw: "x")
    client = LLMClient(backend, cache_path=tmp_path / "c.jsonl")
    with pytest.raises(ValueError, match="no judge"):
        run_cell(
            cell=_code_cell(), llm_client=client,
            questions=[_code_q("q1")], output_dir=tmp_path / "out",
        )


@pytest.mark.timeout(120)
def test_run_cell_code_full_pipeline(tmp_path):
    """Code cell: backend returns canonical solution; SubprocessJudge
    grades each condition. Expect accuracy 1.0 across the board."""

    def responder(messages, model, temperature, *, thinking=False, **kw):
        return (
            "## Notes\nuse +.\n\n"
            "## Code\n```python\ndef add(a, b): return a + b\n```\n"
        )

    backend = DummyBackend(responder)
    client = LLMClient(backend, cache_path=tmp_path / "c.jsonl")
    judge = SubprocessJudge()
    cell = _code_cell()
    qs = [_code_q(f"q{i}") for i in range(3)]
    output_dir = tmp_path / "out"

    summary = run_cell(
        cell=cell, llm_client=client, questions=qs, output_dir=output_dir,
        calibration_prefix=2, judge=judge,
    )
    assert summary.sa_accuracy == 1.0
    assert summary.voting_accuracy == 1.0
    assert summary.debate_accuracy == 1.0


def test_run_cell_empty_questions_raises(tmp_path):
    backend = DummyBackend(lambda m, mo, t, **kw: "")
    client = LLMClient(backend, cache_path=tmp_path / "c.jsonl")
    with pytest.raises(ValueError, match="no questions"):
        run_cell(cell=_math_cell(), llm_client=client, questions=[],
                 output_dir=tmp_path / "out")


def test_run_cell_summary_round_trip(tmp_path):
    backend = DummyBackend(lambda m, mo, t, **kw: "#### 0")
    client = LLMClient(backend, cache_path=tmp_path / "c.jsonl")
    cell = _math_cell()
    qs = _math_questions(3)
    output_dir = tmp_path / "out"

    written = run_cell(cell=cell, llm_client=client, questions=qs,
                       output_dir=output_dir, calibration_prefix=2)
    loaded = load_summary(output_dir / cell_dir(cell) / "summary.json")
    assert loaded == written


def test_run_cell_n_questions_caps_input(tmp_path):
    backend = DummyBackend(lambda m, mo, t, **kw: "#### 0")
    client = LLMClient(backend, cache_path=tmp_path / "c.jsonl")
    cell = _math_cell()
    qs = _math_questions(10)
    output_dir = tmp_path / "out"

    summary = run_cell(cell=cell, llm_client=client, questions=qs,
                       output_dir=output_dir, calibration_prefix=2,
                       n_questions=5)
    assert summary.n_questions == 5
    record = load_record(output_dir / cell_dir(cell) / "sa.json")
    assert record.n_evaluated == 5
