"""Grid types: dataclass round-trip + cell_dir naming + record I/O."""
from __future__ import annotations

import dataclasses
import json

from agentdiet.grid.types import (
    CellSpec,
    CellSummary,
    ConditionRecord,
    QuestionResult,
    cell_dir,
    load_record,
    load_summary,
    save_record,
    save_summary,
)


def _cell() -> CellSpec:
    return CellSpec(
        model="Qwen/Qwen3-30B-A3B",
        model_family="qwen3",
        dataset_name="gsm8k",
        thinking=False,
    )


def test_cell_dir_format():
    cell = _cell()
    assert cell_dir(cell) == "Qwen__Qwen3-30B-A3B__gsm8k__t0"


def test_cell_dir_thinking_on_uses_t1():
    cell = CellSpec(
        model="openai/gpt-oss-20b", model_family="gpt-oss",
        dataset_name="livecodebench", thinking=True,
    )
    assert cell_dir(cell) == "openai__gpt-oss-20b__livecodebench__t1"


def test_cellspec_round_trips_through_asdict_json():
    cell = _cell()
    payload = json.dumps(dataclasses.asdict(cell))
    parsed = CellSpec(**json.loads(payload))
    assert parsed == cell


def test_question_result_total_tokens_field():
    q = QuestionResult(
        qid="q1", gold="42", final_answer="42", correct=True,
        prompt_tokens=10, completion_tokens=5, total_tokens=15,
    )
    assert q.total_tokens == 15
    assert q.correct is True


def test_condition_record_round_trip(tmp_path):
    cell = _cell()
    questions = [
        QuestionResult(qid=f"q{i}", gold=str(i), final_answer=str(i),
                       correct=True, prompt_tokens=10, completion_tokens=5,
                       total_tokens=15)
        for i in range(3)
    ]
    record = ConditionRecord(
        condition="sa", cell=cell,
        questions=questions, n_evaluated=3,
        accuracy=1.0, total_tokens=45,
    )
    path = tmp_path / "sa.json"
    save_record(path, record)
    assert path.is_file()
    loaded = load_record(path)
    assert loaded == record


def test_cell_summary_round_trip(tmp_path):
    cell = _cell()
    summary = CellSummary(
        cell=cell,
        sa_accuracy=0.7,
        voting_accuracy=0.75,
        debate_accuracy=0.78,
        sa_total_tokens=1000,
        voting_total_tokens=3500,
        debate_total_tokens=3000,
        delta_debate_voting=0.03,
        delta_debate_sa=0.08,
        calibration={"N": 12, "N_raw": 10, "over_budget_factor": 1.05,
                     "floor_active": False,
                     "mean_debate_tokens": 30.0, "mean_sa_tokens": 25.0},
        n_questions=80,
    )
    path = tmp_path / "summary.json"
    save_summary(path, summary)
    loaded = load_summary(path)
    assert loaded == summary


def test_atomic_write_does_not_leave_tmp(tmp_path):
    cell = _cell()
    record = ConditionRecord(
        condition="sa", cell=cell, questions=[], n_evaluated=0,
        accuracy=0.0, total_tokens=0,
    )
    path = tmp_path / "sa.json"
    save_record(path, record)
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []
