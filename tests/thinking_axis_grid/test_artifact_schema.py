"""On-disk JSON schema sanity checks — used by phase-mapping-analysis."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentdiet.dataset import Question
from agentdiet.grid.orchestrator import run_cell
from agentdiet.grid.types import CellSpec, cell_dir
from agentdiet.llm_client import DummyBackend, LLMClient


REQUIRED_SUMMARY_KEYS = (
    "cell", "sa_accuracy", "voting_accuracy", "debate_accuracy",
    "sa_total_tokens", "voting_total_tokens", "debate_total_tokens",
    "delta_debate_voting", "delta_debate_sa", "calibration", "n_questions",
)
REQUIRED_CALIBRATION_KEYS = (
    "N_raw", "N", "mean_debate_tokens", "mean_sa_tokens",
    "over_budget_factor", "floor_active",
)
REQUIRED_RECORD_KEYS = (
    "condition", "cell", "questions", "n_evaluated",
    "accuracy", "total_tokens",
)
REQUIRED_QUESTION_KEYS = (
    "qid", "gold", "final_answer", "correct",
    "prompt_tokens", "completion_tokens", "total_tokens",
)


def _setup_cell(tmp_path: Path):
    import re

    def responder(messages, model, temperature, *, thinking=False, **kw):
        text = messages[-1]["content"]
        m = re.search(r"item (\d+)", text)
        if m:
            return f"#### {m.group(1)}"
        m = re.search(r"#### (\d+)", text)
        if m:
            return f"#### {m.group(1)}"
        return "#### 0"

    backend = DummyBackend(responder)
    client = LLMClient(backend, cache_path=tmp_path / "cache.jsonl")
    cell = CellSpec(
        model="Qwen/Qwen3-30B-A3B", model_family="qwen3",
        dataset_name="gsm8k", thinking=False,
    )
    questions = [
        Question(qid=f"q{i}", question=f"Compute item {i}", gold_answer=str(i))
        for i in range(4)
    ]
    out = tmp_path / "out"
    run_cell(
        cell=cell, llm_client=client, questions=questions,
        output_dir=out, calibration_prefix=2,
    )
    return out / cell_dir(cell)


def test_summary_json_has_required_keys(tmp_path):
    cdir = _setup_cell(tmp_path)
    payload = json.loads((cdir / "summary.json").read_text(encoding="utf-8"))
    for key in REQUIRED_SUMMARY_KEYS:
        assert key in payload, f"summary.json missing key {key!r}"
    for key in REQUIRED_CALIBRATION_KEYS:
        assert key in payload["calibration"], (
            f"summary.calibration missing key {key!r}"
        )


def test_per_condition_records_have_required_keys(tmp_path):
    cdir = _setup_cell(tmp_path)
    for fname in ("sa.json", "voting.json", "debate.json"):
        payload = json.loads((cdir / fname).read_text(encoding="utf-8"))
        for key in REQUIRED_RECORD_KEYS:
            assert key in payload, f"{fname} missing key {key!r}"
        # Per-question rows
        for q in payload["questions"]:
            for key in REQUIRED_QUESTION_KEYS:
                assert key in q, f"{fname} question row missing {key!r}"


def test_summary_token_totals_match_records(tmp_path):
    cdir = _setup_cell(tmp_path)
    summary = json.loads((cdir / "summary.json").read_text(encoding="utf-8"))
    for cond, key in (("sa", "sa_total_tokens"),
                      ("voting", "voting_total_tokens"),
                      ("debate", "debate_total_tokens")):
        record = json.loads((cdir / f"{cond}.json").read_text(encoding="utf-8"))
        assert summary[key] == record["total_tokens"]


def test_summary_delta_debate_voting_matches_accuracy_difference(tmp_path):
    cdir = _setup_cell(tmp_path)
    payload = json.loads((cdir / "summary.json").read_text(encoding="utf-8"))
    expected = payload["debate_accuracy"] - payload["voting_accuracy"]
    assert payload["delta_debate_voting"] == pytest.approx(expected)


def test_summary_delta_debate_sa_matches_accuracy_difference(tmp_path):
    cdir = _setup_cell(tmp_path)
    payload = json.loads((cdir / "summary.json").read_text(encoding="utf-8"))
    expected = payload["debate_accuracy"] - payload["sa_accuracy"]
    assert payload["delta_debate_sa"] == pytest.approx(expected)


def test_calibration_prefix_n_recorded(tmp_path):
    cdir = _setup_cell(tmp_path)
    summary = json.loads((cdir / "summary.json").read_text(encoding="utf-8"))
    cb = summary["calibration"]
    assert "calibration_prefix_n" in cb
    assert cb["calibration_prefix_n"] >= 1
