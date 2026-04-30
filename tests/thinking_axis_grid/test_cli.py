"""CLI tests: parsing, expansion, dispatch through fakes."""
from __future__ import annotations

from pathlib import Path

import pytest

from agentdiet.cli.grid import (
    DATASET_NAMES,
    MODEL_FAMILY_TO_ID,
    PILOT_CELLS,
    _expand_all,
    main,
    parse_cell_spec,
)
from agentdiet.config import Config
from agentdiet.dataset import Question
from agentdiet.grid.types import CellSpec
from agentdiet.llm_client import DummyBackend, LLMClient


def test_parse_cell_spec_qwen3_gsm8k_thinking_off():
    cell = parse_cell_spec("qwen3:gsm8k:0")
    assert cell.model == "Qwen/Qwen3-30B-A3B"
    assert cell.model_family == "qwen3"
    assert cell.dataset_name == "gsm8k"
    assert cell.thinking is False


def test_parse_cell_spec_gpt_oss_livecodebench_thinking_on():
    cell = parse_cell_spec("gpt-oss:livecodebench:1")
    assert cell.model == "openai/gpt-oss-20b"
    assert cell.thinking is True


def test_parse_cell_spec_unknown_family_raises():
    with pytest.raises(ValueError, match="unknown model family"):
        parse_cell_spec("llama:gsm8k:0")


def test_parse_cell_spec_unknown_dataset_raises():
    with pytest.raises(ValueError, match="unknown dataset"):
        parse_cell_spec("qwen3:bogus:0")


def test_parse_cell_spec_bad_thinking_value_raises():
    with pytest.raises(ValueError, match="thinking must be 0 or 1"):
        parse_cell_spec("qwen3:gsm8k:2")


def test_parse_cell_spec_wrong_segment_count_raises():
    with pytest.raises(ValueError, match="cell spec must be"):
        parse_cell_spec("qwen3:gsm8k")


def test_pilot_cells_are_two():
    assert len(PILOT_CELLS) == 2
    assert "qwen3:gsm8k:0" in PILOT_CELLS
    assert "qwen3:humaneval_plus:0" in PILOT_CELLS


def test_expand_all_thinking_off_yields_eight_cells():
    all_off = _expand_all(thinking=False)
    assert len(all_off) == 8
    assert len(MODEL_FAMILY_TO_ID) * len(DATASET_NAMES) == 8
    for spec in all_off:
        assert spec.endswith(":0")


def test_expand_all_thinking_on_mirrors_off():
    on = _expand_all(thinking=True)
    assert len(on) == 8
    for spec in on:
        assert spec.endswith(":1")


def _fake_question_set(n: int = 3):
    return [
        Question(qid=f"q{i}", question=f"Compute item {i}", gold_answer=str(i))
        for i in range(n)
    ]


def _fake_cfg(tmp_path: Path) -> Config:
    cfg = Config(
        artifacts_dir=tmp_path / "artifacts",
        hf_cache_dir=tmp_path / "hf_cache",
    )
    cfg.ensure_dirs()
    return cfg


def test_main_with_mocked_factories(tmp_path):
    """End-to-end CLI flow through dataset/client/judge factories."""
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
    client = LLMClient(backend, cache_path=tmp_path / "c.jsonl")

    def fake_dataset(cell, cfg):
        class FakeDS:
            def load(self_inner):
                return _fake_question_set(3)
        return FakeDS()

    def fake_client(cell, cfg):
        return client

    def fake_judge(cell):
        return None

    output_dir = tmp_path / "out"
    cfg = _fake_cfg(tmp_path)

    rc = main(
        ["--cell", "qwen3:gsm8k:0", "--output-dir", str(output_dir),
         "--calibration-prefix", "2"],
        cfg=cfg,
        client_factory=fake_client,
        dataset_factory=fake_dataset,
        judge_factory=fake_judge,
    )
    assert rc == 0
    summary_path = output_dir / "Qwen__Qwen3-30B-A3B__gsm8k__t0" / "summary.json"
    assert summary_path.is_file()


def test_main_no_cells_returns_error(tmp_path):
    cfg = _fake_cfg(tmp_path)
    rc = main([], cfg=cfg)
    assert rc == 2


def test_main_pilot_expands_to_two_cells(tmp_path):
    captured_cells: list[CellSpec] = []
    import re

    def responder(messages, model, temperature, *, thinking=False, **kw):
        text = messages[-1]["content"]
        if "## Code" in text or "## Notes" in text:
            return ("## Notes\nx.\n\n## Code\n```python\n"
                    "def add(a, b): return a + b\n```\n")
        m = re.search(r"item (\d+)", text)
        if m:
            return f"#### {m.group(1)}"
        m = re.search(r"#### (\d+)", text)
        if m:
            return f"#### {m.group(1)}"
        # Code prompt for SA / voting / debate of code path
        return ("## Notes\nx.\n\n## Code\n```python\n"
                "def add(a, b): return a + b\n```\n")

    from agentdiet.eval.base import CodeQuestion, TestCase
    from agentdiet.eval.judges import SubprocessJudge

    def fake_dataset(cell, cfg):
        captured_cells.append(cell)

        class FakeDS:
            def load(self_inner):
                if cell.dataset_name == "gsm8k":
                    return _fake_question_set(3)
                # humaneval_plus
                return [
                    CodeQuestion(
                        qid=f"add{i}",
                        prompt=("def add(a, b):\n    '''Return a+b.\n"
                                "    >>> add(1, 2)\n    3\n    '''"),
                        entry_point="add",
                        public_tests=[TestCase(name="ex",
                                               script="assert add(1, 2) == 3")],
                        hidden_tests=[TestCase(name="h",
                                               script="assert add(2, 3) == 5")],
                    )
                    for i in range(2)
                ]
        return FakeDS()

    backend = DummyBackend(responder)
    client = LLMClient(backend, cache_path=tmp_path / "c.jsonl")

    def fake_client(cell, cfg):
        return client

    def fake_judge(cell):
        if cell.dataset_name in {"humaneval_plus", "livecodebench"}:
            return SubprocessJudge()
        return None

    rc = main(
        ["--pilot", "--output-dir", str(tmp_path / "out"),
         "--calibration-prefix", "2"],
        cfg=_fake_cfg(tmp_path),
        client_factory=fake_client,
        dataset_factory=fake_dataset,
        judge_factory=fake_judge,
    )
    assert rc == 0
    assert len(captured_cells) == 2
    assert {c.dataset_name for c in captured_cells} == {"gsm8k", "humaneval_plus"}
