"""CellSpec.prompt_variant + cell_dir suffix + CLI parser/sub-grid."""
from __future__ import annotations

from pathlib import Path

import pytest

from agentdiet.cli.grid import (
    PROMPT_SUB_GRID_CELLS,
    PROMPT_VARIANT_NAMES,
    main,
    parse_cell_spec,
)
from agentdiet.config import Config
from agentdiet.dataset import Question
from agentdiet.grid.types import CellSpec, cell_dir
from agentdiet.llm_client import DummyBackend, LLMClient


def _cell(variant: str = "cooperative", thinking: bool = False) -> CellSpec:
    return CellSpec(
        model="Qwen/Qwen3-30B-A3B", model_family="qwen3",
        dataset_name="gsm8k", thinking=thinking,
        prompt_variant=variant,
    )


# --- CellSpec + cell_dir --------------------------------------------------


def test_cellspec_default_prompt_variant_is_cooperative():
    cell = CellSpec(
        model="Qwen/Qwen3-30B-A3B", model_family="qwen3",
        dataset_name="gsm8k", thinking=False,
    )
    assert cell.prompt_variant == "cooperative"


def test_cell_dir_cooperative_unchanged_format():
    """Regression guard: cooperative cell_dir matches the legacy format
    so existing artifacts/grid/{...}/ paths keep working."""
    legacy = "Qwen__Qwen3-30B-A3B__gsm8k__t0"
    assert cell_dir(_cell(variant="cooperative", thinking=False)) == legacy


def test_cell_dir_adversarial_strict_appends_suffix():
    cell = _cell(variant="adversarial-strict", thinking=False)
    assert cell_dir(cell) == "Qwen__Qwen3-30B-A3B__gsm8k__t0__pv-adversarial-strict"


def test_cell_dir_symmetric_appends_suffix():
    cell = _cell(variant="symmetric", thinking=True)
    assert cell_dir(cell) == "Qwen__Qwen3-30B-A3B__gsm8k__t1__pv-symmetric"


# --- parse_cell_spec ------------------------------------------------------


def test_parse_three_segment_spec_uses_default_cooperative():
    cell = parse_cell_spec("qwen3:gsm8k:0")
    assert cell.prompt_variant == "cooperative"


def test_parse_three_segment_spec_uses_explicit_default_variant():
    cell = parse_cell_spec("qwen3:gsm8k:1", default_variant="symmetric")
    assert cell.prompt_variant == "symmetric"


def test_parse_four_segment_spec_extracts_variant():
    cell = parse_cell_spec("qwen3:gsm8k:0:adversarial-strict")
    assert cell.prompt_variant == "adversarial-strict"


def test_parse_four_segment_spec_overrides_default_variant():
    cell = parse_cell_spec("qwen3:gsm8k:0:symmetric", default_variant="cooperative")
    assert cell.prompt_variant == "symmetric"


def test_parse_unknown_variant_raises():
    with pytest.raises(ValueError, match="unknown prompt variant"):
        parse_cell_spec("qwen3:gsm8k:0:bogus")


def test_parse_too_many_segments_raises():
    with pytest.raises(ValueError, match="cell spec must be"):
        parse_cell_spec("qwen3:gsm8k:0:symmetric:extra")


# --- prompt sub-grid expansion --------------------------------------------


def test_prompt_sub_grid_has_six_cells():
    assert len(PROMPT_SUB_GRID_CELLS) == 6


def test_prompt_sub_grid_covers_three_variants_two_thinking_states():
    cells = [parse_cell_spec(s) for s in PROMPT_SUB_GRID_CELLS]
    variants = {c.prompt_variant for c in cells}
    assert variants == set(PROMPT_VARIANT_NAMES)
    thinkings = {c.thinking for c in cells}
    assert thinkings == {False, True}
    # All Qwen3 + GSM8K
    assert all(c.model_family == "qwen3" for c in cells)
    assert all(c.dataset_name == "gsm8k" for c in cells)


# --- CLI integration ------------------------------------------------------


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


def test_main_with_explicit_variant_writes_pv_suffixed_dir(tmp_path):
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
    rc = main(
        ["--cell", "qwen3:gsm8k:0:symmetric",
         "--output-dir", str(output_dir),
         "--calibration-prefix", "2"],
        cfg=_fake_cfg(tmp_path),
        client_factory=fake_client,
        dataset_factory=fake_dataset,
        judge_factory=fake_judge,
    )
    assert rc == 0
    expected_dir = output_dir / "Qwen__Qwen3-30B-A3B__gsm8k__t0__pv-symmetric"
    assert (expected_dir / "summary.json").is_file()


def test_main_global_prompt_variant_flag_applies_to_three_segment_specs(tmp_path):
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

    output_dir = tmp_path / "out"
    rc = main(
        ["--cell", "qwen3:gsm8k:1",
         "--prompt-variant", "adversarial-strict",
         "--output-dir", str(output_dir),
         "--calibration-prefix", "2"],
        cfg=_fake_cfg(tmp_path),
        client_factory=lambda cell, cfg: client,
        dataset_factory=fake_dataset,
        judge_factory=lambda cell: None,
    )
    assert rc == 0
    expected = output_dir / "Qwen__Qwen3-30B-A3B__gsm8k__t1__pv-adversarial-strict"
    assert (expected / "summary.json").is_file()


def test_main_explicit_variant_overrides_global_flag(tmp_path):
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

    output_dir = tmp_path / "out"
    rc = main(
        # 4-segment spec wins over global --prompt-variant
        ["--cell", "qwen3:gsm8k:0:symmetric",
         "--prompt-variant", "adversarial-strict",
         "--output-dir", str(output_dir),
         "--calibration-prefix", "2"],
        cfg=_fake_cfg(tmp_path),
        client_factory=lambda cell, cfg: client,
        dataset_factory=fake_dataset,
        judge_factory=lambda cell: None,
    )
    assert rc == 0
    assert (output_dir / "Qwen__Qwen3-30B-A3B__gsm8k__t0__pv-symmetric"
            / "summary.json").is_file()
    # The adversarial-strict directory must NOT exist for this run
    assert not (output_dir / "Qwen__Qwen3-30B-A3B__gsm8k__t0__pv-adversarial-strict").is_dir()


def test_main_prompt_sub_grid_writes_six_distinct_dirs(tmp_path):
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

    output_dir = tmp_path / "out"
    rc = main(
        ["--prompt-sub-grid",
         "--output-dir", str(output_dir),
         "--calibration-prefix", "2"],
        cfg=_fake_cfg(tmp_path),
        client_factory=lambda cell, cfg: client,
        dataset_factory=fake_dataset,
        judge_factory=lambda cell: None,
    )
    assert rc == 0
    # 6 unique cell_dir entries
    cell_dirs = sorted(p.name for p in output_dir.iterdir() if p.is_dir())
    assert len(cell_dirs) == 6
    # 4 of them have pv- suffix (adversarial-strict + symmetric × 2 thinkings),
    # 2 are bare cooperative cells (just t0 + t1)
    pv_dirs = [d for d in cell_dirs if "__pv-" in d]
    assert len(pv_dirs) == 4
    bare_dirs = [d for d in cell_dirs if "__pv-" not in d]
    assert sorted(bare_dirs) == sorted([
        "Qwen__Qwen3-30B-A3B__gsm8k__t0",
        "Qwen__Qwen3-30B-A3B__gsm8k__t1",
    ])
