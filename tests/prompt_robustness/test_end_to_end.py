"""Cross-step integration: sub-grid expansion + variant prompt routing
+ cooperative byte-identity with main-grid path."""
from __future__ import annotations

import re
from pathlib import Path

from agentdiet.cli.grid import main
from agentdiet.config import Config
from agentdiet.dataset import Question
from agentdiet.grid.types import CellSpec, cell_dir
from agentdiet.llm_client import DummyBackend, LLMClient


def _fake_cfg(tmp_path: Path) -> Config:
    cfg = Config(
        artifacts_dir=tmp_path / "artifacts",
        hf_cache_dir=tmp_path / "hf_cache",
    )
    cfg.ensure_dirs()
    return cfg


def _math_responder(captured_systems: list[str] | None = None):
    def responder(messages, model, temperature, *, thinking=False, **kw):
        if captured_systems is not None:
            for m in messages:
                if m["role"] == "system":
                    captured_systems.append(m["content"])
                    break
        text = messages[-1]["content"]
        m = re.search(r"item (\d+)", text)
        if m:
            return f"#### {m.group(1)}"
        m = re.search(r"#### (\d+)", text)
        if m:
            return f"#### {m.group(1)}"
        return "#### 0"
    return responder


def _fake_dataset_factory(n: int = 3):
    def factory(cell, cfg):
        class FakeDS:
            def load(self_inner):
                return [
                    Question(qid=f"q{i}", question=f"Compute item {i}",
                             gold_answer=str(i))
                    for i in range(n)
                ]
        return FakeDS()
    return factory


def test_subgrid_writes_six_dirs_with_expected_paths(tmp_path):
    backend = DummyBackend(_math_responder())
    client = LLMClient(backend, cache_path=tmp_path / "c.jsonl")

    output_dir = tmp_path / "out"
    rc = main(
        ["--prompt-sub-grid",
         "--output-dir", str(output_dir),
         "--calibration-prefix", "2"],
        cfg=_fake_cfg(tmp_path),
        client_factory=lambda cell, cfg: client,
        dataset_factory=_fake_dataset_factory(3),
        judge_factory=lambda cell: None,
    )
    assert rc == 0

    # Cooperative cells share path with main-grid cooperative path
    main_grid_t0 = "Qwen__Qwen3-30B-A3B__gsm8k__t0"
    main_grid_t1 = "Qwen__Qwen3-30B-A3B__gsm8k__t1"
    assert (output_dir / main_grid_t0 / "summary.json").is_file()
    assert (output_dir / main_grid_t1 / "summary.json").is_file()

    # Variant cells get __pv-<variant> suffix
    for variant in ("adversarial-strict", "symmetric"):
        for thinking in (0, 1):
            d = output_dir / f"Qwen__Qwen3-30B-A3B__gsm8k__t{thinking}__pv-{variant}"
            assert (d / "summary.json").is_file()


def test_cooperative_subgrid_cell_path_matches_main_grid_cell_path(tmp_path):
    """Cooperative cell from sub-grid lands at the same on-disk path as
    a main-grid cooperative cell — no second cell_dir is created."""
    main_cell = CellSpec(
        model="Qwen/Qwen3-30B-A3B", model_family="qwen3",
        dataset_name="gsm8k", thinking=False, prompt_variant="cooperative",
    )
    sub_grid_cell = CellSpec(
        model="Qwen/Qwen3-30B-A3B", model_family="qwen3",
        dataset_name="gsm8k", thinking=False, prompt_variant="cooperative",
    )
    assert cell_dir(main_cell) == cell_dir(sub_grid_cell)


def test_adversarial_strict_sends_disagreement_obligation_to_backend(tmp_path):
    captured: list[str] = []
    backend = DummyBackend(_math_responder(captured))
    client = LLMClient(backend, cache_path=tmp_path / "c.jsonl")

    rc = main(
        ["--cell", "qwen3:gsm8k:0:adversarial-strict",
         "--output-dir", str(tmp_path / "out"),
         "--calibration-prefix", "2"],
        cfg=_fake_cfg(tmp_path),
        client_factory=lambda cell, cfg: client,
        dataset_factory=_fake_dataset_factory(3),
        judge_factory=lambda cell: None,
    )
    assert rc == 0
    # Some captured system prompt must contain the adversarial obligation.
    # SA + voting use SOLVER_PROMPT (no "disagreement" word), so the
    # adversarial skeptic prompt only shows up in debate slot 1.
    assert any("disagreement" in s.lower() for s in captured)


def test_cooperative_run_does_not_inject_adversarial_text(tmp_path):
    captured: list[str] = []
    backend = DummyBackend(_math_responder(captured))
    client = LLMClient(backend, cache_path=tmp_path / "c.jsonl")

    rc = main(
        ["--cell", "qwen3:gsm8k:0",  # cooperative default
         "--output-dir", str(tmp_path / "out"),
         "--calibration-prefix", "2"],
        cfg=_fake_cfg(tmp_path),
        client_factory=lambda cell, cfg: client,
        dataset_factory=_fake_dataset_factory(3),
        judge_factory=lambda cell: None,
    )
    assert rc == 0
    # No captured system prompt should carry the adversarial-strict
    # "disagreement" obligation phrasing.
    assert not any(
        "MUST identify at least one concrete disagreement" in s
        for s in captured
    )
