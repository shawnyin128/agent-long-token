"""CLI to run one or more (model, dataset, thinking) cells.

Cell spec format: "model_family:dataset_name:thinking_int"
e.g. "qwen3:gsm8k:0", "gpt-oss:livecodebench:1".

Usage on HPC (after vLLM serve is up at $AGENTDIET_BASE_URL):
    python -m agentdiet.cli.grid --pilot
    python -m agentdiet.cli.grid --cell qwen3:gsm8k:0
    python -m agentdiet.cli.grid --all-thinking-off
    python -m agentdiet.cli.grid --all-thinking-on --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from agentdiet.config import Config, get_config
from agentdiet.eval.base import Judge
from agentdiet.eval.datasets import (
    AIMEMultiYearDataset,
    GSM8KDataset,
    HumanEvalPlusDataset,
    LiveCodeBenchDataset,
)
from agentdiet.eval.judges import SubprocessJudge
from agentdiet.grid.orchestrator import run_cell
from agentdiet.grid.types import CellSpec, cell_dir
from agentdiet.llm_client import LLMClient, OpenAIBackend


# Family -> (HF model id)
MODEL_FAMILY_TO_ID = {
    "qwen3": "Qwen/Qwen3-30B-A3B",
    "gpt-oss": "openai/gpt-oss-20b",
}

DATASET_NAMES = ("gsm8k", "aime", "humaneval_plus", "livecodebench")
CODE_DATASETS = {"humaneval_plus", "livecodebench"}
PROMPT_VARIANT_NAMES = ("cooperative", "adversarial-strict", "symmetric")

DEFAULT_OUTPUT_DIR = Path("artifacts/grid")


def parse_cell_spec(spec: str, default_variant: str = "cooperative") -> CellSpec:
    """Parse 'family:dataset:thinking_int[:variant]' into a CellSpec.

    The variant segment is optional; when absent we use default_variant
    (which the CLI sources from --prompt-variant or 'cooperative').
    """
    parts = spec.split(":")
    if len(parts) not in (3, 4):
        raise ValueError(
            f"cell spec must be 'family:dataset:thinking_int[:variant]', "
            f"got {spec!r}"
        )
    family = parts[0]
    dataset_name = parts[1]
    thinking_str = parts[2]
    variant = parts[3] if len(parts) == 4 else default_variant
    if family not in MODEL_FAMILY_TO_ID:
        raise ValueError(
            f"unknown model family {family!r}; "
            f"choose one of {sorted(MODEL_FAMILY_TO_ID.keys())}"
        )
    if dataset_name not in DATASET_NAMES:
        raise ValueError(
            f"unknown dataset {dataset_name!r}; "
            f"choose one of {list(DATASET_NAMES)}"
        )
    if thinking_str not in ("0", "1"):
        raise ValueError(f"thinking must be 0 or 1, got {thinking_str!r}")
    if variant not in PROMPT_VARIANT_NAMES:
        raise ValueError(
            f"unknown prompt variant {variant!r}; "
            f"choose one of {list(PROMPT_VARIANT_NAMES)}"
        )
    return CellSpec(
        model=MODEL_FAMILY_TO_ID[family],
        model_family=family,
        dataset_name=dataset_name,
        thinking=(thinking_str == "1"),
        prompt_variant=variant,
    )


PILOT_CELLS = ("qwen3:gsm8k:0", "qwen3:humaneval_plus:0")
PROMPT_SUB_GRID_CELLS = tuple(
    f"qwen3:gsm8k:{t}:{v}"
    for v in PROMPT_VARIANT_NAMES
    for t in (0, 1)
)


def _expand_all(thinking: bool) -> list[str]:
    flag = 1 if thinking else 0
    return [
        f"{family}:{dataset}:{flag}"
        for family in MODEL_FAMILY_TO_ID
        for dataset in DATASET_NAMES
    ]


def _build_dataset(cell: CellSpec, cfg: Config):
    if cell.dataset_name == "gsm8k":
        return GSM8KDataset(n=80, seed=cfg.seed, cache_dir=cfg.hf_cache_dir)
    if cell.dataset_name == "aime":
        return AIMEMultiYearDataset(
            data_dir=cfg.artifacts_dir / "datasets",
            sample_seed=cfg.seed,
        )
    if cell.dataset_name == "humaneval_plus":
        return HumanEvalPlusDataset(cap=80)
    if cell.dataset_name == "livecodebench":
        return LiveCodeBenchDataset(cap=80)
    raise ValueError(f"unknown dataset_name: {cell.dataset_name}")


def _build_judge_for_cell(cell: CellSpec) -> Optional[Judge]:
    if cell.dataset_name in CODE_DATASETS:
        return SubprocessJudge()
    return None


def _build_client(cell: CellSpec, cfg: Config) -> LLMClient:
    backend = OpenAIBackend(
        base_url=cfg.base_url, api_key=cfg.api_key,
        timeout_s=cfg.request_timeout_s,
        model_family=cell.model_family,  # type: ignore[arg-type]
    )
    return LLMClient(backend, cache_path=cfg.cache_path,
                     max_retries=cfg.max_retries)


def main(
    argv: list[str] | None = None,
    *,
    cfg: Config | None = None,
    client_factory=None,
    dataset_factory=None,
    judge_factory=None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Run one or more (model, dataset, thinking) grid cells.",
    )
    parser.add_argument("--cell", action="append", default=[],
                        help="Cell spec 'family:dataset:thinking[:variant]'. May repeat.")
    parser.add_argument("--pilot", action="store_true",
                        help="Run pilot cells: qwen3:gsm8k:0 + qwen3:humaneval_plus:0")
    parser.add_argument("--all-thinking-off", action="store_true",
                        help="Run all 8 thinking-off cells across both models and 4 datasets")
    parser.add_argument("--all-thinking-on", action="store_true",
                        help="Run all 8 thinking-on cells across both models and 4 datasets")
    parser.add_argument("--prompt-sub-grid", action="store_true",
                        help="Run the 6-cell prompt-robustness sub-grid (Qwen3 + GSM8K x 3 prompt variants x thinking on/off)")
    parser.add_argument("--prompt-variant",
                        choices=PROMPT_VARIANT_NAMES, default="cooperative",
                        help="Default prompt variant for cells whose spec omits the 4th segment")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to write artifacts/grid/{cell}/ subtrees")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if existing JSON artifacts are present")
    parser.add_argument("--n", type=int, default=None,
                        help="Override n_questions per cell (default: 80)")
    parser.add_argument("--calibration-prefix", type=int, default=10,
                        help="First-N questions used to calibrate voting N (default 10)")
    args = parser.parse_args(argv)

    specs: list[str] = list(args.cell)
    if args.pilot:
        specs.extend(PILOT_CELLS)
    if args.all_thinking_off:
        specs.extend(_expand_all(thinking=False))
    if args.all_thinking_on:
        specs.extend(_expand_all(thinking=True))
    if args.prompt_sub_grid:
        specs.extend(PROMPT_SUB_GRID_CELLS)
    if not specs:
        print(
            "ERROR: no cells specified; use --cell, --pilot, "
            "--all-thinking-{off,on}, or --prompt-sub-grid",
            file=sys.stderr,
        )
        return 2
    # Dedup while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for s in specs:
        if s not in seen:
            seen.add(s)
            deduped.append(s)

    if cfg is None:
        cfg = get_config()
    cfg.ensure_dirs()

    cells: list[CellSpec] = []
    for s in deduped:
        cells.append(parse_cell_spec(s, default_variant=args.prompt_variant))

    print(f"running {len(cells)} cell(s):", file=sys.stderr)
    for cell in cells:
        print(f"  - {cell_dir(cell)}", file=sys.stderr)

    df = dataset_factory or _build_dataset
    cf = client_factory or _build_client
    jf = judge_factory or _build_judge_for_cell

    for cell in cells:
        dataset = df(cell, cfg)
        questions = dataset.load()
        client = cf(cell, cfg)
        judge = jf(cell)
        summary = run_cell(
            cell=cell,
            llm_client=client,
            questions=questions,
            output_dir=args.output_dir,
            judge=judge,
            n_questions=args.n,
            calibration_prefix=args.calibration_prefix,
            force=args.force,
        )
        print(
            f"  {cell_dir(cell)} | "
            f"sa={summary.sa_accuracy:.3f} "
            f"voting={summary.voting_accuracy:.3f} "
            f"debate={summary.debate_accuracy:.3f} "
            f"d_dv={summary.delta_debate_voting:+.3f} "
            f"d_ds={summary.delta_debate_sa:+.3f}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
