"""Paired bootstrap CI for per-cell Delta + cell-record loader.

Per spec §4.7: 10000 paired bootstrap resamples over the question set
to derive a 95% CI for Delta = acc(debate) - acc(voting). "Paired"
means the same question's correctness across two conditions is
sampled together — preserves the matched structure.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    delta: float
    ci_low: float
    ci_high: float
    n_resamples: int


def paired_bootstrap_delta(
    per_q_correct_a: list[bool],
    per_q_correct_b: list[bool],
    n_resamples: int = 10000,
    seed: int = 42,
    ci: float = 0.95,
) -> BootstrapResult:
    """Paired bootstrap on Delta = mean(correct_a) - mean(correct_b).

    Pairs are matched by index (same question across two conditions).
    Returns observed delta + lower/upper quantiles of the resample
    distribution.
    """
    a = np.asarray(per_q_correct_a, dtype=np.float64)
    b = np.asarray(per_q_correct_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(
            f"per_q_correct_a length {a.shape[0]} does not match "
            f"per_q_correct_b length {b.shape[0]}"
        )
    if a.size == 0:
        raise ValueError("per_q_correct vectors are empty")
    if not (0 < ci < 1):
        raise ValueError(f"ci must be in (0, 1); got {ci}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1; got {n_resamples}")

    observed_delta = float(a.mean() - b.mean())

    rng = np.random.default_rng(seed)
    n = a.shape[0]
    deltas = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        deltas[i] = float(a[idx].mean() - b[idx].mean())

    alpha = (1.0 - ci) / 2.0
    ci_low = float(np.quantile(deltas, alpha))
    ci_high = float(np.quantile(deltas, 1.0 - alpha))
    return BootstrapResult(
        delta=observed_delta,
        ci_low=ci_low,
        ci_high=ci_high,
        n_resamples=n_resamples,
    )


# ---------------------------------------------------------------------------
# Cell loader
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellAnalysis:
    cell_dirname: str
    model: str
    model_family: str
    dataset_name: str
    thinking: bool
    prompt_variant: str
    sa_accuracy: float
    voting_accuracy: float
    debate_accuracy: float
    delta_debate_voting: float
    delta_ci_low: float
    delta_ci_high: float
    over_budget_factor: float
    sa_per_q_correct: list[bool]
    voting_per_q_correct: list[bool]
    debate_per_q_correct: list[bool]
    qids: list[str]
    n_questions: int
    extra: dict[str, Any] = field(default_factory=dict)


_REQUIRED_FILES = ("sa.json", "voting.json", "debate.json", "summary.json")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _per_q_correctness(record: dict) -> tuple[list[bool], list[str]]:
    questions = record.get("questions", [])
    correct = [bool(q.get("correct", False)) for q in questions]
    qids = [str(q.get("qid", "")) for q in questions]
    return correct, qids


def load_cell_summary(
    grid_dir: Path,
    cell_dirname: str,
    n_resamples: int = 10000,
    seed: int = 42,
    ci: float = 0.95,
) -> CellAnalysis:
    """Read one cell directory and produce a CellAnalysis with bootstrap CI.

    Raises FileNotFoundError if any of the four required artifact JSONs
    are missing.
    """
    cell_path = Path(grid_dir) / cell_dirname
    for fname in _REQUIRED_FILES:
        p = cell_path / fname
        if not p.is_file():
            raise FileNotFoundError(
                f"cell {cell_dirname!r} missing artifact {fname} (looked in {p})"
            )
    sa = _load_json(cell_path / "sa.json")
    voting = _load_json(cell_path / "voting.json")
    debate = _load_json(cell_path / "debate.json")
    summary = _load_json(cell_path / "summary.json")

    sa_correct, sa_qids = _per_q_correctness(sa)
    voting_correct, _ = _per_q_correctness(voting)
    debate_correct, _ = _per_q_correctness(debate)

    if not (len(sa_correct) == len(voting_correct) == len(debate_correct)):
        raise ValueError(
            f"cell {cell_dirname!r} per-condition question counts disagree: "
            f"sa={len(sa_correct)} voting={len(voting_correct)} "
            f"debate={len(debate_correct)}"
        )

    boot = paired_bootstrap_delta(
        debate_correct, voting_correct,
        n_resamples=n_resamples, seed=seed, ci=ci,
    )

    cell_block = summary.get("cell", {})
    calibration = summary.get("calibration", {})
    return CellAnalysis(
        cell_dirname=cell_dirname,
        model=str(cell_block.get("model", "")),
        model_family=str(cell_block.get("model_family", "")),
        dataset_name=str(cell_block.get("dataset_name", "")),
        thinking=bool(cell_block.get("thinking", False)),
        prompt_variant=str(cell_block.get("prompt_variant", "cooperative")),
        sa_accuracy=float(summary.get("sa_accuracy", 0.0)),
        voting_accuracy=float(summary.get("voting_accuracy", 0.0)),
        debate_accuracy=float(summary.get("debate_accuracy", 0.0)),
        delta_debate_voting=boot.delta,
        delta_ci_low=boot.ci_low,
        delta_ci_high=boot.ci_high,
        over_budget_factor=float(calibration.get("over_budget_factor", 1.0)),
        sa_per_q_correct=sa_correct,
        voting_per_q_correct=voting_correct,
        debate_per_q_correct=debate_correct,
        qids=sa_qids,  # all three should have identical qid order
        n_questions=int(summary.get("n_questions", len(sa_correct))),
        extra={
            "calibration": calibration,
            "sa_total_tokens": summary.get("sa_total_tokens"),
            "voting_total_tokens": summary.get("voting_total_tokens"),
            "debate_total_tokens": summary.get("debate_total_tokens"),
        },
    )


def compute_per_cell_analysis(
    grid_dir: Path,
    n_resamples: int = 10000,
    seed: int = 42,
    ci: float = 0.95,
) -> list[CellAnalysis]:
    """Glob every cell directory under grid_dir and analyze each.

    Subdirectories without all four required JSONs are skipped silently
    (they may be in-progress cells); use load_cell_summary directly to
    raise on missing artifacts.
    """
    grid_dir = Path(grid_dir)
    if not grid_dir.is_dir():
        raise FileNotFoundError(f"grid_dir does not exist: {grid_dir}")

    cells: list[CellAnalysis] = []
    for child in sorted(grid_dir.iterdir()):
        if not child.is_dir():
            continue
        if not all((child / fname).is_file() for fname in _REQUIRED_FILES):
            continue
        cells.append(load_cell_summary(
            grid_dir, child.name,
            n_resamples=n_resamples, seed=seed, ci=ci,
        ))
    return cells
