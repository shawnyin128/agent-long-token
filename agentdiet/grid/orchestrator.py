"""Per-cell driver for the phase-mapping grid.

Sequence per cell:
  1. Load Dataset.load(); cap to n_questions.
  2. Resume check: load sa.json/debate.json from disk if present.
  3. Run SA per question; write sa.json.
  4. Run debate per question; write debate.json.
  5. Calibrate N from first-10 of (sa, debate) total_tokens; write
     sc_calibration.json.
  6. Run voting with calibrated N; write voting.json.
  7. Compute CellSummary; write summary.json.

Math cells: judge=None. Code cells: judge required (default
SubprocessJudge); per-question correctness goes through the judge
and clustering uses public tests.
"""
from __future__ import annotations

import dataclasses
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from agentdiet.eval.base import Judge
from agentdiet.grid.runner import (
    aggregate_condition,
    run_debate_q_code,
    run_debate_q_math,
    run_sa_code,
    run_sa_math,
    run_voting_q_code,
    run_voting_q_math,
)
from agentdiet.grid.types import (
    CellSpec,
    CellSummary,
    ConditionRecord,
    QuestionResult,
    cell_dir,
    load_record,
    save_record,
    save_summary,
)
from agentdiet.llm_client import LLMClient
from agentdiet.voting import calibrate_n


CALIBRATION_PREFIX_DEFAULT = 10


def _is_code_cell(cell: CellSpec) -> bool:
    return cell.dataset_name in {"humaneval_plus", "livecodebench"}


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    tmp.replace(path)


def run_cell(
    cell: CellSpec,
    llm_client: LLMClient,
    questions: list,  # list[Question] or list[CodeQuestion]
    output_dir: Path,
    *,
    judge: Optional[Judge] = None,
    n_questions: Optional[int] = None,
    calibration_prefix: int = CALIBRATION_PREFIX_DEFAULT,
    force: bool = False,
    max_concurrency: int = 1,
) -> CellSummary:
    """Execute all three conditions for one cell + write artifacts.

    Returns the CellSummary written to disk.
    """
    is_code = _is_code_cell(cell)
    if is_code and judge is None:
        raise ValueError(
            f"cell {cell_dir(cell)} is a code cell ({cell.dataset_name}) "
            "but no judge was provided"
        )

    qs = questions if n_questions is None else questions[:n_questions]
    if not qs:
        raise ValueError(f"cell {cell_dir(cell)} has no questions to run")

    cdir = output_dir / cell_dir(cell)
    cdir.mkdir(parents=True, exist_ok=True)
    sa_path = cdir / "sa.json"
    voting_path = cdir / "voting.json"
    debate_path = cdir / "debate.json"
    calib_path = cdir / "sc_calibration.json"
    summary_path = cdir / "summary.json"

    # 1. Resume or run SA
    sa_record = _load_or_run(
        sa_path, force,
        lambda: _run_condition_sa(qs, cell, llm_client, judge, is_code,
                                   max_concurrency),
    )

    # 2. Resume or run debate
    debate_record = _load_or_run(
        debate_path, force,
        lambda: _run_condition_debate(qs, cell, llm_client, judge, is_code,
                                       max_concurrency),
    )

    # 3. Calibrate N from first-prefix of sa + debate token totals
    calib_n_use = min(calibration_prefix, len(sa_record.questions),
                      len(debate_record.questions))
    if calib_n_use < 1:
        raise RuntimeError(
            f"cell {cell_dir(cell)} has no calibration prefix samples"
        )
    sa_tokens = [q.total_tokens for q in sa_record.questions[:calib_n_use]
                 if q.total_tokens > 0]
    debate_tokens = [q.total_tokens for q in debate_record.questions[:calib_n_use]
                     if q.total_tokens > 0]
    if not sa_tokens or not debate_tokens:
        raise RuntimeError(
            f"cell {cell_dir(cell)} calibration prefix has no usable tokens "
            f"(sa={len(sa_tokens)}, debate={len(debate_tokens)})"
        )
    calibration = calibrate_n(debate_tokens, sa_tokens)
    calib_payload = dataclasses.asdict(calibration)
    calib_payload["calibration_prefix_n"] = calib_n_use
    _atomic_write_json(calib_path, calib_payload)

    # 4. Resume or run voting with the calibrated N
    voting_record = _load_or_run(
        voting_path, force,
        lambda: _run_condition_voting(
            qs, cell, llm_client, judge, is_code, calibration.N,
            max_concurrency,
        ),
    )

    # 5. Build summary
    summary = CellSummary(
        cell=cell,
        sa_accuracy=sa_record.accuracy,
        voting_accuracy=voting_record.accuracy,
        debate_accuracy=debate_record.accuracy,
        sa_total_tokens=sa_record.total_tokens,
        voting_total_tokens=voting_record.total_tokens,
        debate_total_tokens=debate_record.total_tokens,
        delta_debate_voting=debate_record.accuracy - voting_record.accuracy,
        delta_debate_sa=debate_record.accuracy - sa_record.accuracy,
        calibration=calib_payload,
        n_questions=len(qs),
    )
    save_summary(summary_path, summary)
    return summary


def _load_or_run(
    path: Path, force: bool, runner,
) -> ConditionRecord:
    if path.exists() and not force:
        return load_record(path)
    record = runner()
    save_record(path, record)
    return record


def _parallel_map(fn, items, max_concurrency: int) -> list:
    """Run fn(item) for each item; preserve input order in the result.

    max_concurrency=1 stays sequential (no thread pool overhead).
    """
    if max_concurrency <= 1:
        return [fn(it) for it in items]
    with ThreadPoolExecutor(max_workers=max_concurrency) as ex:
        futures = [ex.submit(fn, it) for it in items]
        return [f.result() for f in futures]


def _run_condition_sa(qs, cell, llm_client, judge, is_code,
                       max_concurrency: int = 1) -> ConditionRecord:
    if is_code:
        fn = lambda q: run_sa_code(q, cell, llm_client, judge)
    else:
        fn = lambda q: run_sa_math(q, cell, llm_client)
    results = _parallel_map(fn, qs, max_concurrency)
    return aggregate_condition(results, cell, condition="sa")


def _run_condition_debate(qs, cell, llm_client, judge, is_code,
                           max_concurrency: int = 1) -> ConditionRecord:
    if is_code:
        fn = lambda q: run_debate_q_code(q, cell, llm_client, judge)
    else:
        fn = lambda q: run_debate_q_math(
            q, cell, llm_client, prompt_variant=cell.prompt_variant,
        )
    results = _parallel_map(fn, qs, max_concurrency)
    return aggregate_condition(results, cell, condition="debate")


def _run_condition_voting(qs, cell, llm_client, judge, is_code, n_samples,
                           max_concurrency: int = 1) -> ConditionRecord:
    if is_code:
        fn = lambda q: run_voting_q_code(q, cell, llm_client, n_samples, judge)
    else:
        fn = lambda q: run_voting_q_math(q, cell, llm_client, n_samples)
    results = _parallel_map(fn, qs, max_concurrency)
    return aggregate_condition(
        results, cell, condition="voting",
        extra_meta={"n_samples": n_samples},
    )
