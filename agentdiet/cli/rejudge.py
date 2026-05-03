"""Re-judge an existing code-cell artifact directory in place.

Use case: a cell ran end-to-end but the dataset loader was buggy at
the time (e.g. LiveCodeBench hidden_tests were empty), so every
question got correct=False with hidden_pass=hidden_total=0. Now that
the loader is fixed, we want to re-grade the existing
sa.json/voting.json/debate.json final_answer fields against the
correct hidden_tests WITHOUT re-running the LLM.

Usage:
    python -m agentdiet.cli.rejudge --cell-dir artifacts/grid/<cell>
    python -m agentdiet.cli.rejudge --cell-dir <a> --cell-dir <b>

Only code cells are supported (livecodebench, humaneval_plus); math
cells are graded by string match and don't need a judge.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Optional

from agentdiet.eval.base import CodeQuestion, Judge
from agentdiet.eval.datasets import HumanEvalPlusDataset, LiveCodeBenchDataset
from agentdiet.eval.judges import SubprocessJudge
from agentdiet.grid.types import (
    CellSpec,
    CellSummary,
    ConditionRecord,
    QuestionResult,
    load_record,
    save_record,
    save_summary,
)


CODE_DATASETS = {"livecodebench", "humaneval_plus"}
CONDITION_FILES = ("sa.json", "voting.json", "debate.json")


def _build_dataset(cell: CellSpec):
    if cell.dataset_name == "livecodebench":
        return LiveCodeBenchDataset(cap=80)
    if cell.dataset_name == "humaneval_plus":
        return HumanEvalPlusDataset(cap=80)
    raise ValueError(
        f"rejudge only supports code datasets, got {cell.dataset_name}"
    )


def _rejudge_record(
    record: ConditionRecord,
    qid_to_question: dict[str, CodeQuestion],
    judge: Judge,
    timeout_s: float,
    max_workers: int,
) -> ConditionRecord:
    """Re-grade each QuestionResult.final_answer against the (now
    correct) hidden_tests. Preserves all token counts and final_answer.
    Updates correct + meta.hidden_pass + meta.hidden_total only."""

    def regrade_one(qr: QuestionResult) -> QuestionResult:
        q = qid_to_question.get(qr.qid)
        code = qr.final_answer or ""
        if q is None or not q.hidden_tests:
            new_meta = dict(qr.meta)
            new_meta["hidden_pass"] = 0
            new_meta["hidden_total"] = 0
            return replace(qr, correct=False, meta=new_meta)
        res = judge.run(code, q.hidden_tests, timeout_s=timeout_s)
        new_meta = dict(qr.meta)
        new_meta["hidden_pass"] = res.n_passed
        new_meta["hidden_total"] = res.total
        correct = res.total > 0 and res.n_passed == res.total
        return replace(qr, correct=correct, meta=new_meta)

    if max_workers <= 1:
        new_qs = [regrade_one(qr) for qr in record.questions]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            new_qs = list(pool.map(regrade_one, record.questions))

    n_correct = sum(1 for q in new_qs if q.correct)
    n = len(new_qs)
    return replace(
        record,
        questions=new_qs,
        n_evaluated=n,
        accuracy=(n_correct / n) if n else 0.0,
    )


def _cell_from_record_path(record_path: Path) -> CellSpec:
    raw = json.loads(record_path.read_text(encoding="utf-8"))
    return CellSpec(**raw["cell"])


def _rebuild_summary(cell_dir_path: Path, n_questions_default: int) -> None:
    sa_path = cell_dir_path / "sa.json"
    voting_path = cell_dir_path / "voting.json"
    debate_path = cell_dir_path / "debate.json"
    calib_path = cell_dir_path / "sc_calibration.json"
    summary_path = cell_dir_path / "summary.json"

    sa = load_record(sa_path)
    voting = load_record(voting_path)
    debate = load_record(debate_path)
    calib = (json.loads(calib_path.read_text()) if calib_path.exists()
             else {})

    n_q = max(len(sa.questions), len(voting.questions),
              len(debate.questions), n_questions_default)

    summary = CellSummary(
        cell=sa.cell,
        sa_accuracy=sa.accuracy,
        voting_accuracy=voting.accuracy,
        debate_accuracy=debate.accuracy,
        sa_total_tokens=sa.total_tokens,
        voting_total_tokens=voting.total_tokens,
        debate_total_tokens=debate.total_tokens,
        delta_debate_voting=debate.accuracy - voting.accuracy,
        delta_debate_sa=debate.accuracy - sa.accuracy,
        calibration=calib,
        n_questions=n_q,
    )
    save_summary(summary_path, summary)


def rejudge_cell(
    cell_dir_path: Path,
    judge: Judge,
    timeout_s: float = 8.0,
    max_workers: int = 4,
) -> dict:
    """Re-judge all condition files in cell_dir; rewrite summary.

    Returns a dict {sa,voting,debate} -> new accuracy.
    """
    if not cell_dir_path.is_dir():
        raise FileNotFoundError(f"cell dir not found: {cell_dir_path}")
    sa_path = cell_dir_path / "sa.json"
    if not sa_path.exists():
        raise FileNotFoundError(f"missing {sa_path}")
    cell = _cell_from_record_path(sa_path)
    if cell.dataset_name not in CODE_DATASETS:
        raise ValueError(
            f"cell {cell_dir_path.name} dataset_name="
            f"{cell.dataset_name!r} is not a code dataset; rejudge "
            "doesn't apply"
        )

    dataset = _build_dataset(cell)
    questions = dataset.load()
    qid_to_q = {q.qid: q for q in questions}

    out: dict[str, float] = {}
    for fname in CONDITION_FILES:
        path = cell_dir_path / fname
        if not path.exists():
            continue
        record = load_record(path)
        new_record = _rejudge_record(
            record, qid_to_q, judge, timeout_s, max_workers,
        )
        save_record(path, new_record)
        out[fname.removesuffix(".json")] = new_record.accuracy

    _rebuild_summary(cell_dir_path, n_questions_default=40)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-judge an existing code grid cell artifact dir.",
    )
    parser.add_argument("--cell-dir", action="append", required=True,
                        type=Path,
                        help="Path to artifacts/grid/<cell>/ subtree. "
                             "May be passed multiple times.")
    parser.add_argument("--timeout-s", type=float, default=8.0,
                        help="Per-test timeout (default 8s)")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Concurrent test executions (default 4)")
    args = parser.parse_args(argv)

    judge = SubprocessJudge()
    for cdir in args.cell_dir:
        print(f"[rejudge] {cdir}", file=sys.stderr)
        try:
            res = rejudge_cell(cdir, judge,
                               timeout_s=args.timeout_s,
                               max_workers=args.max_workers)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {exc}", file=sys.stderr)
            continue
        msg = "  " + " ".join(f"{k}={v:.3f}" for k, v in res.items())
        print(msg, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
