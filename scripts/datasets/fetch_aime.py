"""Fetch AIME 2024 / 2025 / 2026 from HF and normalize to the schema
expected by agentdiet.eval.datasets.AIMEMultiYearDataset.

Sources (chosen for max coverage as of 2026-04-30):
  - 2024: Maxwell-Jia/AIME_2024     (cols: ID, Problem, Solution, Answer)
  - 2025: MathArena/aime_2025        (cols: problem_idx, problem, answer, problem_type)
  - 2026: MathArena/aime_2026        (cols: problem_idx, answer, problem)

Output:
  artifacts/datasets/aime_{2024,2025,2026}.json — each with schema:
    {"year": int, "questions": [{"id": str, "problem": str, "answer": str}, ...]}

Run from repo root:
    python scripts/datasets/fetch_aime.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "artifacts" / "datasets"


def _normalize_2024(row: dict, idx: int) -> dict:
    """Maxwell-Jia/AIME_2024 row -> normalized dict."""
    qid = str(row.get("ID") or f"2024-{idx + 1}")
    return {
        "id": qid,
        "problem": str(row["Problem"]).strip(),
        "answer": str(row["Answer"]).strip(),
    }


def _normalize_matharena(row: dict, year: int, idx: int) -> dict:
    """MathArena/aime_{2025,2026} row -> normalized dict."""
    pi = row.get("problem_idx")
    qid = f"{year}-{int(pi):02d}" if pi is not None else f"{year}-{idx + 1:02d}"
    return {
        "id": qid,
        "problem": str(row["problem"]).strip(),
        "answer": str(row["answer"]).strip(),
    }


SOURCES = [
    {
        "year": 2024,
        "hf_id": "Maxwell-Jia/AIME_2024",
        "split": "train",
        "normalize": _normalize_2024,
    },
    {
        "year": 2025,
        "hf_id": "MathArena/aime_2025",
        "split": "train",
        "normalize": lambda row, idx: _normalize_matharena(row, 2025, idx),
    },
    {
        "year": 2026,
        "hf_id": "MathArena/aime_2026",
        "split": "train",
        "normalize": lambda row, idx: _normalize_matharena(row, 2026, idx),
    },
]


def fetch_year(spec: dict, output_dir: Path) -> dict:
    from datasets import load_dataset
    print(f"[fetch] {spec['hf_id']} (year={spec['year']})", file=sys.stderr)
    ds = load_dataset(spec["hf_id"], split=spec["split"])
    questions = [spec["normalize"](row, i) for i, row in enumerate(ds)]
    if len(questions) != 30:
        raise RuntimeError(
            f"Year {spec['year']} from {spec['hf_id']} has {len(questions)} "
            "questions, expected 30"
        )
    payload = {"year": spec["year"], "questions": questions}
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"aime_{spec['year']}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"  -> {out_path} ({len(questions)} questions)", file=sys.stderr)
    return payload


def main(argv: list[str] | None = None) -> int:
    output_dir = OUTPUT_DIR
    if argv:
        for arg in argv:
            if arg.startswith("--output="):
                output_dir = Path(arg.split("=", 1)[1])

    for spec in SOURCES:
        fetch_year(spec, output_dir)

    # Final sanity: load via AIMEMultiYearDataset and confirm 80 total
    from agentdiet.eval.datasets import AIMEMultiYearDataset
    qs = AIMEMultiYearDataset(data_dir=output_dir).load()
    print(f"[verify] AIMEMultiYearDataset loaded {len(qs)} questions",
          file=sys.stderr)
    by_year: dict[int, int] = {}
    for q in qs:
        year = int(q.qid.split("-")[1])
        by_year[year] = by_year.get(year, 0) + 1
    print(f"[verify] year breakdown: {dict(sorted(by_year.items()))}",
          file=sys.stderr)
    if len(qs) != 80:
        print("ERROR: expected 80 questions after stratification", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
