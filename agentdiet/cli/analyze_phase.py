"""Phase-mapping analysis CLI.

One-shot driver that reads grid_dir/, computes per-cell Delta CIs,
renders the phase diagram, runs every characterization helper, and
writes the consolidated analysis.json + per-table .tex snippets.

Usage on a host with grid artifacts:
    python -m agentdiet.cli.analyze_phase
    python -m agentdiet.cli.analyze_phase --grid-dir artifacts/grid \\
                                           --output-dir docs/reports \\
                                           --case-study-cell Qwen__Qwen3-30B-A3B__gsm8k__t0
    python -m agentdiet.cli.analyze_phase --skip-figure   # CI without matplotlib

Note: the older agentdiet.cli.analyze (flip events + claim signals,
RQ0 chain) is unrelated and lives at a different module name.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentdiet.analysis_phase.bootstrap import (
    CellAnalysis,
    compute_per_cell_analysis,
)
from agentdiet.analysis_phase.characterize import (
    aime_per_year,
    cross_model_agreement,
    thinking_axis_observations,
    voting_wrong_debate_right,
)


DEFAULT_GRID_DIR = Path("artifacts/grid")
DEFAULT_OUTPUT_DIR = Path("docs/reports")


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _cell_to_row(cell: CellAnalysis) -> dict:
    """Compact row form for analysis.json (drops per-q vectors to keep it small)."""
    return {
        "cell_dirname": cell.cell_dirname,
        "model": cell.model,
        "model_family": cell.model_family,
        "dataset_name": cell.dataset_name,
        "thinking": cell.thinking,
        "prompt_variant": cell.prompt_variant,
        "sa_accuracy": cell.sa_accuracy,
        "voting_accuracy": cell.voting_accuracy,
        "debate_accuracy": cell.debate_accuracy,
        "delta_debate_voting": cell.delta_debate_voting,
        "delta_ci_low": cell.delta_ci_low,
        "delta_ci_high": cell.delta_ci_high,
        "over_budget_factor": cell.over_budget_factor,
        "n_questions": cell.n_questions,
    }


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# LaTeX table rendering
# ---------------------------------------------------------------------------


def _tex_escape(s: Any) -> str:
    return (str(s)
            .replace("\\", r"\textbackslash{}")
            .replace("_", r"\_")
            .replace("%", r"\%")
            .replace("&", r"\&"))


def render_phase_summary_tex(cells: list[dict]) -> str:
    header = (
        "\\begin{tabular}{lllrrrrr}\n"
        "\\toprule\n"
        "Cell & Dataset & Thinking & SA & Voting & Debate & "
        "$\\Delta$ & 95\\% CI \\\\\n"
        "\\midrule\n"
    )
    body = ""
    for r in cells:
        thinking = "on" if r["thinking"] else "off"
        ci = f"[{r['delta_ci_low']:+.3f}, {r['delta_ci_high']:+.3f}]"
        body += (
            f"{_tex_escape(r['model_family'])} & "
            f"{_tex_escape(r['dataset_name'])} & "
            f"{thinking} & "
            f"{r['sa_accuracy']:.3f} & "
            f"{r['voting_accuracy']:.3f} & "
            f"{r['debate_accuracy']:.3f} & "
            f"{r['delta_debate_voting']:+.3f} & "
            f"{ci} \\\\\n"
        )
    return header + body + "\\bottomrule\n\\end{tabular}\n"


def render_aime_per_year_tex(rows: list[dict]) -> str:
    if not rows:
        return ("\\begin{tabular}{l}\n\\toprule\nAIME data not available "
                "in this run\\\\\n\\bottomrule\n\\end{tabular}\n")
    header = (
        "\\begin{tabular}{llcrrrrr}\n"
        "\\toprule\n"
        "Model & Variant & Year & n & SA & Voting & Debate & $\\Delta$ \\\\\n"
        "\\midrule\n"
    )
    body = ""
    for r in rows:
        body += (
            f"{_tex_escape(r['model'])} & "
            f"{_tex_escape(r['prompt_variant'])} & "
            f"{r['year']} & {r['n']} & "
            f"{r['sa_acc']:.3f} & {r['voting_acc']:.3f} & "
            f"{r['debate_acc']:.3f} & "
            f"{r['delta_debate_voting']:+.3f} \\\\\n"
        )
    return header + body + "\\bottomrule\n\\end{tabular}\n"


def render_cross_model_tex(rows: list[dict]) -> str:
    if not rows:
        return ("\\begin{tabular}{l}\n\\toprule\nNo cross-model groups "
                "available\\\\\n\\bottomrule\n\\end{tabular}\n")
    header = (
        "\\begin{tabular}{llllcc}\n"
        "\\toprule\n"
        "Dataset & Thinking & Variant & Families & Signs agree & CIs overlap \\\\\n"
        "\\midrule\n"
    )
    body = ""
    for r in rows:
        thinking = "on" if r["thinking"] else "off"
        families = ", ".join(_tex_escape(f) for f in r["families"])
        body += (
            f"{_tex_escape(r['dataset'])} & "
            f"{thinking} & "
            f"{_tex_escape(r['prompt_variant'])} & "
            f"{families} & "
            f"{'yes' if r['signs_agree'] else 'no'} & "
            f"{'yes' if r['cis_overlap'] else 'no'} \\\\\n"
        )
    return header + body + "\\bottomrule\n\\end{tabular}\n"


def render_thinking_o1_tex(rows: list[dict]) -> str:
    if not rows:
        return ("\\begin{tabular}{l}\n\\toprule\nNo paired thinking on/off "
                "cells available\\\\\n\\bottomrule\n\\end{tabular}\n")
    header = (
        "\\begin{tabular}{lllrrr}\n"
        "\\toprule\n"
        "Model & Dataset & Variant & "
        "$\\Delta$ off & $\\Delta$ on & $\\Delta$ change \\\\\n"
        "\\midrule\n"
    )
    body = ""
    for r in rows:
        body += (
            f"{_tex_escape(r['model_family'])} & "
            f"{_tex_escape(r['dataset'])} & "
            f"{_tex_escape(r['prompt_variant'])} & "
            f"{r['delta_off']:+.3f} & "
            f"{r['delta_on']:+.3f} & "
            f"{r['delta_change']:+.3f} \\\\\n"
        )
    return header + body + "\\bottomrule\n\\end{tabular}\n"


def render_thinking_o2_tex(rows: list[dict]) -> str:
    if not rows:
        return ("\\begin{tabular}{l}\n\\toprule\nNo paired thinking on/off "
                "cells available\\\\\n\\bottomrule\n\\end{tabular}\n")
    header = (
        "\\begin{tabular}{lllrrll}\n"
        "\\toprule\n"
        "Model & Dataset & Variant & "
        "Debate(off) & SA(on) & Winner & $|\\text{diff}|$ \\\\\n"
        "\\midrule\n"
    )
    body = ""
    for r in rows:
        body += (
            f"{_tex_escape(r['model_family'])} & "
            f"{_tex_escape(r['dataset'])} & "
            f"{_tex_escape(r['prompt_variant'])} & "
            f"{r['debate_thinking_off']:.3f} & "
            f"{r['sa_thinking_on']:.3f} & "
            f"{_tex_escape(r['winner'])} & "
            f"{r['magnitude']:.3f} \\\\\n"
        )
    return header + body + "\\bottomrule\n\\end{tabular}\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase-mapping analysis: per-cell Delta CI + figure + "
                    "characterizations + LaTeX tables.",
    )
    parser.add_argument("--grid-dir", type=Path, default=DEFAULT_GRID_DIR,
                        help="Directory with per-cell artifact subtrees")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Where to write data/, figs/, tables/ subdirs")
    parser.add_argument("--n-resamples", type=int, default=10000,
                        help="Bootstrap resamples per cell (default 10000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Bootstrap RNG seed (default 42)")
    parser.add_argument("--ci", type=float, default=0.95,
                        help="CI width (default 0.95)")
    parser.add_argument("--skip-figure", action="store_true",
                        help="Skip phase diagram rendering (no matplotlib)")
    parser.add_argument("--case-study-cell", type=str, default=None,
                        help="Cell directory name to harvest "
                             "voting-wrong / debate-right qids for")
    args = parser.parse_args(argv)

    grid_dir: Path = args.grid_dir
    out_dir: Path = args.output_dir

    if not grid_dir.is_dir():
        print(f"ERROR: grid_dir does not exist: {grid_dir}", file=sys.stderr)
        return 2

    cells = compute_per_cell_analysis(
        grid_dir,
        n_resamples=args.n_resamples,
        seed=args.seed,
        ci=args.ci,
    )
    if not cells:
        print(
            f"ERROR: no complete cell directories found under {grid_dir}",
            file=sys.stderr,
        )
        return 2

    # Per-axis analyses
    cell_rows = [_cell_to_row(c) for c in cells]
    aime_rows = aime_per_year(cells)
    cross_rows = cross_model_agreement(cells)
    thinking = thinking_axis_observations(cells)

    # JSON consolidation
    payload = {
        "meta": {
            "grid_dir": str(grid_dir),
            "n_resamples": args.n_resamples,
            "seed": args.seed,
            "ci": args.ci,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "cells": cell_rows,
        "aime": aime_rows,
        "cross_model": cross_rows,
        "thinking_o1": thinking["o1"],
        "thinking_o2": thinking["o2"],
    }
    data_path = out_dir / "data" / "analysis.json"
    _atomic_write(data_path, json.dumps(payload, indent=2, ensure_ascii=False))

    # LaTeX tables
    tables_dir = out_dir / "tables"
    _atomic_write(tables_dir / "phase_summary.tex",
                  render_phase_summary_tex(cell_rows))
    _atomic_write(tables_dir / "aime_per_year.tex",
                  render_aime_per_year_tex(aime_rows))
    _atomic_write(tables_dir / "cross_model.tex",
                  render_cross_model_tex(cross_rows))
    _atomic_write(tables_dir / "thinking_o1.tex",
                  render_thinking_o1_tex(thinking["o1"]))
    _atomic_write(tables_dir / "thinking_o2.tex",
                  render_thinking_o2_tex(thinking["o2"]))

    # Phase diagram figure (optional)
    if not args.skip_figure:
        try:
            from agentdiet.analysis_phase.figure_phase_diagram import (
                render_phase_diagram,
            )
            figs_dir = out_dir / "figs"
            figs_dir.mkdir(parents=True, exist_ok=True)
            render_phase_diagram(cells, figs_dir / "phase_diagram.pdf")
            print(f"figure   -> {figs_dir / 'phase_diagram.pdf'}",
                  file=sys.stderr)
        except ImportError as exc:
            print(f"WARNING: figure skipped — {exc}", file=sys.stderr)

    # Optional case study
    if args.case_study_cell:
        cs_rows = voting_wrong_debate_right(grid_dir, args.case_study_cell)
        cs_path = out_dir / "data" / "voting_wrong_debate_right.json"
        _atomic_write(cs_path, json.dumps({
            "cell": args.case_study_cell,
            "rows": cs_rows,
        }, indent=2))
        print(f"case-study -> {cs_path} ({len(cs_rows)} rows)",
              file=sys.stderr)

    # Summary
    print(f"analyzed {len(cells)} cell(s) under {grid_dir}", file=sys.stderr)
    print(f"data     -> {data_path}", file=sys.stderr)
    print(f"tables   -> {tables_dir}/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
