"""Per-axis characterization helpers (spec §6 F2 + F3).

Pure functions returning plain dicts/lists. The CLI composes them into
the final analysis.json + LaTeX tables.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

from agentdiet.analysis_phase.bootstrap import (
    CellAnalysis,
    paired_bootstrap_delta,
)


# ---------------------------------------------------------------------------
# AIME per-year stratification (spec §4.2.1, §6 F2)
# ---------------------------------------------------------------------------


def _aime_year_from_qid(qid: str) -> Optional[int]:
    """qid format from AIMEMultiYearDataset: 'aime-<year>-q<idx>'."""
    parts = qid.split("-")
    if len(parts) >= 2 and parts[0] == "aime":
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def aime_per_year(cells: list[CellAnalysis]) -> list[dict]:
    """For each AIME cell, partition per-question correctness by year
    and recompute SA / voting / debate accuracy + delta per year."""
    rows: list[dict] = []
    for cell in cells:
        if cell.dataset_name != "aime":
            continue
        by_year: dict[int, list[int]] = defaultdict(list)
        for i, qid in enumerate(cell.qids):
            year = _aime_year_from_qid(qid)
            if year is None:
                continue
            by_year[year].append(i)

        for year, indices in sorted(by_year.items()):
            n = len(indices)
            if n == 0:
                continue
            sa_acc = sum(cell.sa_per_q_correct[i] for i in indices) / n
            voting_acc = sum(cell.voting_per_q_correct[i] for i in indices) / n
            debate_acc = sum(cell.debate_per_q_correct[i] for i in indices) / n
            rows.append({
                "cell_dirname": cell.cell_dirname,
                "model": cell.model,
                "thinking": cell.thinking,
                "prompt_variant": cell.prompt_variant,
                "year": year,
                "n": n,
                "sa_acc": sa_acc,
                "voting_acc": voting_acc,
                "debate_acc": debate_acc,
                "delta_debate_voting": debate_acc - voting_acc,
            })
    return rows


# ---------------------------------------------------------------------------
# Cross-model agreement (spec §6 F2)
# ---------------------------------------------------------------------------


def _ci_overlap(lo_a: float, hi_a: float, lo_b: float, hi_b: float) -> bool:
    return not (hi_a < lo_b or hi_b < lo_a)


def cross_model_agreement(cells: list[CellAnalysis]) -> list[dict]:
    """Group cells by (dataset, thinking, prompt_variant). For each group
    with >=2 distinct model_family entries, report whether the deltas
    agree on sign and whether their CIs overlap."""
    by_group: dict[tuple, list[CellAnalysis]] = defaultdict(list)
    for cell in cells:
        key = (cell.dataset_name, cell.thinking, cell.prompt_variant)
        by_group[key].append(cell)

    rows: list[dict] = []
    for (dataset, thinking, variant), group in sorted(by_group.items()):
        families = {c.model_family for c in group}
        if len(families) < 2:
            continue
        # Build per-family lookup (one entry per model_family in the group)
        by_family: dict[str, CellAnalysis] = {}
        for c in group:
            # If duplicate family, keep first
            by_family.setdefault(c.model_family, c)
        if len(by_family) < 2:
            continue

        family_list = sorted(by_family.keys())
        deltas = {f: by_family[f].delta_debate_voting for f in family_list}
        ci_low = {f: by_family[f].delta_ci_low for f in family_list}
        ci_high = {f: by_family[f].delta_ci_high for f in family_list}

        # Pairwise: with only 2 families this is one comparison
        f1, f2 = family_list[0], family_list[1]
        signs_agree = (deltas[f1] >= 0) == (deltas[f2] >= 0)
        cis_overlap = _ci_overlap(ci_low[f1], ci_high[f1],
                                   ci_low[f2], ci_high[f2])

        rows.append({
            "dataset": dataset,
            "thinking": thinking,
            "prompt_variant": variant,
            "families": family_list,
            "deltas": deltas,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "signs_agree": signs_agree,
            "cis_overlap": cis_overlap,
        })
    return rows


# ---------------------------------------------------------------------------
# Reverse-attribution: voting wrong & debate right (case study)
# ---------------------------------------------------------------------------


def voting_wrong_debate_right(
    grid_dir: Path,
    cell_dirname: str,
    max_results: int = 5,
) -> list[dict]:
    """Find qids where voting was wrong but debate was right. Return up
    to max_results rows with response previews from voting + debate."""
    cell_path = Path(grid_dir) / cell_dirname
    voting = json.loads((cell_path / "voting.json").read_text(encoding="utf-8"))
    debate = json.loads((cell_path / "debate.json").read_text(encoding="utf-8"))

    voting_qs = {q["qid"]: q for q in voting.get("questions", [])}
    debate_qs = {q["qid"]: q for q in debate.get("questions", [])}

    rows: list[dict] = []
    for qid, dq in debate_qs.items():
        if qid not in voting_qs:
            continue
        vq = voting_qs[qid]
        if vq.get("correct") is False and dq.get("correct") is True:
            rows.append({
                "qid": qid,
                "gold": dq.get("gold"),
                "voting_final_answer": vq.get("final_answer"),
                "debate_final_answer": dq.get("final_answer"),
                "voting_total_tokens": vq.get("total_tokens"),
                "debate_total_tokens": dq.get("total_tokens"),
            })
            if len(rows) >= max_results:
                break
    return rows


# ---------------------------------------------------------------------------
# Thinking-axis observations O1 + O2 (spec §6 F3)
# ---------------------------------------------------------------------------


def thinking_axis_observations(cells: list[CellAnalysis]) -> dict:
    """O1: per (model_family, dataset, prompt_variant), pair thinking on
    vs off cells; report delta_on - delta_off + per-side CIs.

    O2: same grouping, compare acc(debate, off) vs acc(SA, on); report
    which is larger and by how much.

    Skip groups missing either side cleanly."""
    by_group: dict[tuple, dict[bool, CellAnalysis]] = defaultdict(dict)
    for cell in cells:
        key = (cell.model_family, cell.dataset_name, cell.prompt_variant)
        by_group[key][cell.thinking] = cell

    o1_rows: list[dict] = []
    o2_rows: list[dict] = []
    for (family, dataset, variant), pair in sorted(by_group.items()):
        on = pair.get(True)
        off = pair.get(False)
        if on is None or off is None:
            continue

        o1_rows.append({
            "model_family": family,
            "dataset": dataset,
            "prompt_variant": variant,
            "delta_off": off.delta_debate_voting,
            "delta_on": on.delta_debate_voting,
            "delta_change": on.delta_debate_voting - off.delta_debate_voting,
            "off_ci_low": off.delta_ci_low,
            "off_ci_high": off.delta_ci_high,
            "on_ci_low": on.delta_ci_low,
            "on_ci_high": on.delta_ci_high,
        })

        debate_off = off.debate_accuracy
        sa_on = on.sa_accuracy
        o2_rows.append({
            "model_family": family,
            "dataset": dataset,
            "prompt_variant": variant,
            "debate_thinking_off": debate_off,
            "sa_thinking_on": sa_on,
            "winner": "debate_off" if debate_off > sa_on else (
                "sa_on" if sa_on > debate_off else "tie"
            ),
            "magnitude": abs(debate_off - sa_on),
        })
    return {"o1": o1_rows, "o2": o2_rows}
