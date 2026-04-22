"""Report-generation helpers.

Pure-Python data preparation + LaTeX table rendering. Figure renderers
lazy-import matplotlib so the module is importable without the
``[analysis]`` extras.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Data preparation (numpy only)
# ---------------------------------------------------------------------------

def claim_type_distribution(claims: list[dict]) -> dict[tuple[int, int, str], int]:
    counts: dict[tuple[int, int, str], int] = {}
    for c in claims:
        key = (int(c["agent_id"]), int(c["round"]), str(c["type"]))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _pearson_safe(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r with zero-variance fallback to 0.0."""
    if len(x) < 2:
        return 0.0
    sx = np.std(x)
    sy = np.std(y)
    if sx == 0 or sy == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def signal_flip_correlations(signal_rows: list[dict]) -> dict[str, float]:
    if not signal_rows:
        return {"novelty": 0.0, "referenced_later": 0.0, "position": 0.0}
    flip = np.array([float(bool(r["flip_coincidence"])) for r in signal_rows])
    novelty = np.array([float(r["novelty"]) for r in signal_rows])
    ref = np.array([float(bool(r["referenced_later"])) for r in signal_rows])
    pos = np.array([float(r["position"]) for r in signal_rows])
    return {
        "novelty": _pearson_safe(novelty, flip),
        "referenced_later": _pearson_safe(ref, flip),
        "position": _pearson_safe(pos, flip),
    }


def delta_ranking(ablation_summary: dict) -> list[dict]:
    rows = list(ablation_summary.get("per_type", []))
    return sorted(rows, key=lambda r: (-abs(float(r["delta"])), r["type"]))


def pareto_data(evaluation_results: dict) -> list[dict]:
    out: list[dict] = []
    for m in evaluation_results.get("per_method", []):
        out.append({
            "method": m["method"],
            "accuracy": float(m["accuracy"]),
            "total_tokens": int(m["total_tokens"]),
        })
    return out


# ---------------------------------------------------------------------------
# LaTeX tables (pure string templating)
# ---------------------------------------------------------------------------

def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def render_table_baselines(evaluation_results: dict) -> str:
    rows = evaluation_results.get("per_method", [])
    header = (
        "\\begin{tabular}{lrrr}\n"
        "\\toprule\n"
        "Method & Accuracy & Total tokens & Acc/1k \\\\\n"
        "\\midrule\n"
    )
    body = "".join(
        f"{_tex_escape(r['method'])} & {r['accuracy']:.3f} & "
        f"{r['total_tokens']:d} & {r['acc_per_1k']:.3f} \\\\\n"
        for r in rows
    )
    footer = "\\bottomrule\n\\end{tabular}\n"
    return header + body + footer


def render_table_claim_stats(claims: list[dict]) -> str:
    counts: dict[str, int] = {}
    for c in claims:
        t = str(c["type"])
        counts[t] = counts.get(t, 0) + 1
    total = sum(counts.values()) or 1  # avoid zero-div
    header = (
        "\\begin{tabular}{lrr}\n"
        "\\toprule\n"
        "Type & Count & \\% \\\\\n"
        "\\midrule\n"
    )
    body = "".join(
        f"{_tex_escape(t)} & {n:d} & {100.0 * n / total:.1f} \\\\\n"
        for t, n in sorted(counts.items())
    )
    footer = "\\bottomrule\n\\end{tabular}\n"
    return header + body + footer


# ---------------------------------------------------------------------------
# Figure renderers (lazy matplotlib)
# ---------------------------------------------------------------------------

def _ensure_matplotlib():
    try:
        import matplotlib  # noqa: F401
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "matplotlib not installed; run `pip install -e .[analysis]`"
        ) from e


def render_figure_type_distribution(dist: dict, out_path: Path) -> Path:
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    # Aggregate counts per type across all (agent, round).
    per_type: dict[str, int] = {}
    for (_agent, _round, t), n in dist.items():
        per_type[t] = per_type.get(t, 0) + n
    types = sorted(per_type.keys())
    counts = [per_type[t] for t in types]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(types, counts)
    ax.set_xlabel("Claim type")
    ax.set_ylabel("Count")
    ax.set_title("Claim type distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def render_figure_signal_correlations(corrs: dict, out_path: Path) -> Path:
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    names = sorted(corrs.keys())
    values = [corrs[n] for n in names]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(names, values)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("Correlation with flip_coincidence")
    ax.set_title("Per-signal correlation with flip path")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def render_figure_delta_ranking(ranking: list[dict], out_path: Path) -> Path:
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    types = [r["type"] for r in ranking]
    deltas = [float(r["delta"]) for r in ranking]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(types[::-1], deltas[::-1])
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Δ accuracy (with − without)")
    ax.set_title("Per-type causal importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def render_figure_pareto(points: list[dict], out_path: Path) -> Path:
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for p in points:
        ax.scatter(p["total_tokens"], p["accuracy"], label=p["method"])
        ax.annotate(p["method"], (p["total_tokens"], p["accuracy"]),
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Total input tokens")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs tokens (Pareto)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
