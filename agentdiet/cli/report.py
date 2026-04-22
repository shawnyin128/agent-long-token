"""Report-generation CLI.

Reads ablation + evaluation + claim + signal artifacts, writes
figures (if matplotlib is available) and LaTeX tables to
``docs/reports/{figs,tables}/``. Leaves PDF assembly to the author
(``pandoc final-report.template.md -o final-report.pdf``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from agentdiet import report
from agentdiet.config import Config, get_config


DEFAULT_REPORTS_DIR = Path("docs/reports")


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_all_claims(cfg: Config) -> list[dict]:
    claims: list[dict] = []
    for p in cfg.claims_dir.glob("*.json"):
        doc = json.loads(p.read_text(encoding="utf-8"))
        claims.extend(doc.get("claims", []))
    return claims


def _collect_signal_rows(cfg: Config) -> list[dict]:
    path = cfg.analysis_dir / "signal_scores.parquet"
    if not path.exists():
        return []
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError:
        return []
    return pq.read_table(path).to_pylist()


def run_report_cli(
    *, cfg: Config, reports_dir: Path,
) -> dict:
    reports_dir = Path(reports_dir)
    figs_dir = reports_dir / "figs"
    tables_dir = reports_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    ablation_summary = _load_json(cfg.analysis_dir / "ablation_summary.json")
    evaluation_results = _load_json(cfg.evaluation_dir / "results.json")
    claims = _collect_all_claims(cfg)
    signal_rows = _collect_signal_rows(cfg)

    # Tables.
    tables_written: list[str] = []
    if evaluation_results is not None:
        (tables_dir / "baselines.tex").write_text(
            report.render_table_baselines(evaluation_results), encoding="utf-8"
        )
        tables_written.append("baselines.tex")
    if claims:
        (tables_dir / "claim_stats.tex").write_text(
            report.render_table_claim_stats(claims), encoding="utf-8"
        )
        tables_written.append("claim_stats.tex")

    # Figures (skip if matplotlib unavailable).
    figures_written: list[str] = []
    figures_skipped = False
    try:
        report._ensure_matplotlib()
    except ImportError as e:
        print(f"WARNING: {e} — skipping figures", file=sys.stderr)
        figures_skipped = True

    if not figures_skipped:
        figs_dir.mkdir(parents=True, exist_ok=True)
        if claims:
            report.render_figure_type_distribution(
                report.claim_type_distribution(claims),
                figs_dir / "claim_type_distribution.png",
            )
            figures_written.append("claim_type_distribution.png")
        if signal_rows:
            report.render_figure_signal_correlations(
                report.signal_flip_correlations(signal_rows),
                figs_dir / "signal_correlations.png",
            )
            figures_written.append("signal_correlations.png")
        if ablation_summary is not None:
            report.render_figure_delta_ranking(
                report.delta_ranking(ablation_summary),
                figs_dir / "delta_ranking.png",
            )
            figures_written.append("delta_ranking.png")
        if evaluation_results is not None:
            report.render_figure_pareto(
                report.pareto_data(evaluation_results),
                figs_dir / "pareto.png",
            )
            figures_written.append("pareto.png")

    return {
        "tables": tables_written,
        "figures": figures_written,
        "figures_skipped": figures_skipped,
        "reports_dir": str(reports_dir),
    }


def main(
    argv: list[str] | None = None,
    *,
    cfg: Config | None = None,
    reports_dir: Path | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Render figures + LaTeX tables from pipeline artifacts"
    )
    parser.add_argument("--reports-dir", type=Path, default=None,
                        help="Output dir (default docs/reports)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate artifacts without writing")
    args = parser.parse_args(argv)

    if cfg is None:
        cfg = get_config()
    cfg.ensure_dirs()
    target = reports_dir or args.reports_dir or DEFAULT_REPORTS_DIR

    if args.dry_run:
        print("dry-run OK")
        print(f"  reports_dir       = {target}")
        print(f"  ablation_summary  = {cfg.analysis_dir / 'ablation_summary.json'}")
        print(f"  evaluation        = {cfg.evaluation_dir / 'results.json'}")
        print(f"  claims_dir        = {cfg.claims_dir}")
        return 0

    have_ablation = (cfg.analysis_dir / "ablation_summary.json").exists()
    have_evaluation = (cfg.evaluation_dir / "results.json").exists()
    if not have_ablation and not have_evaluation:
        print("ERROR: need at least ablation_summary.json or results.json "
              "— run `make ablate` and/or `make evaluate` first",
              file=sys.stderr)
        return 2

    out = run_report_cli(cfg=cfg, reports_dir=target)
    print(f"wrote tables: {out['tables']}", file=sys.stderr)
    print(f"wrote figures: {out['figures']}", file=sys.stderr)
    if out["figures_skipped"]:
        print("figures were skipped — install `.[analysis]` for matplotlib",
              file=sys.stderr)
    print(f"next: edit {target}/final-report.template.md, then "
          f"`pandoc final-report.template.md -o final-report.pdf`",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
