"""Type-level ablation CLI (Gate 2).

Selects a ``single_wrong ∧ debate_right`` subset, replays the final
round for each (qid, claim_type) pair with that type's claims
span-masked out of history, and emits:

  * ``artifacts/analysis/ablation.jsonl`` — one row per (qid, type)
  * ``artifacts/analysis/ablation_summary.json`` — per-type aggregates
  * ``artifacts/analysis/ablation_manifest.json`` — run metadata
  * ``artifacts/analysis/gate2_report.md`` (on ``--gate2``)

A hard ``--max-calls`` cap enforces spec §6.1's 500-call budget.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from agentdiet.analysis.ablate import (
    MAX_NEW_LLM_CALLS_DEFAULT, run_ablation, run_control_ablation,
    select_subset,
)
from agentdiet.config import Config, get_config
from agentdiet.llm_client import Backend, LLMClient, OpenAIBackend
from agentdiet.types import CLAIM_TYPES


# Gate-2 thresholds (D3).
GATE2_LIKELY_THRESHOLD = 0.10
GATE2_NOISE_THRESHOLD = 0.03
GATE2_PASS_THRESHOLD = 0.05


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _jsonl_path(cfg: Config) -> Path:
    return cfg.analysis_dir / "ablation.jsonl"


def _summary_path(cfg: Config) -> Path:
    return cfg.analysis_dir / "ablation_summary.json"


def _manifest_path(cfg: Config) -> Path:
    return cfg.analysis_dir / "ablation_manifest.json"


def _gate2_path(cfg: Config) -> Path:
    return cfg.analysis_dir / "gate2_report.md"


def _control_path(cfg: Config) -> Path:
    return cfg.analysis_dir / "control_result.json"


def _make_client(cfg: Config, backend: Backend | None = None) -> LLMClient:
    if backend is None:
        backend = OpenAIBackend(
            base_url=cfg.base_url, api_key=cfg.api_key, timeout_s=cfg.request_timeout_s
        )
    return LLMClient(backend, cache_path=cfg.cache_path, max_retries=cfg.max_retries)


def _summarize(rows: list[dict]) -> dict:
    per_type: list[dict] = []
    for t in CLAIM_TYPES:
        entries = [r for r in rows if r.get("drop_type") == t]
        used = [r for r in entries if not r.get("skipped")]
        skipped = len(entries) - len(used)
        if used:
            acc_with = sum(1 for r in used if r["correct_with"]) / len(used)
            acc_without = sum(1 for r in used if r["correct_without"]) / len(used)
            delta = acc_with - acc_without
        else:
            acc_with = 0.0
            acc_without = 0.0
            delta = 0.0
        per_type.append({
            "type": t,
            "n_used": len(used),
            "n_skipped": skipped,
            "acc_with": acc_with,
            "acc_without": acc_without,
            "delta": delta,
        })
    return {"per_type": per_type}


def _classify_delta(delta: float) -> str:
    abs_d = abs(delta)
    if abs_d >= GATE2_LIKELY_THRESHOLD:
        return "likely"
    if abs_d <= GATE2_NOISE_THRESHOLD:
        return "noise"
    return "unclear"


def _render_gate2(summary: dict) -> tuple[str, int]:
    """Return (markdown_report, exit_code).

    Exit 0 = PASS (at least one type's |delta| >= PASS_THRESHOLD).
    Exit 10 = NULL_RESULT (all types within NOISE_THRESHOLD).
    Exit 20 = INCONCLUSIVE (between thresholds).
    """
    rows = summary["per_type"]
    rows_sorted = sorted(rows, key=lambda r: -abs(r["delta"]))
    lines = [
        "# Gate-2 — Type-level ablation decision",
        "",
        "| Type | n | Δ | |Δ| | acc(with) | acc(without) | class |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    max_abs = 0.0
    all_in_noise = True
    for r in rows_sorted:
        cls = _classify_delta(r["delta"])
        max_abs = max(max_abs, abs(r["delta"]))
        if abs(r["delta"]) > GATE2_NOISE_THRESHOLD:
            all_in_noise = False
        lines.append(
            f"| {r['type']} | {r['n_used']} | {r['delta']:+.3f} "
            f"| {abs(r['delta']):.3f} | {r['acc_with']:.2f} | "
            f"{r['acc_without']:.2f} | {cls} |"
        )

    if max_abs >= GATE2_PASS_THRESHOLD:
        verdict = "PASS"
        exit_code = 0
    elif all_in_noise:
        verdict = "NULL_RESULT"
        exit_code = 10
    else:
        verdict = "INCONCLUSIVE"
        exit_code = 20

    lines.extend([
        "",
        f"**Verdict: {verdict}** (exit {exit_code})",
        "",
        f"- LIKELY threshold: |Δ| ≥ {GATE2_LIKELY_THRESHOLD}",
        f"- NOISE threshold:  |Δ| ≤ {GATE2_NOISE_THRESHOLD}",
        f"- PASS threshold:   |Δ| ≥ {GATE2_PASS_THRESHOLD} on at least one type",
    ])
    if verdict == "NULL_RESULT":
        lines.append(
            "- Day-3 scope: drop 'data-supported rule'; do descriptive comparison."
        )
    return "\n".join(lines) + "\n", exit_code


def run_ablation_cli(
    *, cfg: Config, target_size: int, max_new_llm_calls: int,
    backend: Backend | None = None, granularity: str = "message",
) -> dict:
    qids = select_subset(cfg, target_size=target_size)
    client = _make_client(cfg, backend=backend)
    rows = run_ablation(
        cfg=cfg, qids=qids, llm_client=client,
        max_new_llm_calls=max_new_llm_calls,
        granularity=granularity,
    )

    cfg.analysis_dir.mkdir(parents=True, exist_ok=True)
    jsonl = "\n".join(json.dumps(r) for r in rows)
    _atomic_write(_jsonl_path(cfg), jsonl + ("\n" if jsonl else ""))

    summary = _summarize(rows)
    _atomic_write(_summary_path(cfg), json.dumps(summary, indent=2))

    manifest = {
        "model": cfg.model,
        "qids": qids,
        "target_size": target_size,
        "max_new_llm_calls": max_new_llm_calls,
        "granularity": granularity,
        "counts": {
            "total_rows": len(rows),
            "skipped": sum(1 for r in rows if r.get("skipped")),
            "completed": sum(1 for r in rows if not r.get("skipped")),
            "new_llm_calls": client.call_count,
            "cache_hits": client.cache_hits,
        },
    }
    _atomic_write(_manifest_path(cfg), json.dumps(manifest, indent=2))
    return manifest


def _print_report(summary: dict) -> None:
    print("Type-level ablation summary")
    for r in summary["per_type"]:
        print(
            f"  {r['type']:12s} n={r['n_used']:3d} skipped={r['n_skipped']:3d} "
            f"acc_with={r['acc_with']:.2f} acc_without={r['acc_without']:.2f} "
            f"delta={r['delta']:+.3f}"
        )


def main(
    argv: list[str] | None = None,
    *,
    cfg: Config | None = None,
    backend: Backend | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Type-level ablation: per-type Delta accuracy (Gate 2)"
    )
    parser.add_argument("--n", type=int, default=20,
                        help="Target subset size (default 20)")
    parser.add_argument("--max-calls", type=int, default=MAX_NEW_LLM_CALLS_DEFAULT,
                        help=f"Hard cap on new LLM calls (default {MAX_NEW_LLM_CALLS_DEFAULT})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config without running ablation")
    parser.add_argument("--report", action="store_true",
                        help="Print existing summary and exit")
    parser.add_argument("--gate2", action="store_true",
                        help="Emit gate2_report.md and return exit code 0/10/20")
    parser.add_argument("--granularity", choices=("message", "span"),
                        default="message",
                        help="Masking granularity (default message). "
                             "'span' deletes only the claim's source span "
                             "— produces Δ≈0 on sparsely-covered claims.")
    parser.add_argument("--control", action="store_true",
                        help="Control experiment: blank ALL messages in "
                             "rounds 1..final-1 (no type filter) and replay "
                             "the final round on the same subset. Writes "
                             "control_result.json. Use to distinguish 'no "
                             "type matters' from 'single-type ablation too "
                             "weak'.")
    args = parser.parse_args(argv)

    if cfg is None:
        cfg = get_config()
    cfg.ensure_dirs()

    if args.dry_run:
        dialogues = sum(1 for _ in cfg.dialogues_dir.glob("*.json"))
        claims = sum(1 for _ in cfg.claims_dir.glob("*.json"))
        print("dry-run OK")
        print(f"  model         = {cfg.model}")
        print(f"  n (target)    = {args.n}")
        print(f"  max_calls     = {args.max_calls}")
        print(f"  dialogues_dir = {cfg.dialogues_dir} ({dialogues} files)")
        print(f"  claims_dir    = {cfg.claims_dir} ({claims} files)")
        return 0

    if args.report:
        path = _summary_path(cfg)
        if not path.exists():
            print(f"ERROR: no summary at {path}", file=sys.stderr)
            return 2
        _print_report(json.loads(path.read_text(encoding="utf-8")))
        return 0

    if args.gate2:
        path = _summary_path(cfg)
        if not path.exists():
            print(f"ERROR: no summary at {path} — run `make ablate` first",
                  file=sys.stderr)
            return 2
        summary = json.loads(path.read_text(encoding="utf-8"))
        report, exit_code = _render_gate2(summary)
        _atomic_write(_gate2_path(cfg), report)
        print(report)
        return exit_code

    if args.control:
        qids = select_subset(cfg, target_size=args.n, require_claims=False)
        if not qids:
            print("ERROR: subset empty for control", file=sys.stderr)
            return 1
        client = _make_client(cfg, backend=backend)
        rows = run_control_ablation(
            cfg=cfg, qids=qids, llm_client=client,
            max_new_llm_calls=args.max_calls,
        )
        completed = [r for r in rows if not r.get("skipped")]
        n_used = len(completed)
        n_correct_without = sum(1 for r in completed if r.get("correct_without"))
        n_correct_with = sum(1 for r in completed if r.get("correct_with"))
        acc_with = n_correct_with / n_used if n_used else 0.0
        acc_without = n_correct_without / n_used if n_used else 0.0
        delta = acc_with - acc_without
        payload = {
            "model": cfg.model,
            "qids": qids,
            "rows": rows,
            "summary": {
                "n_used": n_used,
                "n_skipped": len(rows) - n_used,
                "acc_with": acc_with,
                "acc_without": acc_without,
                "delta": delta,
                "new_llm_calls": client.call_count,
            },
        }
        _atomic_write(_control_path(cfg), json.dumps(payload, indent=2))
        print("# Control (drop-all-messages) result", file=sys.stderr)
        print(f"  n                = {n_used}", file=sys.stderr)
        print(f"  acc(with)        = {acc_with:.3f}", file=sys.stderr)
        print(f"  acc(without_all) = {acc_without:.3f}", file=sys.stderr)
        print(f"  delta            = {delta:+.3f}", file=sys.stderr)
        print(file=sys.stderr)
        if delta <= 0.03:
            print("Interpretation: Δ ≤ 0.03 → history content has NO causal "
                  "effect. Debate gain is voting-based, not dialogue-based. "
                  "Gate-2 null-result is a REAL finding.", file=sys.stderr)
        elif delta >= 0.10:
            print("Interpretation: Δ ≥ 0.10 → history content DOES matter. "
                  "Type-level Δ=0 reflects signal-too-weak, not 'no type "
                  "matters'. Descriptive attribution via signal_scores is "
                  "needed (spec §5.4 path).", file=sys.stderr)
        else:
            print("Interpretation: Δ between thresholds — history has some "
                  "effect but single-type ablation likely misses it. "
                  "Consider descriptive attribution.", file=sys.stderr)
        return 0

    manifest = run_ablation_cli(
        cfg=cfg, target_size=args.n, max_new_llm_calls=args.max_calls,
        backend=backend, granularity=args.granularity,
    )
    if not manifest["qids"]:
        print(
            "ERROR: subset is empty — needs pilot single artifacts + "
            "collected debate dialogues + extracted claims for qids "
            "where single_wrong and debate_right",
            file=sys.stderr,
        )
        return 1
    c = manifest["counts"]
    print(
        f"done. rows={c['total_rows']} completed={c['completed']} "
        f"skipped={c['skipped']} new_calls={c['new_llm_calls']}",
        file=sys.stderr,
    )
    print("next: python -m agentdiet.cli.ablate --report", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
