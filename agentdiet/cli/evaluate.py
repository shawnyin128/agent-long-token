"""Evaluation sweep CLI — 5 methods × N qids → results.json.

Reads policy from ``artifacts/compression/policy.json`` (default path;
override with ``--policy-path``) for the ``ours`` mode; uses canonical
defaults for b1/b2/b3/b5. Writes ``artifacts/evaluation/results.json``
with per-question rows, per-method aggregates, and sanity invariant
violations.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from agentdiet.compress import Policy, load_policy
from agentdiet.config import Config, get_config
from agentdiet.evaluate import run_sweep
from agentdiet.llm_client import Backend, LLMClient, OpenAIBackend
from agentdiet.types import Dialogue


def _results_path(cfg: Config) -> Path:
    return cfg.evaluation_dir / "results.json"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _make_client(cfg: Config, backend: Backend | None = None) -> LLMClient:
    if backend is None:
        backend = OpenAIBackend(
            base_url=cfg.base_url, api_key=cfg.api_key, timeout_s=cfg.request_timeout_s
        )
    return LLMClient(backend, cache_path=cfg.cache_path, max_retries=cfg.max_retries)


def _fs_loader(cfg: Config, qid: str) -> tuple[Dialogue, dict, Any]:
    d = Dialogue.model_validate_json(
        (cfg.dialogues_dir / f"{qid}.json").read_text(encoding="utf-8")
    )
    claims_doc = json.loads(
        (cfg.claims_dir / f"{qid}.json").read_text(encoding="utf-8")
    )
    # signal_scores (optional) — if signal_scores.parquet exists with this qid,
    # load it as list-of-dict. Otherwise None.
    scores_path = cfg.analysis_dir / "signal_scores.parquet"
    if not scores_path.exists():
        return d, claims_doc, None
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError:
        return d, claims_doc, None
    table = pq.read_table(scores_path)
    df = table.to_pylist()
    rows = [r for r in df if r.get("qid") == qid]
    return d, claims_doc, rows


def _all_policies(cfg: Config, ours: Policy) -> dict[str, Policy]:
    return {
        "b1": Policy(mode="b1"),
        "b2": Policy(mode="b2"),
        "b3": Policy(mode="b3", last_k=1),
        "b5": Policy(mode="b5", drop_rate=0.3, random_seed=cfg.seed),
        "ours": ours,
    }


def _eligible_qids(cfg: Config, n: int) -> list[str]:
    dialogues = {p.stem for p in cfg.dialogues_dir.glob("*.json")}
    claims = {p.stem for p in cfg.claims_dir.glob("*.json")}
    both = sorted(dialogues & claims)
    if n <= 0 or n >= len(both):
        return both
    return both[:n]


def run_evaluation_cli(
    *, cfg: Config, n: int, backend: Backend | None = None,
    policy_path: Optional[Path] = None,
) -> dict:
    if policy_path is None:
        policy_path = cfg.compression_dir / "policy.json"
    ours = load_policy(policy_path)
    policies = _all_policies(cfg, ours)
    client = _make_client(cfg, backend=backend)

    qids = _eligible_qids(cfg, n)
    result = run_sweep(
        cfg=cfg, qids=qids, policies=policies,
        llm_client=client, loader=_fs_loader,
    )

    result["config"] = {
        "model": cfg.model,
        "seed": cfg.seed,
        "n": len(qids),
        "qids": qids,
        "policy_path": str(policy_path),
    }
    _atomic_write(_results_path(cfg), json.dumps(result, indent=2))
    return result


def _print_report(result: dict) -> None:
    print("Evaluation-sweep summary")
    for m in result["per_method"]:
        print(
            f"  {m['method']:6s} n={m['n_evaluated']:3d} "
            f"acc={m['accuracy']:.3f} tokens={m['total_tokens']:6d} "
            f"acc_per_1k={m['acc_per_1k']:.3f}"
        )
    violations = result.get("invariant_violations", [])
    if violations:
        print("\nInvariant violations:")
        for v in violations:
            print(f"  ! {v}")
    else:
        print("\nAll sanity invariants satisfied.")


def main(
    argv: list[str] | None = None,
    *,
    cfg: Config | None = None,
    backend: Backend | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluation sweep: 5 methods × N questions"
    )
    parser.add_argument("--n", type=int, default=0,
                        help="Subset size (0 = all eligible qids)")
    parser.add_argument("--policy-path", type=Path, default=None,
                        help="Path to ours policy.json (default cfg.compression_dir/policy.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config; do not call LLM")
    parser.add_argument("--report", action="store_true",
                        help="Print existing results summary and exit")
    args = parser.parse_args(argv)

    if cfg is None:
        cfg = get_config()
    cfg.ensure_dirs()

    if args.dry_run:
        print("dry-run OK")
        print(f"  model         = {cfg.model}")
        print(f"  n             = {args.n} (0 = all)")
        print(f"  dialogues_dir = {cfg.dialogues_dir}")
        print(f"  claims_dir    = {cfg.claims_dir}")
        print(f"  evaluation_dir= {cfg.evaluation_dir}")
        return 0

    if args.report:
        path = _results_path(cfg)
        if not path.exists():
            print(f"ERROR: no results at {path}", file=sys.stderr)
            return 2
        _print_report(json.loads(path.read_text(encoding="utf-8")))
        return 0

    qids = _eligible_qids(cfg, args.n)
    if not qids:
        print("ERROR: no qids with both dialogue and claim artifacts",
              file=sys.stderr)
        return 1

    policy_path = args.policy_path or (cfg.compression_dir / "policy.json")
    if not policy_path.exists():
        print(f"ERROR: policy file missing at {policy_path} — "
              f"run `make policy-sample` and edit it",
              file=sys.stderr)
        return 2

    result = run_evaluation_cli(
        cfg=cfg, n=args.n, backend=backend, policy_path=policy_path,
    )
    summ = {m["method"]: m for m in result["per_method"]}
    print(
        f"done. n={result['config']['n']} "
        f"acc(b1)={summ['b1']['accuracy']:.2f} "
        f"acc(ours)={summ['ours']['accuracy']:.2f} "
        f"violations={len(result['invariant_violations'])}",
        file=sys.stderr,
    )
    print("next: python -m agentdiet.cli.evaluate --report", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
