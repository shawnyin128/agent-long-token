from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path

from agentdiet.baseline import run_single_agent
from agentdiet.config import Config, get_config
from agentdiet.dataset import Question, load_gsm8k
from agentdiet.debate import run_debate
from agentdiet.llm_client import LLMClient, OpenAIBackend


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _pilot_paths(cfg: Config, method: str) -> Path:
    return cfg.artifacts_dir / "pilot" / method / cfg.model_slug


def _write_failure(cfg: Config, method: str, qid: str, exc: BaseException) -> None:
    path = cfg.failures_dir / "pilot" / f"{method}__{qid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        path,
        json.dumps(
            {
                "qid": qid,
                "method": method,
                "model": cfg.model,
                "exc_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
            },
            indent=2,
        ),
    )


def _make_llm_client(cfg: Config, backend=None) -> LLMClient:
    if backend is None:
        backend = OpenAIBackend(base_url=cfg.base_url, api_key=cfg.api_key, timeout_s=cfg.request_timeout_s)
    return LLMClient(backend, cache_path=cfg.cache_path, max_retries=cfg.max_retries)


def run_pilot(
    cfg: Config,
    questions: list[Question],
    *,
    run_single: bool,
    run_debate_flag: bool,
    backend=None,
) -> dict:
    client = _make_llm_client(cfg, backend=backend)
    single_dir = _pilot_paths(cfg, "single")
    debate_dir = _pilot_paths(cfg, "debate")
    single_dir.mkdir(parents=True, exist_ok=True)
    debate_dir.mkdir(parents=True, exist_ok=True)

    outcomes: list[dict] = []
    for i, q in enumerate(questions, 1):
        row = {"qid": q.qid, "single": "skip", "debate": "skip"}

        if run_single:
            out = single_dir / f"{q.qid}.json"
            if out.exists():
                row["single"] = "cached"
            else:
                try:
                    d = run_single_agent(q, client, model=cfg.model, temperature=cfg.temperature)
                    _atomic_write(out, d.model_dump_json())
                    row["single"] = "ok" if d.final_answer is not None else "unparsed"
                except Exception as exc:
                    _write_failure(cfg, "single", q.qid, exc)
                    row["single"] = "failed"

        if run_debate_flag:
            out = debate_dir / f"{q.qid}.json"
            if out.exists():
                row["debate"] = "cached"
            else:
                try:
                    d = run_debate(
                        q, client, model=cfg.model,
                        n_agents=cfg.n_agents, n_rounds=cfg.n_rounds,
                        temperature=cfg.temperature, seed=cfg.seed,
                    )
                    _atomic_write(out, d.model_dump_json())
                    row["debate"] = "ok" if d.final_answer is not None else "unparsed"
                except Exception as exc:
                    _write_failure(cfg, "debate", q.qid, exc)
                    row["debate"] = "failed"

        outcomes.append(row)
        print(f"[{i}/{len(questions)}] qid={q.qid} single={row['single']} debate={row['debate']}",
              file=sys.stderr)

    manifest = {
        "model": cfg.model,
        "n": len(questions),
        "seed": cfg.seed,
        "n_agents": cfg.n_agents,
        "n_rounds": cfg.n_rounds,
        "outcomes": outcomes,
    }
    manifest_path = cfg.artifacts_dir / "pilot" / "manifest.json"
    _atomic_write(manifest_path, json.dumps(manifest, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Gate-1 pilot")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--no-single", action="store_true")
    parser.add_argument("--no-debate", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = get_config()
    cfg.ensure_dirs()

    n = args.n if args.n is not None else cfg.n_pilot
    seed = args.seed if args.seed is not None else cfg.seed

    print(f"pilot model={cfg.model} n={n} seed={seed} base_url={cfg.base_url}", file=sys.stderr)
    questions = load_gsm8k(split=args.split, n=n, seed=seed, cache_dir=cfg.hf_cache_dir)
    print(f"loaded {len(questions)} questions", file=sys.stderr)

    manifest = run_pilot(
        cfg, questions,
        run_single=not args.no_single,
        run_debate_flag=not args.no_debate,
    )
    n_failed = sum(1 for o in manifest["outcomes"]
                   if o["single"] == "failed" or o["debate"] == "failed")
    print(f"done. outcomes written; {n_failed} failures", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
