from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from agentdiet.config import Config, get_config
from agentdiet.dataset import Question, load_gsm8k
from agentdiet.debate import run_debate
from agentdiet.llm_client import LLMClient, OpenAIBackend


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _write_failure(cfg: Config, qid: str, exc: BaseException, question_text: str) -> None:
    path = cfg.failures_dir / "debate" / f"{qid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        path,
        json.dumps(
            {
                "qid": qid,
                "model": cfg.model,
                "question": question_text,
                "exc_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
            },
            indent=2,
        ),
    )


def _manifest_path(cfg: Config) -> Path:
    return cfg.artifacts_dir / "dialogues" / "manifest.json"


def _make_client(cfg: Config, backend=None) -> LLMClient:
    if backend is None:
        backend = OpenAIBackend(
            base_url=cfg.base_url, api_key=cfg.api_key, timeout_s=cfg.request_timeout_s
        )
    return LLMClient(backend, cache_path=cfg.cache_path, max_retries=cfg.max_retries)


def run_collection(
    cfg: Config,
    questions: list[Question],
    *,
    backend=None,
) -> dict:
    client = _make_client(cfg, backend=backend)
    out_dir = cfg.dialogues_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    outcomes: list[dict] = []
    for i, q in enumerate(questions, 1):
        out = out_dir / f"{q.qid}.json"
        if out.exists():
            outcomes.append({"qid": q.qid, "outcome": "cached"})
            print(f"[{i}/{len(questions)}] qid={q.qid} cached", file=sys.stderr)
            continue
        try:
            d = run_debate(
                q, client, model=cfg.model,
                n_agents=cfg.n_agents, n_rounds=cfg.n_rounds,
                temperature=cfg.temperature, seed=cfg.seed,
            )
            _atomic_write(out, d.model_dump_json())
            outcome = "ok" if d.final_answer is not None else "unparsed"
            outcomes.append({"qid": q.qid, "outcome": outcome})
            print(f"[{i}/{len(questions)}] qid={q.qid} {outcome}", file=sys.stderr)
        except Exception as exc:
            _write_failure(cfg, q.qid, exc, q.question)
            outcomes.append({"qid": q.qid, "outcome": "failed"})
            print(f"[{i}/{len(questions)}] qid={q.qid} FAILED ({type(exc).__name__})",
                  file=sys.stderr)

    # End-of-run invariant (S2).
    assert len(outcomes) == len(questions), \
        f"outcome count {len(outcomes)} != questions {len(questions)}"
    counts = {k: 0 for k in ("ok", "cached", "unparsed", "failed")}
    for o in outcomes:
        counts[o["outcome"]] += 1
    assert sum(counts.values()) == len(questions), \
        f"outcome bucket sum {sum(counts.values())} != questions {len(questions)}"

    manifest = {
        "model": cfg.model,
        "n": len(questions),
        "seed": cfg.seed,
        "n_agents": cfg.n_agents,
        "n_rounds": cfg.n_rounds,
        "counts": counts,
        "outcomes": outcomes,
    }
    _atomic_write(_manifest_path(cfg), json.dumps(manifest, indent=2))
    return manifest


def _print_report(manifest: dict) -> None:
    c = manifest["counts"]
    print("Collection manifest")
    print(f"  model: {manifest['model']}")
    print(f"  requested: {manifest['n']} seed={manifest['seed']} "
          f"n_agents={manifest['n_agents']} n_rounds={manifest['n_rounds']}")
    print(f"  outcomes:")
    for k in ("ok", "cached", "unparsed", "failed"):
        print(f"    {k:9s} {c.get(k, 0)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run 100-question debate collection")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate imports and config; do not call LLM")
    parser.add_argument("--report-manifest", action="store_true",
                        help="Print existing manifest summary and exit")
    args = parser.parse_args(argv)

    cfg = get_config()
    cfg.ensure_dirs()

    if args.dry_run:
        print("dry-run OK")
        print(f"  model         = {cfg.model}")
        print(f"  n_agents      = {cfg.n_agents}")
        print(f"  n_rounds      = {cfg.n_rounds}")
        print(f"  artifacts_dir = {cfg.artifacts_dir}")
        print(f"  dialogues_dir = {cfg.dialogues_dir}")
        assert callable(run_debate)
        return 0

    if args.report_manifest:
        path = _manifest_path(cfg)
        if not path.exists():
            print(f"ERROR: no manifest at {path}", file=sys.stderr)
            return 2
        _print_report(json.loads(path.read_text(encoding="utf-8")))
        return 0

    n = args.n if args.n is not None else cfg.n_questions
    seed = args.seed if args.seed is not None else cfg.seed

    print(f"collection model={cfg.model} n={n} seed={seed} base_url={cfg.base_url}",
          file=sys.stderr)
    questions = load_gsm8k(split=args.split, n=n, seed=seed, cache_dir=cfg.hf_cache_dir)
    print(f"loaded {len(questions)} questions", file=sys.stderr)

    manifest = run_collection(cfg, questions)

    c = manifest["counts"]
    print(f"done. ok={c['ok']} cached={c['cached']} unparsed={c['unparsed']} "
          f"failed={c['failed']}", file=sys.stderr)
    print("next: python -m agentdiet.cli.collect --report-manifest", file=sys.stderr)

    # Exit 1 only if ALL qids failed (catastrophic).
    if c["failed"] == len(questions) and len(questions) > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
