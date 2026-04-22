"""Claim extraction CLI.

Reads dialogues from ``cfg.dialogues_dir`` and writes per-qid claim
artifacts to ``cfg.claims_dir``. Resumable (skips qids whose claim
file already exists); writes a manifest summary to
``artifacts/claims/manifest.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from agentdiet.config import Config, get_config
from agentdiet.extract_claims import extract_claims_for_dialogue
from agentdiet.llm_client import Backend, LLMClient, OpenAIBackend
from agentdiet.types import Dialogue


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _manifest_path(cfg: Config) -> Path:
    return cfg.artifacts_dir / "claims" / "manifest.json"


def _make_client(cfg: Config, backend: Backend | None = None) -> LLMClient:
    if backend is None:
        backend = OpenAIBackend(
            base_url=cfg.base_url, api_key=cfg.api_key, timeout_s=cfg.request_timeout_s
        )
    return LLMClient(backend, cache_path=cfg.cache_path, max_retries=cfg.max_retries)


def _load_dialogue(path: Path) -> Dialogue:
    return Dialogue.model_validate_json(path.read_text(encoding="utf-8"))


def _write_failure(cfg: Config, qid: str, exc: BaseException) -> None:
    path = cfg.failures_dir / "claim_extraction" / f"{qid}__dialogue_level.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        path,
        json.dumps(
            {
                "qid": qid,
                "stage": "claim_extraction",
                "exc_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
            },
            indent=2,
        ),
    )


def run_extraction(cfg: Config, *, backend: Backend | None = None) -> dict:
    client = _make_client(cfg, backend=backend)
    in_dir = cfg.dialogues_dir
    out_dir = cfg.claims_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dialogue_files = sorted(in_dir.glob("*.json"))
    outcomes: list[dict] = []

    for i, dpath in enumerate(dialogue_files, 1):
        qid = dpath.stem
        out_path = out_dir / f"{qid}.json"
        if out_path.exists():
            outcomes.append({"qid": qid, "outcome": "cached"})
            print(f"[{i}/{len(dialogue_files)}] qid={qid} cached", file=sys.stderr)
            continue

        try:
            dialogue = _load_dialogue(dpath)
        except Exception as exc:  # noqa: BLE001
            _write_failure(cfg, qid, exc)
            outcomes.append({"qid": qid, "outcome": "failed"})
            print(
                f"[{i}/{len(dialogue_files)}] qid={qid} FAILED (load {type(exc).__name__})",
                file=sys.stderr,
            )
            continue

        try:
            result = extract_claims_for_dialogue(
                dialogue=dialogue,
                llm_client=client,
                model=cfg.model,
                temperature=cfg.temperature,
                failures_dir=cfg.failures_dir,
            )
            _atomic_write(out_path, json.dumps(result, indent=2, ensure_ascii=False))
            outcome = "partial" if result["extraction_failed"] else "ok"
            outcomes.append({"qid": qid, "outcome": outcome})
            print(
                f"[{i}/{len(dialogue_files)}] qid={qid} {outcome} "
                f"(claims={len(result['claims'])})",
                file=sys.stderr,
            )
        except Exception as exc:  # noqa: BLE001
            _write_failure(cfg, qid, exc)
            outcomes.append({"qid": qid, "outcome": "failed"})
            print(
                f"[{i}/{len(dialogue_files)}] qid={qid} FAILED ({type(exc).__name__})",
                file=sys.stderr,
            )

    counts = {k: 0 for k in ("ok", "partial", "failed", "cached")}
    for o in outcomes:
        counts[o["outcome"]] += 1
    assert sum(counts.values()) == len(outcomes), \
        f"outcome bucket sum {sum(counts.values())} != dialogues {len(outcomes)}"

    manifest = {
        "model": cfg.model,
        "n": len(dialogue_files),
        "counts": counts,
        "outcomes": outcomes,
    }
    _atomic_write(_manifest_path(cfg), json.dumps(manifest, indent=2))
    return manifest


def _print_report(manifest: dict) -> None:
    c = manifest["counts"]
    print("Claim extraction manifest")
    print(f"  model: {manifest['model']}")
    print(f"  dialogues scanned: {manifest['n']}")
    print(f"  outcomes:")
    for k in ("ok", "partial", "failed", "cached"):
        print(f"    {k:9s} {c.get(k, 0)}")


def main(
    argv: list[str] | None = None,
    *,
    cfg: Config | None = None,
    backend: Backend | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Extract structured claims from collected dialogues"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate imports and config; do not call LLM")
    parser.add_argument("--report-manifest", action="store_true",
                        help="Print existing manifest summary and exit")
    args = parser.parse_args(argv)

    if cfg is None:
        cfg = get_config()
    cfg.ensure_dirs()

    if args.dry_run:
        scan_ok = cfg.dialogues_dir.is_dir()
        print("dry-run OK")
        print(f"  model           = {cfg.model}")
        print(f"  dialogues_dir   = {cfg.dialogues_dir} (exists={scan_ok})")
        print(f"  claims_dir      = {cfg.claims_dir}")
        return 0

    if args.report_manifest:
        path = _manifest_path(cfg)
        if not path.exists():
            print(f"ERROR: no manifest at {path}", file=sys.stderr)
            return 2
        _print_report(json.loads(path.read_text(encoding="utf-8")))
        return 0

    print(
        f"extraction model={cfg.model} dialogues_dir={cfg.dialogues_dir}",
        file=sys.stderr,
    )
    manifest = run_extraction(cfg, backend=backend)
    c = manifest["counts"]
    print(
        f"done. ok={c['ok']} partial={c['partial']} cached={c['cached']} "
        f"failed={c['failed']}",
        file=sys.stderr,
    )
    print("next: python -m agentdiet.cli.extract --report-manifest", file=sys.stderr)

    # Exit 1 only if every dialogue failed at the dialogue level.
    total = manifest["n"]
    if total > 0 and c["failed"] == total:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
