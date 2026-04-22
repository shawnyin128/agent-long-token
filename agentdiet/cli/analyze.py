"""Analysis CLI: flip events + per-claim signals.

Reads dialogues + claim artifacts, computes ``FlipEvent`` records via
``analysis.flip`` and independent signals via ``analysis.signals``,
and writes three artifacts to ``artifacts/analysis/``:

  * ``flip_events.jsonl`` — one JSON object per line with a ``qid``
    field prepended for joinability
  * ``signal_scores.parquet`` — 6-column pyarrow table
  * ``manifest.json`` — summary counts

Usage::

    python -m agentdiet.cli.analyze            # real embedder
    python -m agentdiet.cli.analyze --embedder fake
    python -m agentdiet.cli.analyze --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from agentdiet.analysis.flip import locate_flips
from agentdiet.analysis.signals import (
    Embedder,
    HashingFakeEmbedder,
    SentenceTransformerEmbedder,
    compute_signals,
)
from agentdiet.config import Config, get_config
from agentdiet.types import Dialogue


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(content)
    os.replace(tmp, path)


def _manifest_path(cfg: Config) -> Path:
    return cfg.analysis_dir / "manifest.json"


def _load_dialogue(path: Path) -> Dialogue:
    return Dialogue.model_validate_json(path.read_text(encoding="utf-8"))


def _select_embedder(*, use_fake: bool) -> Embedder:
    if use_fake:
        return HashingFakeEmbedder()
    try:
        emb = SentenceTransformerEmbedder()
        # Force early import failure so the warning surfaces now.
        emb._ensure()  # noqa: SLF001
        return emb
    except ImportError as e:
        print(
            f"WARNING: {e} — falling back to HashingFakeEmbedder. "
            "Install `.[analysis]` for real signals.",
            file=sys.stderr,
        )
        return HashingFakeEmbedder()


def run_analysis(cfg: Config, *, use_fake_embedder: bool = False) -> dict:
    dialogue_qids = {p.stem for p in cfg.dialogues_dir.glob("*.json")}
    claim_qids = {p.stem for p in cfg.claims_dir.glob("*.json")}
    eligible = sorted(dialogue_qids & claim_qids)
    skipped_no_claims = sorted(dialogue_qids - claim_qids)

    embedder = _select_embedder(use_fake=use_fake_embedder)

    flip_lines: list[str] = []
    signal_cols: dict[str, list] = {k: [] for k in (
        "qid", "claim_id", "flip_coincidence", "novelty",
        "referenced_later", "position",
    )}
    flip_events_total = 0

    for qid in eligible:
        dialogue = _load_dialogue(cfg.dialogues_dir / f"{qid}.json")
        claims_doc = json.loads(
            (cfg.claims_dir / f"{qid}.json").read_text(encoding="utf-8")
        )
        events = locate_flips(dialogue, claims_doc)
        for fe in events:
            payload = fe.model_dump()
            payload["qid"] = qid
            flip_lines.append(json.dumps(payload))
        flip_events_total += len(events)

        rows = compute_signals(
            claims_doc.get("claims", []),
            flip_events=events,
            embedder=embedder,
        )
        for r in rows:
            signal_cols["qid"].append(qid)
            signal_cols["claim_id"].append(r["claim_id"])
            signal_cols["flip_coincidence"].append(bool(r["flip_coincidence"]))
            signal_cols["novelty"].append(float(r["novelty"]))
            signal_cols["referenced_later"].append(bool(r["referenced_later"]))
            signal_cols["position"].append(int(r["position"]))

    cfg.analysis_dir.mkdir(parents=True, exist_ok=True)
    flip_path = cfg.analysis_dir / "flip_events.jsonl"
    _atomic_write_text(
        flip_path,
        ("\n".join(flip_lines) + "\n") if flip_lines else "",
    )

    sig_path = cfg.analysis_dir / "signal_scores.parquet"
    table = pa.table(signal_cols)
    buf = pa.BufferOutputStream()
    pq.write_table(table, buf)
    _atomic_write_bytes(sig_path, buf.getvalue().to_pybytes())

    manifest = {
        "model": cfg.model,
        "counts": {
            "qids_processed": len(eligible),
            "qids_skipped_no_claims": len(skipped_no_claims),
            "flip_events": flip_events_total,
            "signal_rows": len(signal_cols["claim_id"]),
        },
        "qids_processed": eligible,
        "qids_skipped_no_claims": skipped_no_claims,
    }
    _atomic_write_text(_manifest_path(cfg), json.dumps(manifest, indent=2))
    return manifest


def _print_report(manifest: dict) -> None:
    c = manifest["counts"]
    print("Analysis manifest")
    print(f"  model: {manifest['model']}")
    print(f"  qids_processed:         {c['qids_processed']}")
    print(f"  qids_skipped_no_claims: {c['qids_skipped_no_claims']}")
    print(f"  flip_events:            {c['flip_events']}")
    print(f"  signal_rows:            {c['signal_rows']}")


def main(
    argv: list[str] | None = None,
    *,
    cfg: Config | None = None,
    use_fake_embedder: bool | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Compute flip events + independent per-claim signals"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config; do not run embedder or write artifacts")
    parser.add_argument("--report", action="store_true",
                        help="Print existing manifest summary and exit")
    parser.add_argument("--embedder", choices=("real", "fake"), default="real",
                        help="Embedder to use (default real; falls back to "
                             "fake with warning if sentence-transformers missing)")
    args = parser.parse_args(argv)

    if cfg is None:
        cfg = get_config()
    cfg.ensure_dirs()

    if args.dry_run:
        dialogues = sum(1 for _ in cfg.dialogues_dir.glob("*.json"))
        claims = sum(1 for _ in cfg.claims_dir.glob("*.json"))
        print("dry-run OK")
        print(f"  model              = {cfg.model}")
        print(f"  dialogues_dir      = {cfg.dialogues_dir} ({dialogues} files)")
        print(f"  claims_dir         = {cfg.claims_dir} ({claims} files)")
        print(f"  analysis_dir       = {cfg.analysis_dir}")
        return 0

    if args.report:
        path = _manifest_path(cfg)
        if not path.exists():
            print(f"ERROR: no manifest at {path}", file=sys.stderr)
            return 2
        _print_report(json.loads(path.read_text(encoding="utf-8")))
        return 0

    fake_flag = args.embedder == "fake" if use_fake_embedder is None else use_fake_embedder
    manifest = run_analysis(cfg, use_fake_embedder=fake_flag)

    if manifest["counts"]["qids_processed"] == 0:
        print(
            "ERROR: no qids had both a dialogue and a claims artifact — "
            "run `make collect` and `make extract` first",
            file=sys.stderr,
        )
        return 1

    c = manifest["counts"]
    print(
        f"done. qids={c['qids_processed']} flips={c['flip_events']} "
        f"signals={c['signal_rows']}",
        file=sys.stderr,
    )
    print("next: python -m agentdiet.cli.analyze --report", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
