from __future__ import annotations

import argparse
import sys

from agentdiet.config import get_config
from agentdiet.debate import run_debate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run debate collection over GSM8K sample")
    parser.add_argument("--n", type=int, default=None, help="Number of questions to collect")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate imports and config; do not call LLM")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip qids whose dialogue already exists")
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
        # reference run_debate so unused-import lint would flag regression
        assert callable(run_debate)
        return 0

    # Full collection is implemented in feature 'full-collection'.
    print("ERROR: full collection not yet implemented (feature: full-collection). Use --dry-run.",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
