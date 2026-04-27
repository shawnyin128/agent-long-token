"""Smoke-test Qwen3-30B-A3B + enable_thinking toggle on a running vLLM.

Issues 5 GSM8K-style calls each at thinking=True and thinking=False
against the same 5 questions. Asserts that the thinking-on output
is on average longer (more output tokens) than the thinking-off
output. Writes artifacts/serving/qwen3_smoke.json.

Usage on HPC:
    export AGENTDIET_BASE_URL=http://localhost:8000/v1
    python scripts/serving/smoke_qwen3.py

To dry-run with a synthetic backend (no vLLM needed):
    pytest tests/cross_model_grid_hpc_serving/test_smoke_qwen3_local.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agentdiet.llm_client import Backend, OpenAIBackend


SMOKE_QUESTIONS: list[tuple[str, str]] = [
    ("q1", "Janet has 16 eggs/day. She eats 3, bakes 4, sells the rest at $2 each. "
            "How many dollars does she make per day? End with '#### <answer>'."),
    ("q2", "A train travels 60 miles in 1.5 hours. What is its speed in miles per hour? "
            "End with '#### <answer>'."),
    ("q3", "If x + 7 = 19, what is x? End with '#### <answer>'."),
    ("q4", "A shirt costs $20 with a 25% discount. What is the sale price? "
            "End with '#### <answer>'."),
    ("q5", "There are 12 cookies, divided equally among 4 friends. "
            "How many cookies does each friend get? End with '#### <answer>'."),
]

DEFAULT_OUTPUT = Path("artifacts/serving/qwen3_smoke.json")


def _call_one(backend: Backend, qid: str, prompt: str, model: str,
              thinking: bool) -> dict:
    messages = [
        {"role": "system", "content": "You are a careful math problem solver."},
        {"role": "user", "content": prompt},
    ]
    result = backend.chat_full(
        messages, model=model, temperature=0.0, thinking=thinking,
    )
    return {
        "qid": qid,
        "thinking": thinking,
        "response_chars": len(result.response or ""),
        "response_preview": (result.response or "")[:200],
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
    }


def run_smoke(*, backend: Backend, model: str,
              questions: Optional[list[tuple[str, str]]] = None) -> dict:
    qs = questions or SMOKE_QUESTIONS
    on_results: list[dict] = []
    off_results: list[dict] = []
    for qid, prompt in qs:
        on_results.append(_call_one(backend, qid, prompt, model, thinking=True))
        off_results.append(_call_one(backend, qid, prompt, model, thinking=False))

    def _mean_tok(rows: list[dict]) -> float:
        toks = [r.get("completion_tokens") or 0 for r in rows]
        return float(sum(toks)) / len(toks) if toks else 0.0

    mean_on = _mean_tok(on_results)
    mean_off = _mean_tok(off_results)
    delta = mean_on - mean_off
    passed = delta > 0  # thinking-on should produce more output tokens

    return {
        "model": model,
        "tested_at": datetime.now(timezone.utc).isoformat(),
        "thinking_on": on_results,
        "thinking_off": off_results,
        "summary": {
            "mean_tokens_on": mean_on,
            "mean_tokens_off": mean_off,
            "delta": delta,
            "passed": passed,
        },
    }


def main(argv: list[str] | None = None, *,
         backend: Backend | None = None,
         output_path: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-url",
                        default=os.environ.get("AGENTDIET_BASE_URL",
                                               "http://localhost:8000/v1"))
    parser.add_argument("--api-key",
                        default=os.environ.get("AGENTDIET_API_KEY", "EMPTY"))
    args = parser.parse_args(argv)

    if backend is None:
        backend = OpenAIBackend(
            base_url=args.base_url,
            api_key=args.api_key,
            model_family="qwen3",
        )
    target = output_path or args.output

    artifact = run_smoke(backend=backend, model=args.model)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    s = artifact["summary"]
    print(f"smoke artifact -> {target}", file=sys.stderr)
    print(f"  mean_tokens_on  = {s['mean_tokens_on']:.1f}", file=sys.stderr)
    print(f"  mean_tokens_off = {s['mean_tokens_off']:.1f}", file=sys.stderr)
    print(f"  delta           = {s['delta']:+.1f}", file=sys.stderr)
    print(f"  passed          = {s['passed']}", file=sys.stderr)
    return 0 if s["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
