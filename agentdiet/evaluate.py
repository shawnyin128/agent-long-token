"""Evaluation sweep: 5 methods × N questions × (accuracy, tokens, acc_per_1k).

One synthesizer-style replay per (qid, method) pair. Keeps the history
quality signal isolated from 3-agent majority-vote variance.

Token counting uses a simple char/4 heuristic (``TOKENS_PER_CHAR=0.25``).
This is approximate and intentionally monotone in prompt length — good
enough for the relative invariants (``tokens_b1 >= tokens_b3 >= tokens_ours``)
without taking on a model-specific tokenizer dependency.

Sanity invariants (asserted as warnings, not raises; spec §9.3):
  - ``acc(b1) >= acc(b2)``
  - ``acc(ours) >= acc(b5)``
  - ``tokens(b1) >= tokens(b3)``
  - ``tokens(b1) >= tokens(ours)``
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from agentdiet import compress
from agentdiet.agents import FORMAT_INSTR, SYNTHESIZER_PROMPT
from agentdiet.compress import Policy
from agentdiet.config import Config
from agentdiet.dataset import parse_answer
from agentdiet.llm_client import LLMClient
from agentdiet.types import Dialogue


TOKENS_PER_CHAR = 0.25


MethodName = Literal["b1", "b2", "b3", "b5", "ours"]


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return int(round(len(text) * TOKENS_PER_CHAR))


class PerQuestionResult(BaseModel):
    qid: str
    method: MethodName
    compressed_tokens: int = Field(ge=0)
    final_answer: Optional[str]
    gold: str
    correct: bool


class MethodSummary(BaseModel):
    method: MethodName
    accuracy: float
    total_tokens: int
    acc_per_1k: float
    n_evaluated: int

    @classmethod
    def build(cls, method: MethodName,
              per_question: list[PerQuestionResult]) -> "MethodSummary":
        rows = list(per_question)
        for r in rows:
            if r.method != method:
                raise ValueError(
                    f"row method {r.method!r} != summary method {method!r}"
                )
        n = len(rows)
        correct = sum(1 for r in rows if r.correct)
        total_tokens = sum(r.compressed_tokens for r in rows)
        accuracy = (correct / n) if n else 0.0
        acc_per_1k = (accuracy / (total_tokens / 1000.0)) if total_tokens > 0 else 0.0
        return cls(
            method=method,
            accuracy=accuracy,
            total_tokens=total_tokens,
            acc_per_1k=acc_per_1k,
            n_evaluated=n,
        )


EVAL_USER_TEMPLATE = (
    "You are reviewing a transcript of other agents' reasoning on this "
    "math problem:\n\n{question}\n\n"
    "Transcript (compressed):\n\n{history}\n\n"
    "Produce your own final answer. " + FORMAT_INSTR + "."
)


def _synthesize_final_answer(
    *, dialogue: Dialogue, compressed_history: str,
    llm_client: LLMClient, model: str, temperature: float = 0.0,
) -> tuple[Optional[str], str]:
    """Call the synthesizer-style prompt once on the compressed history.
    Returns (parsed_answer, raw_response_text)."""
    user = EVAL_USER_TEMPLATE.format(
        question=dialogue.question, history=compressed_history or "(empty)",
    )
    messages = [
        {"role": "system", "content": SYNTHESIZER_PROMPT},
        {"role": "user", "content": user},
    ]
    raw = llm_client.chat(messages, model=model, temperature=temperature)
    return parse_answer(raw), raw


def evaluate_method_on_qid(
    *,
    dialogue: Dialogue,
    claims_doc: Optional[dict],
    signal_scores: Any,
    method: MethodName,
    policy: Policy,
    llm_client: LLMClient,
    model: str,
    temperature: float = 0.0,
) -> PerQuestionResult:
    history = compress.apply(
        dialogue, policy, claims_doc=claims_doc,
        signal_scores=signal_scores,
    )
    parsed, _ = _synthesize_final_answer(
        dialogue=dialogue, compressed_history=history,
        llm_client=llm_client, model=model, temperature=temperature,
    )
    gold = str(dialogue.gold_answer).strip()
    correct = (parsed is not None and str(parsed).strip() == gold)
    return PerQuestionResult(
        qid=dialogue.question_id,
        method=method,
        compressed_tokens=count_tokens(history),
        final_answer=parsed,
        gold=gold,
        correct=correct,
    )


def _check_invariants(summaries: dict[MethodName, MethodSummary]) -> list[str]:
    violations: list[str] = []

    def a(m): return summaries[m].accuracy
    def t(m): return summaries[m].total_tokens

    if "b1" in summaries and "b2" in summaries:
        if a("b1") < a("b2"):
            violations.append(
                f"acc(b1)={a('b1'):.3f} < acc(b2)={a('b2'):.3f} "
                "— debate should not be worse than single-agent"
            )
    if "ours" in summaries and "b5" in summaries:
        if a("ours") < a("b5"):
            violations.append(
                f"acc(ours)={a('ours'):.3f} < acc(b5)={a('b5'):.3f} "
                "— our selection should not lose to random drop"
            )
    if "b1" in summaries and "b3" in summaries:
        if t("b1") < t("b3"):
            violations.append(
                f"tokens(b1)={t('b1')} < tokens(b3)={t('b3')} "
                "— full history should use more tokens than sliding window"
            )
    if "b1" in summaries and "ours" in summaries:
        if t("b1") < t("ours"):
            violations.append(
                f"tokens(b1)={t('b1')} < tokens(ours)={t('ours')} "
                "— full history should use more tokens than 'ours'"
            )
    return violations


def run_sweep(
    *,
    cfg: Config,
    qids: list[str],
    policies: dict[str, Policy],
    llm_client: LLMClient,
    loader,
) -> dict:
    """Run 5 methods × qids sweep.

    ``loader`` is a callable ``(cfg, qid) -> (Dialogue, claims_doc,
    signal_scores)`` so the CLI can inject filesystem lookups while
    tests inject fixtures.
    """
    required = {"b1", "b2", "b3", "b5", "ours"}
    missing = required - set(policies.keys())
    if missing:
        raise KeyError(f"policies dict missing methods: {sorted(missing)}")

    per_question: list[PerQuestionResult] = []
    for qid in qids:
        dialogue, claims_doc, signal_scores = loader(cfg, qid)
        for method in ("b1", "b2", "b3", "b5", "ours"):
            r = evaluate_method_on_qid(
                dialogue=dialogue, claims_doc=claims_doc,
                signal_scores=signal_scores,
                method=method,  # type: ignore[arg-type]
                policy=policies[method],
                llm_client=llm_client, model=cfg.model,
                temperature=cfg.temperature,
            )
            per_question.append(r)

    per_method: list[MethodSummary] = []
    summaries: dict[str, MethodSummary] = {}
    for method in ("b1", "b2", "b3", "b5", "ours"):
        rows = [r for r in per_question if r.method == method]
        summary = MethodSummary.build(method=method, per_question=rows)  # type: ignore[arg-type]
        per_method.append(summary)
        summaries[method] = summary

    violations = _check_invariants(summaries)

    return {
        "per_question": [r.model_dump() for r in per_question],
        "per_method": [s.model_dump() for s in per_method],
        "invariant_violations": violations,
    }
