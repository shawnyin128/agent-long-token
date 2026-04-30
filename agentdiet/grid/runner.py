"""Per-condition runners (SA / voting / debate) for math and code cells.

Each runner takes one Question (math) or CodeQuestion (code) and produces
one QuestionResult with correct + token totals. The orchestrator
aggregates these into a ConditionRecord per cell.
"""
from __future__ import annotations

import re
from typing import Any, Literal, Optional

from agentdiet.agents import SOLVER_PROMPT
from agentdiet.dataset import Question, parse_answer
from agentdiet.debate import run_debate as run_math_debate
from agentdiet.debate.code_protocol import (
    parse_code_message,
    run_code_debate,
)
from agentdiet.eval.base import CodeQuestion, Judge, JudgeResult, TestCase
from agentdiet.eval.clustering import cluster_by_signature
from agentdiet.grid.types import (
    CellSpec,
    ConditionName,
    ConditionRecord,
    QuestionResult,
)
from agentdiet.llm_client import LLMClient
from agentdiet.voting import run_voting


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------


CODE_SA_SYSTEM_PROMPT = (
    "You write Python solutions to coding problems. Output exactly:\n"
    "## Notes\n"
    "<short rationale>\n\n"
    "## Code\n"
    "```python\n"
    "<your full Python solution>\n"
    "```\n"
    "Do not include any text outside these two sections."
)


def default_sa_system_prompt(domain: Literal["math", "code"]) -> str:
    if domain == "math":
        return SOLVER_PROMPT
    if domain == "code":
        return CODE_SA_SYSTEM_PROMPT
    raise ValueError(f"unknown domain: {domain}")


# ---------------------------------------------------------------------------
# Math conditions
# ---------------------------------------------------------------------------


def run_sa_math(
    question: Question, cell: CellSpec, llm_client: LLMClient,
) -> QuestionResult:
    messages = [
        {"role": "system", "content": SOLVER_PROMPT},
        {"role": "user", "content": question.question},
    ]
    result = llm_client.chat_full(
        messages, cell.model, temperature=0.0, thinking=cell.thinking,
    )
    parsed = parse_answer(result.response)
    correct = parsed is not None and str(parsed).strip() == str(question.gold_answer).strip()
    pt = result.prompt_tokens or 0
    ct = result.completion_tokens or 0
    return QuestionResult(
        qid=question.qid,
        gold=question.gold_answer,
        final_answer=parsed,
        correct=correct,
        prompt_tokens=pt,
        completion_tokens=ct,
        total_tokens=pt + ct,
        meta={"raw_response": result.response},
    )


def run_voting_q_math(
    question: Question, cell: CellSpec, llm_client: LLMClient,
    n_samples: int,
) -> QuestionResult:
    voting = run_voting(
        question=question.question,
        n_samples=n_samples,
        llm_client=llm_client,
        model=cell.model,
        system_prompt=SOLVER_PROMPT,
        thinking=cell.thinking,
    )
    correct = (
        voting.final_answer is not None
        and str(voting.final_answer).strip() == str(question.gold_answer).strip()
    )
    return QuestionResult(
        qid=question.qid,
        gold=question.gold_answer,
        final_answer=voting.final_answer,
        correct=correct,
        prompt_tokens=0,  # voting reports total only
        completion_tokens=0,
        total_tokens=voting.total_tokens,
        meta={
            "n_samples": n_samples,
            "parsed_answers": voting.parsed_answers,
        },
    )


def run_debate_q_math(
    question: Question, cell: CellSpec, llm_client: LLMClient,
    prompt_variant: str = "cooperative",
) -> QuestionResult:
    dialogue = run_math_debate(
        question=question,
        llm_client=llm_client,
        model=cell.model,
        n_agents=3, n_rounds=3,
        temperature=0.0,
        thinking=cell.thinking,
        prompt_variant=prompt_variant,
    )
    final = dialogue.final_answer
    correct = final is not None and str(final).strip() == str(question.gold_answer).strip()
    pt = int(dialogue.meta.get("total_prompt_tokens", 0) or 0)
    ct = int(dialogue.meta.get("total_completion_tokens", 0) or 0)
    return QuestionResult(
        qid=question.qid,
        gold=question.gold_answer,
        final_answer=final,
        correct=correct,
        prompt_tokens=pt,
        completion_tokens=ct,
        total_tokens=pt + ct,
        meta={
            "n_agents": dialogue.meta.get("n_agents"),
            "n_rounds": dialogue.meta.get("n_rounds"),
            "per_agent_final_answers": dialogue.meta.get("per_agent_final_answers"),
        },
    )


# ---------------------------------------------------------------------------
# Code conditions
# ---------------------------------------------------------------------------


_CODE_FENCE_RE = re.compile(
    r"```(?:python)?\s*\n(?P<code>.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def _extract_code(response: str) -> str:
    """Extract code from a model response — prefer Notes/Code schema,
    fall back to first python fence, fall back to whole text."""
    _, code = parse_code_message(response)
    if code:
        return code
    m = _CODE_FENCE_RE.search(response)
    if m:
        return m.group("code").rstrip()
    return response.strip()


def _judge_correct(
    judge: Judge, code: str, hidden_tests: list[TestCase], timeout_s: float = 10.0,
) -> tuple[bool, JudgeResult]:
    if not hidden_tests:
        # No hidden tests means we can't grade objectively.
        return False, JudgeResult(passed=(), errors=(), total=0, n_passed=0)
    result = judge.run(code, hidden_tests, timeout_s=timeout_s)
    return (result.n_passed == result.total and result.total > 0), result


def run_sa_code(
    question: CodeQuestion, cell: CellSpec, llm_client: LLMClient,
    judge: Judge,
) -> QuestionResult:
    messages = [
        {"role": "system", "content": CODE_SA_SYSTEM_PROMPT},
        {"role": "user", "content": question.prompt},
    ]
    result = llm_client.chat_full(
        messages, cell.model, temperature=0.0, thinking=cell.thinking,
    )
    code = _extract_code(result.response)
    correct, judge_res = _judge_correct(judge, code, question.hidden_tests)
    pt = result.prompt_tokens or 0
    ct = result.completion_tokens or 0
    return QuestionResult(
        qid=question.qid,
        gold=str(question.entry_point),
        final_answer=code,
        correct=correct,
        prompt_tokens=pt,
        completion_tokens=ct,
        total_tokens=pt + ct,
        meta={"hidden_pass": judge_res.n_passed,
              "hidden_total": judge_res.total},
    )


def run_voting_q_code(
    question: CodeQuestion, cell: CellSpec, llm_client: LLMClient,
    n_samples: int, judge: Judge,
) -> QuestionResult:
    """Functional-clustering voting on code: collect N samples, cluster
    by public-test signature, evaluate the largest-cluster representative
    on hidden tests."""
    voting = run_voting(
        question=question.prompt,
        n_samples=n_samples,
        llm_client=llm_client,
        model=cell.model,
        system_prompt=CODE_SA_SYSTEM_PROMPT,
        thinking=cell.thinking,
        # Use identity parser so the raw response (which may be
        # multi-line code) is preserved; we'll cluster via Judge.
        parser=lambda text: text,
    )
    code_samples = [_extract_code(s) for s in voting.samples]
    if question.public_tests:
        clustering = cluster_by_signature(
            code_samples, judge, question.public_tests,
        )
        representative = clustering.representative_sample
    else:
        representative = code_samples[0] if code_samples else ""
    correct, judge_res = _judge_correct(judge, representative, question.hidden_tests)
    return QuestionResult(
        qid=question.qid,
        gold=str(question.entry_point),
        final_answer=representative,
        correct=correct,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=voting.total_tokens,
        meta={
            "n_samples": n_samples,
            "hidden_pass": judge_res.n_passed,
            "hidden_total": judge_res.total,
        },
    )


def run_debate_q_code(
    question: CodeQuestion, cell: CellSpec, llm_client: LLMClient,
    judge: Judge,
) -> QuestionResult:
    dialogue = run_code_debate(
        question=question,
        llm_client=llm_client,
        model=cell.model,
        n_agents=3, n_rounds=3,
        temperature=0.0,
        thinking=cell.thinking,
    )
    final_round = max(m.round for m in dialogue.messages)
    round_n_codes = [m.code for m in dialogue.messages
                     if m.round == final_round and m.code]
    if question.public_tests and round_n_codes:
        clustering = cluster_by_signature(
            round_n_codes, judge, question.public_tests,
        )
        representative = clustering.representative_sample
    else:
        representative = round_n_codes[0] if round_n_codes else ""
    correct, judge_res = _judge_correct(judge, representative, question.hidden_tests)
    pt = int(dialogue.meta.get("total_prompt_tokens", 0) or 0)
    ct = int(dialogue.meta.get("total_completion_tokens", 0) or 0)
    return QuestionResult(
        qid=question.qid,
        gold=str(question.entry_point),
        final_answer=representative,
        correct=correct,
        prompt_tokens=pt,
        completion_tokens=ct,
        total_tokens=pt + ct,
        meta={
            "n_agents": dialogue.meta.get("n_agents"),
            "n_rounds": dialogue.meta.get("n_rounds"),
            "hidden_pass": judge_res.n_passed,
            "hidden_total": judge_res.total,
        },
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_condition(
    results: list[QuestionResult],
    cell: CellSpec,
    condition: ConditionName,
    extra_meta: Optional[dict[str, Any]] = None,
) -> ConditionRecord:
    n = len(results)
    n_correct = sum(1 for r in results if r.correct)
    accuracy = n_correct / n if n else 0.0
    total_tokens = sum(r.total_tokens for r in results)
    return ConditionRecord(
        condition=condition,
        cell=cell,
        questions=results,
        n_evaluated=n,
        accuracy=accuracy,
        total_tokens=total_tokens,
        meta=extra_meta or {},
    )
