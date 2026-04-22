from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentdiet.evaluate import MethodSummary, PerQuestionResult


def test_per_question_result_happy():
    r = PerQuestionResult(
        qid="q1", method="b1", compressed_tokens=128,
        final_answer="7", gold="7", correct=True,
    )
    assert r.method == "b1"
    assert r.correct is True


def test_per_question_result_rejects_invalid_method():
    with pytest.raises(ValidationError):
        PerQuestionResult(
            qid="q1", method="b99", compressed_tokens=10,
            final_answer="7", gold="7", correct=True,
        )


def test_method_summary_computes_acc_per_1k_correctly():
    s = MethodSummary.build(
        method="b1",
        per_question=[
            PerQuestionResult(qid=f"q{i}", method="b1", compressed_tokens=500,
                              final_answer="7", gold="7", correct=(i < 2))
            for i in range(4)
        ],
    )
    assert s.accuracy == 0.5           # 2 / 4 correct
    assert s.total_tokens == 2000      # 500 × 4
    assert s.acc_per_1k == pytest.approx(0.5 / 2.0)
    assert s.n_evaluated == 4


def test_method_summary_zero_tokens_gives_zero_acc_per_1k():
    s = MethodSummary.build(method="b1", per_question=[])
    assert s.n_evaluated == 0
    assert s.accuracy == 0.0
    assert s.total_tokens == 0
    assert s.acc_per_1k == 0.0


def test_method_summary_rejects_mixed_method_rows():
    mixed = [
        PerQuestionResult(qid="q1", method="b1", compressed_tokens=1,
                          final_answer="7", gold="7", correct=True),
        PerQuestionResult(qid="q2", method="b2", compressed_tokens=1,
                          final_answer="7", gold="7", correct=True),
    ]
    with pytest.raises(ValueError):
        MethodSummary.build(method="b1", per_question=mixed)
