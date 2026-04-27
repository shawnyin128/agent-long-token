"""Dataset and Judge Protocols are runtime-checkable; JudgeResult exposes
.signature for clustering."""
from __future__ import annotations

from agentdiet.eval import (
    CodeQuestion,
    Dataset,
    Judge,
    JudgeResult,
    TestCase,
)


class _FakeCodeDataset:
    name = "fake"
    domain = "code"

    def load(self):
        return []


class _FakeJudge:
    def run(self, code, tests, timeout_s=10.0):
        return JudgeResult(
            passed=tuple(True for _ in tests),
            errors=tuple(None for _ in tests),
            total=len(tests),
            n_passed=len(tests),
        )


def test_fake_code_dataset_is_a_dataset():
    assert isinstance(_FakeCodeDataset(), Dataset)


def test_fake_judge_is_a_judge():
    assert isinstance(_FakeJudge(), Judge)


def test_judge_result_signature_is_passed_tuple():
    r = JudgeResult(
        passed=(True, False, True),
        errors=(None, "AssertionError", None),
        total=3,
        n_passed=2,
    )
    assert r.signature == (True, False, True)


def test_judge_result_pass_at_1():
    r = JudgeResult(passed=(True,), errors=(None,), total=1, n_passed=1)
    assert r.pass_at_1 == 1.0
    r0 = JudgeResult(passed=(False,), errors=("err",), total=1, n_passed=0)
    assert r0.pass_at_1 == 0.0
    r_empty = JudgeResult(passed=(), errors=(), total=0, n_passed=0)
    assert r_empty.pass_at_1 == 0.0


def test_code_question_holds_test_cases():
    q = CodeQuestion(
        qid="q1",
        prompt="def add(a,b): ...",
        entry_point="add",
        public_tests=[TestCase(name="ex", script="assert add(1,2)==3")],
        hidden_tests=[TestCase(name="h1", script="assert add(0,0)==0")],
    )
    assert q.public_tests[0].name == "ex"
    assert q.hidden_tests[0].script.startswith("assert")
