"""Eval framework Protocols + shared dataclasses.

Datasets produce questions (math) or code-questions (code). Judges
take a generated solution and a list of test cases and return a
JudgeResult whose .signature property is what functional clustering
keys on (spec §5.4).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable


@dataclass(frozen=True)
class TestCase:
    """A single executable test snippet for a code question.

    `script` is a Python statement (typically an `assert` line) that
    must succeed for the test to pass. `name` is for traceability.
    For LiveCodeBench-style I/O tests, callers wrap (input, expected)
    into an assert against the entry-point function.
    """
    name: str
    script: str

    # Tell pytest not to try collecting this dataclass as a test class
    # just because its name starts with "Test".
    __test__ = False


@dataclass(frozen=True)
class CodeQuestion:
    qid: str
    prompt: str
    entry_point: str
    public_tests: list[TestCase]
    hidden_tests: list[TestCase]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JudgeResult:
    passed: tuple[bool, ...]
    errors: tuple[str | None, ...]
    total: int
    n_passed: int

    @property
    def signature(self) -> tuple[bool, ...]:
        return self.passed

    @property
    def pass_at_1(self) -> float:
        return self.n_passed / self.total if self.total else 0.0


@runtime_checkable
class Dataset(Protocol):
    name: str
    domain: Literal["math", "code"]

    def load(self) -> list[Any]: ...


@runtime_checkable
class Judge(Protocol):
    def run(
        self,
        code: str,
        tests: list[TestCase],
        timeout_s: float = 10.0,
    ) -> JudgeResult: ...
