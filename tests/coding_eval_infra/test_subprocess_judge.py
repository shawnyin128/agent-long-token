"""SubprocessJudge runs each test in an isolated Python subprocess."""
from __future__ import annotations

import pytest

from agentdiet.eval.base import TestCase
from agentdiet.eval.judges import SubprocessJudge


def _judge():
    return SubprocessJudge()


def test_correct_code_all_pass():
    code = "def add(a, b): return a + b"
    tests = [
        TestCase(name="t1", script="assert add(2, 3) == 5"),
        TestCase(name="t2", script="assert add(0, 0) == 0"),
    ]
    result = _judge().run(code, tests)
    assert result.passed == (True, True)
    assert result.n_passed == 2
    assert result.total == 2
    assert result.pass_at_1 == 1.0


def test_incorrect_code_some_fail():
    code = "def add(a, b): return a + b"
    tests = [
        TestCase(name="t1", script="assert add(2, 3) == 5"),
        TestCase(name="t2", script="assert add(2, 3) == 99"),  # wrong
    ]
    result = _judge().run(code, tests)
    assert result.passed == (True, False)
    assert result.n_passed == 1
    assert result.errors[1] is not None  # has error message


def test_user_code_compile_error_marks_all_fail():
    code = "def broken(:"  # syntax error
    tests = [TestCase(name="t1", script="assert broken(1) == 1")]
    result = _judge().run(code, tests)
    assert result.passed == (False,)
    assert result.errors[0] is not None


def test_timeout_within_specified_time():
    code = "def loop(): \n    while True: pass"
    tests = [TestCase(name="t1", script="loop()")]
    import time
    t0 = time.monotonic()
    result = _judge().run(code, tests, timeout_s=1.0)
    elapsed = time.monotonic() - t0
    # Should not run far beyond the timeout
    assert elapsed < 5.0
    assert result.passed == (False,)
    assert "timeout" in (result.errors[0] or "").lower()


def test_runs_in_isolated_cwd(tmp_path):
    """Code's CWD is a fresh tempdir, not the project root."""
    code = "import os; ROOT = os.getcwd()"
    tests = [TestCase(name="t", script="assert ROOT != '/' and 'final-project' not in ROOT")]
    result = _judge().run(code, tests)
    assert result.passed == (True,)


def test_no_proxy_env_propagates():
    """HTTPS_PROXY / HTTP_PROXY are stripped from sandbox env."""
    code = "import os"
    tests = [
        TestCase(name="t", script="assert os.environ.get('HTTPS_PROXY') is None"),
        TestCase(name="t2", script="assert os.environ.get('HTTP_PROXY') is None"),
    ]
    result = _judge().run(code, tests)
    assert result.passed == (True, True)


def test_signature_drives_clustering():
    """JudgeResult.signature == passed tuple (used by clustering)."""
    code = "def f(x): return x * 2"
    tests = [
        TestCase(name="a", script="assert f(2) == 4"),
        TestCase(name="b", script="assert f(3) == 99"),  # fails
        TestCase(name="c", script="assert f(0) == 0"),
    ]
    result = _judge().run(code, tests)
    assert result.signature == (True, False, True)


def test_empty_test_list():
    result = _judge().run("def f(): return 1", tests=[])
    assert result.signature == ()
    assert result.total == 0
    assert result.n_passed == 0
    assert result.pass_at_1 == 0.0
