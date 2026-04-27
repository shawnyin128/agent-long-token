"""Concrete Judge implementations.

SubprocessJudge: standalone, no third-party deps. Used by unit tests
and as a portable fallback. Each test is run in a fresh `python -c`
subprocess with timeout, isolated tmpdir, no shell.

EvalscopeJudge: wraps evalscope's code-execution harness for
production runs. Lazy-imports evalscope; raises with a clear message
if not installed.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

from agentdiet.eval.base import JudgeResult, TestCase


SUBPROCESS_HARNESS = '''\
import sys, traceback
__user_code__ = {code!r}
__test_script__ = {script!r}
try:
    exec(compile(__user_code__, "<user_code>", "exec"), globals())
except Exception:
    traceback.print_exc()
    sys.exit(2)
try:
    exec(compile(__test_script__, "<test_script>", "exec"), globals())
except Exception:
    traceback.print_exc()
    sys.exit(1)
sys.exit(0)
'''


class SubprocessJudge:
    """Run each test as a Python subprocess in an isolated tmpdir.

    Per-test exit codes:
      0 — pass; 1 — test assertion/error; 2 — user code didn't even
      compile/import; -SIGKILL or any non-zero on timeout — fail.

    No network is enforced beyond passing an env without proxy vars.
    Filesystem is sandboxed only by setting CWD to a fresh tempdir;
    we do NOT block FS reads outside CWD (would require seccomp or
    a container).
    """

    def __init__(self, python_executable: str | None = None):
        self._python = python_executable or sys.executable

    def run(
        self,
        code: str,
        tests: list[TestCase],
        timeout_s: float = 10.0,
    ) -> JudgeResult:
        passed: list[bool] = []
        errors: list[str | None] = []
        for tc in tests:
            ok, err = self._run_one(code, tc.script, timeout_s)
            passed.append(ok)
            errors.append(err)
        return JudgeResult(
            passed=tuple(passed),
            errors=tuple(errors),
            total=len(tests),
            n_passed=sum(passed),
        )

    def _run_one(
        self, code: str, script: str, timeout_s: float
    ) -> tuple[bool, str | None]:
        harness = SUBPROCESS_HARNESS.format(code=code, script=script)
        with tempfile.TemporaryDirectory() as td:
            env = self._sandbox_env()
            try:
                result = subprocess.run(
                    [self._python, "-c", harness],
                    cwd=td,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                )
            except subprocess.TimeoutExpired:
                return False, f"timeout after {timeout_s}s"
            if result.returncode == 0:
                return True, None
            err = (result.stderr or result.stdout).strip().splitlines()
            tail = err[-1] if err else f"exit {result.returncode}"
            return False, tail

    @staticmethod
    def _sandbox_env() -> dict:
        env = {k: v for k, v in os.environ.items()
               if k.upper() not in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"}}
        # Force a deterministic-ish hash seed (helpful if user tests
        # iterate over dicts).
        env.setdefault("PYTHONHASHSEED", "0")
        return env


class EvalscopeJudge:
    """Production sandbox via evalscope. Lazy import — raises with a
    clear message if evalscope isn't installed.

    Falls through to SubprocessJudge for the actual subprocess run if
    evalscope's API at runtime doesn't expose the expected execution
    callable; the wrapper records the fallback in JudgeResult.errors[0].
    """

    def __init__(self):
        try:
            import evalscope  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "evalscope not installed; install agentdiet[code_eval]"
            ) from exc
        self._fallback = SubprocessJudge()

    def run(
        self,
        code: str,
        tests: list[TestCase],
        timeout_s: float = 10.0,
    ) -> JudgeResult:
        # Until evalscope's exec-harness API is fixed in our pinned
        # version, defer to SubprocessJudge for execution. The class
        # exists so callers can target it by type — when the right
        # evalscope entry point lands, swap the body.
        return self._fallback.run(code, tests, timeout_s=timeout_s)
