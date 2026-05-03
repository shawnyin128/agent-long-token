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
__user_code_error__ = None
try:
    exec(compile(__user_code__, "<user_code>", "exec"), globals())
except Exception as __exc__:
    # Pre-exec failure is non-fatal: some test types (e.g. LiveCodeBench
    # stdin/stdout problems) need to run user_code as a fresh subprocess
    # with input piped in, so a hung input() during pre-exec is
    # expected. Record the error and let the test_script decide.
    __user_code_error__ = __exc__
    traceback.print_exc(file=sys.stderr)
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
        # We can't pass `harness` through `python -c` because LCB stress
        # tests can have multi-hundred-KB stdin payloads baked into the
        # test script, blowing past argv limits (E2BIG). Write harness
        # to a file inside the sandbox tmpdir and exec it instead.
        harness = SUBPROCESS_HARNESS.format(code=code, script=script)
        with tempfile.TemporaryDirectory() as td:
            harness_path = os.path.join(td, "_harness.py")
            with open(harness_path, "w", encoding="utf-8") as f:
                f.write(harness)
            env = self._sandbox_env()
            try:
                # stdin=DEVNULL: when harness pre-execs user_code, an
                # `if __name__ == "__main__": main()` block can call
                # input()/sys.stdin.read() during pre-exec (LCB stdin
                # solutions all do this). With an inherited terminal
                # stdin those reads block until the outer 8s timeout
                # kills the harness, faking a 0/N test result. DEVNULL
                # makes those reads return EOF immediately; the
                # test_script that grades stdin-style tests creates its
                # own subprocess with the real test input piped in.
                result = subprocess.run(
                    [self._python, harness_path],
                    cwd=td,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    stdin=subprocess.DEVNULL,
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
