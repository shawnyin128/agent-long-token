"""Concrete Dataset adapters for code + math benchmarks.

LiveCodeBench: contest-date filtered (>= 2024-08), cap 80.
HumanEval+: prefers evalplus for canonical test suite.
AIMEMultiYear: custom JSON loader, year-stratified 30/30/20 with seed 42.
GSM8K: thin wrapper around agentdiet.dataset.load_gsm8k.
"""
from __future__ import annotations

import json
import random
from datetime import date, datetime
from pathlib import Path
from typing import Literal, Optional

from agentdiet.dataset import Question
from agentdiet.eval.base import CodeQuestion, TestCase


LIVECODEBENCH_CUTOFF = date(2024, 8, 1)
LIVECODEBENCH_CAP = 80


# ---------------------------------------------------------------------------
# LiveCodeBench
# ---------------------------------------------------------------------------


class LiveCodeBenchDataset:
    """Loads contest problems from a JSON fixture or via the
    `livecodebench` package. Problems are filtered to contest_date
    >= 2024-08-01 and capped at 80.

    Fixture JSON schema:
      [{"qid": str, "contest_date": "YYYY-MM-DD", "prompt": str,
        "entry_point": str,
        "public_tests": [{"name": str, "script": str}, ...],
        "hidden_tests": [{"name": str, "script": str}, ...]}, ...]
    """
    name = "livecodebench"
    domain: Literal["code"] = "code"

    def __init__(self, fixture_path: Optional[Path] = None,
                 cap: int = LIVECODEBENCH_CAP):
        self._fixture_path = Path(fixture_path) if fixture_path else None
        self._cap = cap

    def load(self) -> list[CodeQuestion]:
        if self._fixture_path is not None:
            raw = json.loads(self._fixture_path.read_text(encoding="utf-8"))
        else:
            raw = self._load_from_package()
        kept: list[CodeQuestion] = []
        for entry in raw:
            d = _parse_iso_date(entry["contest_date"])
            if d < LIVECODEBENCH_CUTOFF:
                continue
            kept.append(_dict_to_code_question(entry))
            if len(kept) >= self._cap:
                break
        return kept

    @staticmethod
    def _load_from_package() -> list[dict]:
        """Load LiveCodeBench by directly downloading the JSONL data
        files from the livecodebench/code_generation_lite HF repo.

        We can't use `datasets.load_dataset` here because that repo
        ships a dataset-loading script (`code_generation_lite.py`) and
        modern HF `datasets` refuses to execute scripts. Instead, we
        pull the per-release JSONL shards via `huggingface_hub`:
          - release_v1: test.jsonl
          - release_v2: + test2.jsonl
          - release_v3: + test3.jsonl
          - release_v4: + test4.jsonl
          - release_v5: + test5.jsonl
          - release_v6: + test6.jsonl
        and parse each line as a row. We default to the full v6 set so
        we get all contests through 2025; the contest_date filter
        (>= 2024-08-01) trims pre-cutoff problems downstream.
        """
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub package required for LiveCodeBench; "
                "install agentdiet[code_eval] (or `pip install huggingface_hub`)"
            ) from exc

        repo_id = "livecodebench/code_generation_lite"
        shards = ("test.jsonl", "test2.jsonl", "test3.jsonl",
                  "test4.jsonl", "test5.jsonl", "test6.jsonl")

        local_paths: list[str] = []
        last_err: Exception | None = None
        for fname in shards:
            try:
                p = hf_hub_download(repo_id=repo_id, filename=fname,
                                    repo_type="dataset")
                local_paths.append(p)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                # Older shards may be absent in some repo states; keep
                # going. We only error if ALL shards failed.
                continue

        if not local_paths:
            raise RuntimeError(
                "could not download any livecodebench/code_generation_lite "
                f"shard from HF: {last_err}"
            ) from last_err

        rows: list[dict] = []
        for path in local_paths:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        rows.append(_lcb_hf_row_to_dict(rec))
                    except Exception:  # noqa: BLE001
                        # Skip malformed/unparseable rows
                        continue
        return rows


def _lcb_hf_row_to_dict(r: dict) -> dict:
    """Convert one HF LiveCodeBench row to our LCB dict schema."""
    qid = str(r["question_id"])
    contest_date = str(r["contest_date"])[:10]  # ISO YYYY-MM-DD slice
    prompt = str(r["question_content"])
    starter = str(r.get("starter_code") or "")
    platform = str(r.get("platform") or "").lower()

    # Functional (leetcode) starter_code defines `class Solution: def NAME(...)`.
    # Parse the method name; for stdin problems, we never use it.
    entry_point = _lcb_parse_method_name(starter) or "main"

    public_raw = _lcb_decode_test_cases(r.get("public_test_cases"))
    hidden_raw = _lcb_decode_test_cases(r.get("private_test_cases"))

    public_tests = [_lcb_test_to_testcase(tc, i, entry_point, prefix="pub")
                    for i, tc in enumerate(public_raw)]
    hidden_tests = [_lcb_test_to_testcase(tc, i, entry_point, prefix="hid")
                    for i, tc in enumerate(hidden_raw)]

    return {
        "qid": qid,
        "contest_date": contest_date,
        "prompt": prompt + ("\n\n" + starter if starter else ""),
        "entry_point": entry_point,
        "public_tests": public_tests,
        "hidden_tests": hidden_tests,
        "meta": {"platform": platform,
                 "difficulty": str(r.get("difficulty") or "")},
    }


def _lcb_decode_test_cases(field) -> list[dict]:
    """Decode the test_cases field.

    LCB stores public_test_cases as a JSON-encoded string, and
    private_test_cases as base64(zlib(pickle(<JSON-encoded string>))).
    Note that the inner pickle payload is itself a JSON string, NOT a
    list — we have to json.loads it after unpickling.
    """
    if field is None:
        return []
    if isinstance(field, list):
        return field
    if not isinstance(field, str) or not field:
        return []
    s = field.strip()
    # Plain JSON path (public test cases)
    try:
        out = json.loads(s)
        if isinstance(out, list):
            return out
        if isinstance(out, str):
            inner = json.loads(out)
            if isinstance(inner, list):
                return inner
    except Exception:  # noqa: BLE001
        pass
    # zlib+base64+pickle path (private test cases). Pickle payload is a
    # JSON string we still need to parse.
    try:
        import base64
        import pickle
        import zlib
        data = pickle.loads(zlib.decompress(base64.b64decode(s)))
        if isinstance(data, list):
            return data
        if isinstance(data, str):
            inner = json.loads(data)
            if isinstance(inner, list):
                return inner
    except Exception:  # noqa: BLE001
        pass
    return []


def _lcb_parse_method_name(starter_code: str) -> str:
    """Extract method name from leetcode starter code 'def NAME(self, ...)'."""
    import re
    m = re.search(r"def\s+([A-Za-z_]\w*)\s*\(\s*self\b", starter_code)
    return m.group(1) if m else ""


def _lcb_test_to_testcase(tc: dict, idx: int, entry_point: str,
                           prefix: str = "tc") -> dict:
    """Convert one LCB test case to our {name, script} dict."""
    testtype = str(tc.get("testtype") or "stdin").lower()
    inp = tc.get("input") or ""
    out = tc.get("output") or ""
    if testtype == "functional":
        # input is JSON-encoded args; output is JSON-encoded expected return.
        # User code defines class Solution with method `entry_point`.
        # The harness pre-execs user_code so Solution is in globals.
        script = _LCB_FUNCTIONAL_SCRIPT.format(
            entry_point=entry_point,
            input_json=repr(str(inp)),
            output_json=repr(str(out)),
        )
    else:
        # stdin: input is stdin string, output is expected stdout.
        # Run user code as a fresh subprocess with input piped in.
        script = _LCB_STDIN_SCRIPT.format(
            input_text=repr(str(inp)),
            expected_text=repr(str(out)),
        )
    return {"name": f"{prefix}_{idx}", "script": script}


_LCB_STDIN_SCRIPT = '''\
import os, subprocess, sys, tempfile
INPUT = {input_text}
EXPECTED = {expected_text}
with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as __f:
    __f.write(__user_code__)
    __p = __f.name
try:
    __r = subprocess.run(
        [sys.executable, __p],
        input=INPUT, capture_output=True, text=True, timeout=8,
    )
finally:
    os.unlink(__p)
__got = (__r.stdout or "").strip()
__exp = EXPECTED.strip()
assert __got == __exp, "got: " + repr(__got)[:200] + " expected: " + repr(__exp)[:200]
'''

_LCB_FUNCTIONAL_SCRIPT = '''\
import json
__inp = json.loads({input_json})
__exp = json.loads({output_json})
__sol = Solution()
__method = getattr(__sol, "{entry_point}")
__got = __method(*__inp) if isinstance(__inp, list) else __method(__inp)
assert __got == __exp, "got: " + repr(__got)[:200] + " expected: " + repr(__exp)[:200]
'''


def _parse_iso_date(s: str) -> date:
    return datetime.fromisoformat(s).date()


def _dict_to_code_question(d: dict) -> CodeQuestion:
    return CodeQuestion(
        qid=str(d["qid"]),
        prompt=str(d["prompt"]),
        entry_point=str(d["entry_point"]),
        public_tests=[TestCase(name=t["name"], script=t["script"])
                      for t in d.get("public_tests", [])],
        hidden_tests=[TestCase(name=t["name"], script=t["script"])
                      for t in d.get("hidden_tests", [])],
        meta=d.get("meta", {}),
    )


# ---------------------------------------------------------------------------
# HumanEval+
# ---------------------------------------------------------------------------


class HumanEvalPlusDataset:
    """Loads HumanEval+ via evalplus.data.get_human_eval_plus().

    Optional fixture_path lets tests inject a dict-shaped fixture
    matching evalplus's return value:
      {task_id: {"prompt": str, "entry_point": str,
                  "test": str,  # full test block
                  "canonical_solution": str},
       ...}
    Public tests are extracted from the docstring examples in `prompt`;
    hidden tests are the `test` block.
    """
    name = "humaneval_plus"
    domain: Literal["code"] = "code"

    def __init__(self, fixture: Optional[dict] = None,
                 fixture_path: Optional[Path] = None,
                 cap: int = 80):
        self._fixture = fixture
        self._fixture_path = Path(fixture_path) if fixture_path else None
        self._cap = cap

    def load(self) -> list[CodeQuestion]:
        data = self._get_data()
        questions: list[CodeQuestion] = []
        for task_id, task in data.items():
            prompt = task["prompt"]
            entry_point = task.get("entry_point") or _entry_point_from_prompt(prompt)
            public = _extract_docstring_tests(prompt, entry_point)
            hidden = [TestCase(name="hidden", script=task["test"])] if task.get("test") else []
            questions.append(CodeQuestion(
                qid=str(task_id),
                prompt=prompt,
                entry_point=entry_point,
                public_tests=public,
                hidden_tests=hidden,
                meta={"source": "humaneval_plus"},
            ))
            if len(questions) >= self._cap:
                break
        return questions

    def _get_data(self) -> dict:
        if self._fixture is not None:
            return self._fixture
        if self._fixture_path is not None:
            return json.loads(self._fixture_path.read_text(encoding="utf-8"))
        try:
            from evalplus.data import get_human_eval_plus  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "evalplus not installed; "
                "install agentdiet[code_eval] or pass fixture/fixture_path"
            ) from exc
        return get_human_eval_plus()


def _entry_point_from_prompt(prompt: str) -> str:
    """Best-effort: parse `def NAME(` from prompt."""
    import re
    m = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", prompt)
    return m.group(1) if m else "candidate"


def _extract_docstring_tests(prompt: str, entry_point: str) -> list[TestCase]:
    """Pull `>>> NAME(...)` doctest examples from the prompt as TestCases.

    Each line `>>> entry_point(args)` followed by `result` becomes
    `assert entry_point(args) == result`.
    """
    import re
    tests: list[TestCase] = []
    lines = prompt.splitlines()
    i = 0
    pattern = re.compile(r"\s*>>>\s*(" + re.escape(entry_point) + r"\(.*\))")
    while i < len(lines):
        m = pattern.match(lines[i])
        if m and i + 1 < len(lines):
            call = m.group(1)
            expected = lines[i + 1].strip()
            if expected and not expected.startswith(">>>"):
                tests.append(TestCase(
                    name=f"docstring_{len(tests)}",
                    script=f"assert {call} == {expected}",
                ))
            i += 2
        else:
            i += 1
    return tests


# ---------------------------------------------------------------------------
# Multi-year AIME
# ---------------------------------------------------------------------------


AIME_YEARS = (2026, 2025, 2024)
AIME_FULL_PER_YEAR = 30
AIME_2024_SAMPLE = 20
AIME_TOTAL = AIME_FULL_PER_YEAR * 2 + AIME_2024_SAMPLE  # 80


class AIMEMultiYearDataset:
    """80-question pool: AIME 2026 (30) + 2025 (30) + 2024 (sample 20, seed 42).

    Reads JSON files at data_dir / "aime_{year}.json" with schema
      {"year": int, "questions": [{"id": str, "problem": str,
                                     "answer": str}, ...]}
    """
    name = "aime_multi_year"
    domain: Literal["math"] = "math"

    def __init__(self, data_dir: Path, sample_seed: int = 42):
        self._data_dir = Path(data_dir)
        self._sample_seed = sample_seed

    def load(self) -> list[Question]:
        all_questions: list[Question] = []
        for year in AIME_YEARS:
            year_qs = self._load_year(year)
            if year == 2024:
                rng = random.Random(self._sample_seed)
                indices = sorted(rng.sample(range(len(year_qs)), AIME_2024_SAMPLE))
                year_qs = [year_qs[i] for i in indices]
            all_questions.extend(year_qs)
        if len(all_questions) != AIME_TOTAL:
            raise ValueError(
                f"AIME pool size {len(all_questions)} != expected {AIME_TOTAL}"
            )
        return all_questions

    def _load_year(self, year: int) -> list[Question]:
        path = self._data_dir / f"aime_{year}.json"
        if not path.exists():
            raise FileNotFoundError(f"AIME data file missing: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        questions = payload.get("questions", [])
        if len(questions) != AIME_FULL_PER_YEAR:
            raise ValueError(
                f"AIME {year} has {len(questions)} questions, "
                f"expected {AIME_FULL_PER_YEAR}"
            )
        return [
            _aime_dict_to_question(q, year, idx)
            for idx, q in enumerate(questions)
        ]


def _aime_dict_to_question(d: dict, year: int, idx: int) -> Question:
    qid = f"aime-{year}-q{idx:02d}"
    # Use object.__setattr__ via dataclass; Question is frozen so we
    # construct fresh and attach meta via the meta dict on a wrapper —
    # but Question doesn't have a meta field. We extend it with a thin
    # wrapper dict approach: Question is frozen with only qid/question/
    # gold_answer; year info is encoded in qid prefix and recoverable
    # by callers. For per-year stratification, callers can split on
    # qid prefix.
    return Question(
        qid=qid,
        question=str(d["problem"]),
        gold_answer=str(d["answer"]),
    )


# ---------------------------------------------------------------------------
# GSM8K (thin Dataset wrapper around agentdiet.dataset.load_gsm8k)
# ---------------------------------------------------------------------------


class GSM8KDataset:
    """Wraps agentdiet.dataset.load_gsm8k as a Dataset Protocol object.

    Defaults: split=test, n=80, seed=42 (RQ1 phase-mapping defaults).
    """
    name = "gsm8k"
    domain: Literal["math"] = "math"

    def __init__(self, split: str = "test", n: int = 80, seed: int = 42,
                 cache_dir: Optional[Path] = None):
        self._split = split
        self._n = n
        self._seed = seed
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None

    def load(self) -> list[Question]:
        from agentdiet.dataset import load_gsm8k
        return load_gsm8k(
            split=self._split, n=self._n, seed=self._seed,
            cache_dir=self._cache_dir,
        )
