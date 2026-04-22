from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Question:
    qid: str
    question: str
    gold_answer: str


_ANSWER_RE_FINAL_HASHES = re.compile(r"####\s*(-?[\d,]+\.?\d*)")
_ANSWER_RE_DOLLAR = re.compile(r"(?:^|[^\d])\$\s*(-?[\d,]+\.?\d*)")
_ANSWER_RE_LAST_NUMBER = re.compile(r"(-?\d[\d,]*\.?\d*)")


def _clean_num(raw: str) -> Optional[str]:
    s = raw.replace(",", "").replace("−", "-").strip().strip(".")
    if not s or s in {"-", "."}:
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    if f.is_integer():
        return str(int(f))
    return s.rstrip("0").rstrip(".")


def parse_answer(text: str) -> Optional[str]:
    if text is None:
        return None
    t = text.strip()
    if not t:
        return None

    hash_matches = list(_ANSWER_RE_FINAL_HASHES.finditer(t))
    if hash_matches:
        return _clean_num(hash_matches[-1].group(1))

    m = _ANSWER_RE_DOLLAR.search(t)
    if m:
        return _clean_num(m.group(1))

    nums = _ANSWER_RE_LAST_NUMBER.findall(t)
    if nums:
        return _clean_num(nums[-1])
    return None


def _parse_gsm8k_gold(answer_field: str) -> str:
    matches = list(_ANSWER_RE_FINAL_HASHES.finditer(answer_field))
    if not matches:
        raise ValueError(f"GSM8K gold answer missing '#### N' marker: {answer_field!r}")
    cleaned = _clean_num(matches[-1].group(1))
    if cleaned is None:
        raise ValueError(f"GSM8K gold answer not numeric: {answer_field!r}")
    return cleaned


def load_gsm8k(
    split: str = "test",
    n: Optional[int] = None,
    seed: int = 42,
    cache_dir: Optional[Path] = None,
) -> list[Question]:
    from datasets import load_dataset

    kwargs = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    ds = load_dataset("gsm8k", "main", split=split, **kwargs)

    questions = []
    for i, row in enumerate(ds):
        try:
            gold = _parse_gsm8k_gold(row["answer"])
        except ValueError:
            continue
        questions.append(
            Question(qid=f"gsm8k-{split}-{i}", question=row["question"], gold_answer=gold)
        )

    if n is None or n >= len(questions):
        return questions
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(questions)), n))
    return [questions[i] for i in indices]
