"""HumanEvalPlus Dataset adapter."""
from __future__ import annotations

import pytest

from agentdiet.eval.datasets import HumanEvalPlusDataset


def _fixture():
    return {
        "HumanEval/0": {
            "prompt": (
                'def has_close_elements(numbers, threshold):\n'
                '    """ Check whether any pair is closer than threshold.\n'
                '    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n'
                '    False\n'
                '    >>> has_close_elements([1.0, 2.8, 3.0, 4.0], 0.3)\n'
                '    True\n'
                '    """\n'
            ),
            "entry_point": "has_close_elements",
            "test": (
                "def check(candidate):\n"
                "    assert candidate([1.0,2.0,3.0], 0.5) == False\n"
                "check(has_close_elements)\n"
            ),
            "canonical_solution": "    return False\n",
        },
        "HumanEval/1": {
            "prompt": (
                'def double(x):\n'
                '    """Return 2*x.\n'
                '    >>> double(3)\n'
                '    6\n'
                '    """\n'
            ),
            "entry_point": "double",
            "test": "assert double(4) == 8",
            "canonical_solution": "    return x * 2\n",
        },
    }


def test_loads_each_task_with_public_tests():
    ds = HumanEvalPlusDataset(fixture=_fixture())
    qs = ds.load()
    assert len(qs) == 2
    qid_to_q = {q.qid: q for q in qs}
    assert qid_to_q["HumanEval/0"].entry_point == "has_close_elements"
    assert qid_to_q["HumanEval/1"].entry_point == "double"


def test_extracts_doctest_examples_as_public_tests():
    ds = HumanEvalPlusDataset(fixture=_fixture())
    qs = ds.load()
    q0 = next(q for q in qs if q.qid == "HumanEval/0")
    assert len(q0.public_tests) == 2
    assert "has_close_elements" in q0.public_tests[0].script
    q1 = next(q for q in qs if q.qid == "HumanEval/1")
    assert len(q1.public_tests) == 1
    assert q1.public_tests[0].script == "assert double(3) == 6"


def test_hidden_tests_are_evalplus_test_block():
    ds = HumanEvalPlusDataset(fixture=_fixture())
    qs = ds.load()
    q1 = next(q for q in qs if q.qid == "HumanEval/1")
    assert len(q1.hidden_tests) == 1
    assert "assert double(4) == 8" in q1.hidden_tests[0].script


def test_dataset_attrs():
    ds = HumanEvalPlusDataset(fixture={})
    assert ds.name == "humaneval_plus"
    assert ds.domain == "code"


def test_cap_caps_loaded_count():
    fixture = {f"HumanEval/{i}": {
        "prompt": f"def f{i}(): pass\n",
        "entry_point": f"f{i}",
        "test": "assert True",
    } for i in range(20)}
    ds = HumanEvalPlusDataset(fixture=fixture, cap=5)
    qs = ds.load()
    assert len(qs) == 5


def test_no_fixture_no_evalplus_raises():
    """Without fixture and without evalplus installed,
    load() should raise ImportError pointing at the extras group."""
    ds = HumanEvalPlusDataset(fixture=None, fixture_path=None)
    with pytest.raises(ImportError, match="evalplus"):
        ds.load()
