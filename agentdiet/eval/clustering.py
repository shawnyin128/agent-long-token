"""Functional clustering of code samples (spec §5.4).

Each sample is run against `public_tests` via the supplied Judge;
the resulting JudgeResult.signature (tuple of bool) is the clustering
key. Samples sharing an exact signature go into one cluster. The
representative for the largest cluster is returned; ties are broken
by smallest-lex signature (False<True; deterministic per spec).

Used by:
  - the voting baseline (cluster N independent samples, pick
    representative, evaluate against hidden tests)
  - debate aggregation for code (cluster the round-N code samples)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agentdiet.eval.base import Judge, TestCase


Signature = tuple[bool, ...]


@dataclass(frozen=True)
class ClusteringResult:
    representative_index: int
    representative_sample: str
    signature: Signature
    cluster_size: int
    all_clusters: dict[Signature, list[int]] = field(default_factory=dict)


def cluster_by_signature(
    samples: list[str],
    judge: Judge,
    public_tests: list[TestCase],
    timeout_s: float = 10.0,
) -> ClusteringResult:
    if not samples:
        raise ValueError("samples must be non-empty")

    sig_to_indices: dict[Signature, list[int]] = {}
    for idx, sample in enumerate(samples):
        result = judge.run(sample, public_tests, timeout_s=timeout_s)
        sig = result.signature
        sig_to_indices.setdefault(sig, []).append(idx)

    # Pick winner: largest cluster; ties → smallest-lex signature.
    # Convert bool tuples to int tuples for consistent ordering.
    def _sort_key(item: tuple[Signature, list[int]]) -> tuple[int, tuple[int, ...]]:
        sig, indices = item
        # Negative size so sorted ascending gives largest first; lex
        # ordering on int-cast signature gives False (0) before True (1).
        return (-len(indices), tuple(int(b) for b in sig))

    ordered = sorted(sig_to_indices.items(), key=_sort_key)
    winner_sig, winner_indices = ordered[0]
    rep_idx = winner_indices[0]

    return ClusteringResult(
        representative_index=rep_idx,
        representative_sample=samples[rep_idx],
        signature=winner_sig,
        cluster_size=len(winner_indices),
        all_clusters=sig_to_indices,
    )
