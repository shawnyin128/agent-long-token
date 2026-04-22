"""Independent per-claim signals.

Four signals per spec §5.1, stored as **independent fields** with NO
composite score (spec §5.2 prohibits it):

  * ``flip_coincidence`` — bool; claim.round matches a FlipEvent.round
  * ``novelty`` — 1 - max cosine similarity to any strictly-earlier claim
  * ``referenced_later`` — any strictly-later claim has cos sim ≥ 0.7
  * ``position`` — claim.round (int)

Cosine similarity is computed on L2-normalized embeddings (one dot
product). Embedders conform to the ``Embedder`` protocol; the default
``SentenceTransformerEmbedder`` wraps ``all-MiniLM-L6-v2`` behind a
lazy import, and ``HashingFakeEmbedder`` is a deterministic,
network-free fallback for tests and offline environments.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Iterable, Protocol

import numpy as np

from agentdiet.types import FlipEvent


log = logging.getLogger(__name__)


REFERENCED_LATER_THRESHOLD = 0.7


SIGNAL_KEYS: tuple[str, ...] = (
    "claim_id",
    "flip_coincidence",
    "novelty",
    "referenced_later",
    "position",
)


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:  # noqa: D401
        """Return shape (N, D) array of L2-normalized row vectors."""


class HashingFakeEmbedder:
    """Deterministic, network-free embedder for tests and offline use.

    Hashes each text to a fixed-length byte vector and L2-normalizes.
    Produces consistent similarity structure (identical texts → vector
    equality; distinct texts → low but non-zero similarity), enough to
    verify signal semantics without a real sentence-transformer.
    """

    def __init__(self, dim: int = 64):
        self.dim = int(dim)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)
        vecs = np.zeros((len(texts), self.dim), dtype=np.float64)
        for i, t in enumerate(texts):
            digest = hashlib.sha256(t.encode("utf-8")).digest()
            # Repeat or truncate digest to fill dim bytes.
            needed = self.dim
            blob = (digest * ((needed // len(digest)) + 1))[:needed]
            # Map uint8 [0,255] -> signed float centered at 0 so distinct
            # texts produce near-orthogonal vectors instead of all-positive
            # vectors (which would have a high baseline cosine similarity).
            raw = np.frombuffer(blob, dtype=np.uint8).astype(np.float64)
            vecs[i] = raw - 127.5
            n = np.linalg.norm(vecs[i])
            if n > 0:
                vecs[i] = vecs[i] / n
        return vecs


class SentenceTransformerEmbedder:
    """Wraps sentence-transformers/all-MiniLM-L6-v2. Lazy import so
    the module still loads without the ``[analysis]`` extras installed.
    Raises ImportError on first ``encode`` if the package is missing."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed; "
                "run `pip install -e .[analysis]` or use HashingFakeEmbedder"
            ) from e
        self._model = SentenceTransformer(self.model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        self._ensure()
        assert self._model is not None
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float64)


def _claim_sort_key(c: dict[str, Any]) -> tuple[int, int, str]:
    return (int(c["round"]), int(c["agent_id"]), str(c["id"]))


def compute_signals(
    claims: list[dict[str, Any]],
    *,
    flip_events: Iterable[FlipEvent],
    embedder: Embedder,
) -> list[dict[str, Any]]:
    if not claims:
        return []

    ordered = sorted(claims, key=_claim_sort_key)
    texts = [c["text"] for c in ordered]
    embs = embedder.encode(texts)
    n = len(ordered)

    flip_rounds = {int(fe.round) for fe in flip_events}

    # Pairwise similarity via matmul on L2-normalized rows.
    # sim[i, j] = cos_sim(claim_i, claim_j).
    sim = embs @ embs.T

    rows: list[dict[str, Any]] = []
    for i, c in enumerate(ordered):
        if i == 0:
            novelty = 1.0
        else:
            novelty = 1.0 - float(np.max(sim[i, :i]))
        if i == n - 1:
            referenced_later = False
        else:
            referenced_later = bool(
                np.any(sim[i, i + 1 :] >= REFERENCED_LATER_THRESHOLD)
            )
        rows.append({
            "claim_id": c["id"],
            "flip_coincidence": int(c["round"]) in flip_rounds,
            "novelty": max(0.0, min(1.0, novelty)),
            "referenced_later": referenced_later,
            "position": int(c["round"]),
        })
    return rows
