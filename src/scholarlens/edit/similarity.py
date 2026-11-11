"""Semantic-similarity gate between the author's paragraph and an edit.

The threshold depends on the embedding model: calibrate it on Bench data for
whichever embedder you use (``scholarlens.bench`` reports the distribution of
similarities for human-validated edits).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class SimilarityScorer(Protocol):
    name: str

    def __call__(self, a: str, b: str) -> float: ...


class EmbeddingSimilarity:
    def __init__(self, embedder, name: str = "embedding") -> None:
        self.embedder = embedder
        self.name = name

    def __call__(self, a: str, b: str) -> float:
        if a.strip() == b.strip():
            return 1.0
        e = np.asarray(self.embedder.encode([a, b]), dtype=float)
        na, nb = np.linalg.norm(e[0]), np.linalg.norm(e[1])
        return float(e[0] @ e[1] / (na * nb)) if na and nb else 0.0


def sentence_transformer_similarity(model: str = "sentence-transformers/all-MiniLM-L6-v2") -> EmbeddingSimilarity:
    from scholarlens.features.transitions import SentenceTransformerEmbedder

    return EmbeddingSimilarity(SentenceTransformerEmbedder(model), name=model)
