"""Retrieval defence (Krishna et al., 2023, "Paraphrasing evades detectors of
AI-generated text, but retrieval is an effective defense").

A model provider keeps an index of the text it generated. A candidate text is
flagged when it is semantically close to a stored generation: paraphrasing
changes surface statistics but keeps meaning, so it remains retrievable.

In Bench, the index holds the original AI generations; paraphrased versions
are queries. The score is the maximum cosine similarity to the index.
"""

from __future__ import annotations

import numpy as np


class RetrievalDetector:
    name = "retrieval"

    def __init__(self, embedder, *, batch_size: int = 256) -> None:
        self.embedder = embedder
        self.batch_size = batch_size
        self._index: np.ndarray | None = None
        self._ids: list[str] = []

    def _encode(self, texts: list[str]) -> np.ndarray:
        chunks = [np.asarray(self.embedder.encode(texts[i:i + self.batch_size]), dtype=np.float32)
                  for i in range(0, len(texts), self.batch_size)]
        e = np.vstack(chunks) if chunks else np.zeros((0, 1), dtype=np.float32)
        norms = np.linalg.norm(e, axis=1, keepdims=True)
        return e / np.where(norms == 0, 1, norms)

    def add(self, texts: list[str], ids: list[str] | None = None) -> None:
        e = self._encode(texts)
        self._index = e if self._index is None else np.vstack([self._index, e])
        self._ids += ids or [str(len(self._ids) + i) for i in range(len(texts))]

    def search(self, texts: list[str], *, exclude: list[set[str]] | None = None) -> tuple[np.ndarray, list[str | None]]:
        """Max similarity and best-matching id per query. ``exclude[i]`` drops ids for query i
        (e.g. the query's own id when the query is itself in the index)."""
        if self._index is None or not texts:
            return np.zeros(len(texts)), [None] * len(texts)
        sims = self._encode(texts) @ self._index.T
        best, best_ids = np.zeros(len(texts)), []
        for i in range(len(texts)):
            row = sims[i].copy()
            if exclude and exclude[i]:
                for j, rid in enumerate(self._ids):
                    if rid in exclude[i]:
                        row[j] = -1.0
            k = int(np.argmax(row))
            best[i] = row[k]
            best_ids.append(self._ids[k])
        return best, best_ids

    def score(self, texts: list[str]) -> np.ndarray:
        return self.search(texts)[0]
