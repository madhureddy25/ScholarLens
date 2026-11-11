"""Feature extraction for Lens, with an on-disk cache keyed by record id.

Text is normalised exactly as for documents (masked, then rendered: citations →
"[1]", inline math → "x", LaTeX stripped), so a model trained on Bench
abstracts sees the same kind of input as one scoring a real manuscript.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from scholarlens.core.text import split_sentences
from scholarlens.features.stylometry import stylometric_features
from scholarlens.features.transitions import HashingEmbedder, transition_features
from scholarlens.masking import Masker

# Features that encode length, or depend on it mechanically, rather than style.
# Excluded so the classifier cannot shortcut on "AI abstracts are shorter/longer".
# Raw TTR and hapax ratio fall with text length; MATTR measures the same lexical
# diversity without that dependence and is kept. (Found in the CPU pilot, where
# TTR/hapax were the top features while generated abstracts were ~20% shorter.)
LENGTH_FEATURES = {"n_sentences", "n_words", "log_n_words", "lm_n_tokens", "ttr", "hapax_ratio"}


def normalize(text: str, fmt: str = "text") -> str:
    return Masker().mask(text, fmt=fmt).render()


class Featurizer:
    def __init__(self, *, lm=None, embedder=None, cache: str | Path | None = None,
                 drop: Iterable[str] = LENGTH_FEATURES) -> None:
        self.lm = lm
        self.embedder = embedder or HashingEmbedder()
        self.cache_path = Path(cache) if cache else None
        self.drop = set(drop)
        self._cache: dict[str, dict[str, float]] = {}
        if self.cache_path and self.cache_path.is_file():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._cache[row["key"]] = row["features"]

    @property
    def signature(self) -> str:
        lm = getattr(self.lm, "model_name", None) or "none"
        emb = getattr(self.embedder, "name", None) or type(self.embedder).__name__
        return f"lm={lm}|emb={emb}"

    def features(self, text: str) -> dict[str, float]:
        sents = split_sentences(text)
        f = stylometric_features(text, sents)
        f.update({f"tr_{k}": v for k, v in transition_features(sents, self.embedder).items()})
        if self.lm is not None:
            f.update(self.lm.score(text).to_dict())
        return f

    def _key(self, rid: str) -> str:
        return f"{self.signature}|{rid}"

    def featurize(self, items: list[tuple[str, str]], *, on_progress=None) -> tuple[np.ndarray, list[str]]:
        """``items`` = [(id, text)]. Returns (X, feature_names)."""
        rows: list[dict[str, float]] = []
        new: list[dict] = []
        for i, (rid, text) in enumerate(items, 1):
            key = self._key(rid)
            if key not in self._cache:
                self._cache[key] = self.features(normalize(text))
                new.append({"key": key, "features": self._cache[key]})
            rows.append(self._cache[key])
            if on_progress and (i % 50 == 0 or i == len(items)):
                on_progress(i, len(items))
        if new and self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as f:
                for row in new:
                    f.write(json.dumps(row) + "\n")
        names = sorted({k for r in rows for k in r} - self.drop)
        X = np.array([[r.get(n, np.nan) for n in names] for r in rows], dtype=float)
        return X, names
