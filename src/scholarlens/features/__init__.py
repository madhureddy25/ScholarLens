"""Statistical and structural signals for academic prose."""

from __future__ import annotations

from scholarlens.core.text import split_sentences
from scholarlens.features.stylometry import burstiness, stylometric_features
from scholarlens.features.transitions import (
    HashingEmbedder,
    transition_deviation,
    transition_features,
    transition_matrix,
)

__all__ = ["extract_features", "burstiness", "stylometric_features", "transition_features",
           "transition_matrix", "transition_deviation", "HashingEmbedder"]


def extract_features(text: str, *, embedder=None, lm=None) -> dict[str, float]:
    """All signals for one prose segment. ``lm`` is an optional ``CausalLMScorer``;
    ``embedder`` defaults to the dependency-free ``HashingEmbedder``."""
    sentences = split_sentences(text)
    feats = stylometric_features(text, sentences)
    feats.update({f"tr_{k}": v for k, v in transition_features(sentences, embedder or HashingEmbedder()).items()})
    if lm is not None:
        feats.update(lm.score(text).to_dict())
    return feats
