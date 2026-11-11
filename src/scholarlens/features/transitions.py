"""Sentence-transition structure (Relational Over-Regularization signals).

Given sentence embeddings s_1..s_M, the transition matrix T(D) holds pairwise
cosine similarities and the local transition deviation is

    δ_ij = cos(s_i, s_j) − 1/(M−1) · Σ_{k≠i} cos(s_i, s_k)

i.e. how much more (or less) sentence j relates to sentence i than the average
sentence does. Machine-generated prose tends to show suppressed variance in T
with templated similarity spikes; these features expose that, and
``sentence_graph`` builds the graph used by the GNN detector.
"""

from __future__ import annotations

import re
import zlib
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def encode(self, sentences: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Dependency-free bag-of-n-grams embedder (feature hashing).

    Deterministic and fast; useful for tests and as a lexical-overlap baseline.
    Use ``SentenceTransformerEmbedder`` for semantic similarity.
    """

    def __init__(self, dim: int = 2048) -> None:
        self.dim = dim

    def encode(self, sentences: list[str]) -> np.ndarray:
        out = np.zeros((len(sentences), self.dim), dtype=np.float32)
        for i, s in enumerate(sentences):
            toks = re.findall(r"\w+", s.lower())
            grams = toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]
            for g in grams:
                h = zlib.crc32(g.encode("utf-8"))
                out[i, h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        return out


class SentenceTransformerEmbedder:
    """Semantic embedder. Good defaults: ``sentence-transformers/all-MiniLM-L6-v2``
    (fast, general) or ``sentence-transformers/allenai-specter`` (scientific)."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        from scholarlens._hub import default_device

        self.model = SentenceTransformer(model, device=device or default_device())
        self.name = model

    def encode(self, sentences: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(sentences, batch_size=32, show_progress_bar=False))


def transition_matrix(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    unit = emb / np.where(norms == 0, 1.0, norms)
    return unit @ unit.T


def transition_deviation(T: np.ndarray) -> np.ndarray:
    """δ_ij = T_ij − mean_{k≠i} T_ik  (diagonal set to 0)."""
    M = T.shape[0]
    if M < 2:
        return np.zeros_like(T)
    row_mean = (T.sum(axis=1) - np.diag(T)) / (M - 1)
    delta = T - row_mean[:, None]
    np.fill_diagonal(delta, 0.0)
    return delta


_EMPTY = {
    "adj_sim_mean": 0.0, "adj_sim_std": 0.0, "adj_sim_min": 0.0, "adj_sim_max": 0.0,
    "offdiag_sim_mean": 0.0, "offdiag_sim_var": 0.0, "delta_abs_mean": 0.0,
    "delta_adj_mean": 0.0, "delta_adj_std": 0.0, "adj_spike_rate": 0.0, "local_global_ratio": 0.0,
}


def transition_features(sentences: list[str], embedder: Embedder) -> dict[str, float]:
    M = len(sentences)
    if M < 3:
        return dict(_EMPTY)
    T = transition_matrix(embedder.encode(sentences))
    off = T[~np.eye(M, dtype=bool)]
    adj = np.diag(T, k=1)
    delta = transition_deviation(T)
    d_adj = np.diag(delta, k=1)
    spike_thr = off.mean() + 2 * off.std()
    return {
        "adj_sim_mean": float(adj.mean()),
        "adj_sim_std": float(adj.std()),
        "adj_sim_min": float(adj.min()),
        "adj_sim_max": float(adj.max()),
        "offdiag_sim_mean": float(off.mean()),
        "offdiag_sim_var": float(off.var()),  # Var(T(D))
        "delta_abs_mean": float(np.abs(delta[~np.eye(M, dtype=bool)]).mean()),
        "delta_adj_mean": float(d_adj.mean()),
        "delta_adj_std": float(d_adj.std()),
        "adj_spike_rate": float((adj > spike_thr).mean()),
        "local_global_ratio": float(adj.mean() / off.mean()) if off.mean() else 0.0,
    }


def sentence_graph(sentences: list[str], embedder: Embedder, knn: int = 3):
    """Sentence-relation graph: nodes = sentences (with embeddings), edges =
    sequential links plus k nearest semantic neighbours, weighted by δ_ij."""
    import networkx as nx

    emb = embedder.encode(sentences)
    T = transition_matrix(emb)
    delta = transition_deviation(T)
    g = nx.DiGraph()
    for i, s in enumerate(sentences):
        g.add_node(i, text=s, emb=emb[i])
    for i in range(len(sentences)):
        if i + 1 < len(sentences):
            g.add_edge(i, i + 1, kind="seq", sim=float(T[i, i + 1]), delta=float(delta[i, i + 1]))
        order = [j for j in np.argsort(-T[i]) if j != i][:knn]
        for j in order:
            if not g.has_edge(i, int(j)):
                g.add_edge(i, int(j), kind="sem", sim=float(T[i, j]), delta=float(delta[i, j]))
    return g
