"""Sentence-relation graph classifier (pure PyTorch, CPU-friendly).

Each document becomes a graph: nodes are sentences (embedding + relative position
+ length), edges link consecutive sentences and each sentence's k most similar
sentences, weighted by cosine similarity. A two-layer GCN with mean+max readout
classifies the graph. This is the document-level structural model targeting the
Relational Over-Regularization signal: machine text tends to have unusually
uniform sentence-to-sentence transitions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scholarlens.bench.metrics import auroc
from scholarlens.bench.schema import LABEL_AI, Record
from scholarlens.core.text import split_sentences, words
from scholarlens.detect.featurize import normalize
from scholarlens.features.transitions import transition_matrix


def build_graph(sentences: list[str], emb: np.ndarray, knn: int = 3) -> tuple[np.ndarray, np.ndarray]:
    m = len(sentences)
    pos = np.linspace(0, 1, m)[:, None] if m > 1 else np.zeros((1, 1))
    lens = np.array([[min(len(words(s)), 80) / 80] for s in sentences])
    X = np.hstack([emb, pos, lens]).astype(np.float32)
    T = transition_matrix(emb) if m else np.zeros((0, 0))
    A = np.zeros((m, m), dtype=np.float32)
    for i in range(m - 1):
        A[i, i + 1] = A[i + 1, i] = 1.0
    for i in range(m):
        for j in [j for j in np.argsort(-T[i]) if j != i][:knn]:
            w = max(float(T[i, j]), 0.0)
            A[i, j] = max(A[i, j], w)
            A[j, i] = max(A[j, i], w)
    return X, A


def _model(d_in: int, hidden: int, dropout: float):
    import torch
    from torch import nn

    class GCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1, self.l2 = nn.Linear(d_in, hidden), nn.Linear(hidden, hidden)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

        def forward(self, X, A):
            A = A + torch.eye(A.shape[0])
            d = A.sum(1).rsqrt()
            An = d[:, None] * A * d[None, :]
            H = self.drop(torch.relu(An @ self.l1(X)))
            H = torch.relu(An @ self.l2(H))
            return self.head(torch.cat([H.mean(0), H.max(0).values])).squeeze(-1)

    return GCN()


class GNNDetector:
    name = "lens-gnn"

    def __init__(self, embedder, *, knn: int = 3, hidden: int = 128, dropout: float = 0.2, epochs: int = 40,
                 lr: float = 1e-3, batch: int = 16, patience: int = 8, seed: int = 13) -> None:
        self.embedder = embedder
        self.cfg = dict(knn=knn, hidden=hidden, dropout=dropout, epochs=epochs, lr=lr, batch=batch,
                        patience=patience, seed=seed)
        self.net = None
        self._cache: dict[str, tuple] = {}

    def _graph(self, rid: str, text: str):
        import torch

        if rid not in self._cache:
            sents = split_sentences(normalize(text)) or [text]
            X, A = build_graph(sents, np.asarray(self.embedder.encode(sents), dtype=np.float32), self.cfg["knn"])
            self._cache[rid] = (torch.from_numpy(X), torch.from_numpy(A))
        return self._cache[rid]

    def _logits(self, graphs) -> np.ndarray:
        import torch

        self.net.eval()
        with torch.no_grad():
            return np.array([float(self.net(X, A)) for X, A in graphs])

    def fit(self, train: list[Record], val: list[Record] | None = None, *, log=None) -> GNNDetector:
        import torch

        torch.manual_seed(self.cfg["seed"])
        rng = np.random.default_rng(self.cfg["seed"])
        g_tr = [self._graph(r.id, r.text) for r in train]
        y_tr = torch.tensor([1.0 if r.label == LABEL_AI else 0.0 for r in train])
        g_va = [self._graph(r.id, r.text) for r in val] if val else []
        y_va = np.array([1 if r.label == LABEL_AI else 0 for r in val]) if val else None
        self.net = _model(g_tr[0][0].shape[1], self.cfg["hidden"], self.cfg["dropout"])
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.cfg["lr"], weight_decay=1e-4)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        best, best_state, bad = -1.0, None, 0
        for epoch in range(self.cfg["epochs"]):
            self.net.train()
            order = rng.permutation(len(g_tr))
            for s in range(0, len(order), self.cfg["batch"]):
                idx = order[s:s + self.cfg["batch"]]
                out = torch.stack([self.net(*g_tr[i]) for i in idx])
                loss = loss_fn(out, y_tr[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            if g_va:
                score = auroc(y_va, self._logits(g_va))
                if log:
                    log(f"  gnn epoch {epoch + 1}: val AUROC {score:.4f}")
                if score > best:
                    best, bad = score, 0
                    best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.cfg["patience"]:
                        break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.val_auroc = best
        return self

    def score_records(self, records: list[Record]) -> np.ndarray:
        return self._logits([self._graph(r.id, r.text) for r in records])

    def score_texts(self, texts: list[str]) -> np.ndarray:
        return self._logits([self._graph(f"adhoc:{hash(t)}", t) for t in texts])

    def save(self, path: str | Path) -> None:
        import torch

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path / "gnn.pt")
        d_in = self.net.l1.in_features
        (path / "gnn.json").write_text(json.dumps({**self.cfg, "d_in": d_in}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, embedder) -> GNNDetector:
        import torch

        path = Path(path)
        cfg = json.loads((path / "gnn.json").read_text(encoding="utf-8"))
        d_in = cfg.pop("d_in")
        det = cls(embedder, **cfg)
        det.net = _model(d_in, cfg["hidden"], cfg["dropout"])
        det.net.load_state_dict(torch.load(path / "gnn.pt"))
        return det
