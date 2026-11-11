"""Detector evaluation metrics (scores: higher = more likely AI).

* AUROC and TPR at a fixed low FPR (1%) — the operating point that matters when
  a false accusation is costly.
* Robustness: the same metrics for each attack family against clean human text.
* Fairness: the false-positive rate of every *human* family (clean, Quill-edited,
  grammar-checked, per domain) at the threshold calibrated on clean human text.
* Calibration: expected calibration error of probability outputs.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from scholarlens.bench.schema import LABEL_AI, LABEL_HUMAN, Record


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann–Whitney U formulation with average ranks for ties."""
    y, s = np.asarray(y, dtype=int), np.asarray(s, dtype=float)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def threshold_at_fpr(neg_scores: np.ndarray, fpr: float = 0.01) -> float:
    """Smallest threshold whose FPR on these negatives is ≤ ``fpr`` (predict AI if score > t)."""
    neg = np.sort(np.asarray(neg_scores, dtype=float))
    if len(neg) == 0:
        return float("inf")
    k = int(np.floor(fpr * len(neg)))  # number of negatives allowed above the threshold
    return float(neg[len(neg) - k - 1]) if k < len(neg) else float(neg[0]) - 1e-12


def rate_above(scores: np.ndarray, threshold: float) -> float:
    scores = np.asarray(scores, dtype=float)
    return float((scores > threshold).mean()) if len(scores) else float("nan")


def tpr_at_fpr(y: np.ndarray, s: np.ndarray, fpr: float = 0.01) -> float:
    y, s = np.asarray(y, dtype=int), np.asarray(s, dtype=float)
    return rate_above(s[y == 1], threshold_at_fpr(s[y == 0], fpr))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y, p = np.asarray(y, dtype=float), np.clip(np.asarray(p, dtype=float), 0, 1)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def bootstrap_ci(metric, y, s, *, n: int = 500, alpha: float = 0.05, seed: int = 13) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            vals.append(metric(y[i], s[i]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def evaluate(records: list[Record], scores: dict[str, float], *, fpr: float = 0.01, ci: bool = True) -> dict:
    """Full robustness + fairness report for one detector on one split."""
    fam: dict[str, list[float]] = defaultdict(list)
    dom_human: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if r.id not in scores:
            continue
        fam[r.family].append(scores[r.id])
        if r.label == LABEL_HUMAN and r.transform == "none":
            dom_human[r.domain or "unknown"].append(scores[r.id])
    human = np.asarray(fam.get(LABEL_HUMAN, []))
    if len(human) == 0:
        raise ValueError("no clean human records with scores")
    thr = threshold_at_fpr(human, fpr)
    report: dict = {"threshold": thr, "target_fpr": fpr, "n_human": len(human), "attacks": {}, "human_fpr": {}}
    for name, vals in sorted(fam.items()):
        vals = np.asarray(vals)
        if name.startswith(LABEL_AI):
            y = np.r_[np.zeros(len(human)), np.ones(len(vals))]
            s = np.r_[human, vals]
            row = {"n": len(vals), "auroc": auroc(y, s), f"tpr@{fpr:g}fpr": rate_above(vals, thr)}
            if ci:
                row["auroc_ci95"] = bootstrap_ci(auroc, y, s)
            report["attacks"][name] = row
        else:
            report["human_fpr"][name] = {"n": len(vals), "fpr": rate_above(vals, thr)}
    report["human_fpr_by_domain"] = {d: {"n": len(v), "fpr": rate_above(np.asarray(v), thr)}
                                     for d, v in sorted(dom_human.items())}
    clean_ai = report["attacks"].get(LABEL_AI)
    if clean_ai:
        for row in report["attacks"].values():
            row["auroc_drop"] = clean_ai["auroc"] - row["auroc"]
    return report


__all__ = ["auroc", "threshold_at_fpr", "tpr_at_fpr", "rate_above", "ece", "bootstrap_ci", "evaluate",
           "LABEL_AI", "LABEL_HUMAN"]
