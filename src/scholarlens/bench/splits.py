"""Paper-level train/val/test splits.

The split is a deterministic function of ``paper_id``, so every text derived
from a paper (its AI counterpart, paraphrases, edits) is in the same split and
no paper's style or topic can leak from train into test.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable

from scholarlens.bench.schema import Record

DEFAULT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def split_of(paper_id: str, ratios: dict[str, float] = DEFAULT_RATIOS, seed: int = 13) -> str:
    h = int(hashlib.sha256(f"{seed}:{paper_id}".encode()).hexdigest()[:12], 16) / 16**12
    total, acc = sum(ratios.values()), 0.0
    for name, r in ratios.items():
        acc += r / total
        if h < acc:
            return name
    return list(ratios)[-1]


def assign_splits(records: Iterable[Record], ratios: dict[str, float] = DEFAULT_RATIOS, seed: int = 13) -> list[Record]:
    out = []
    for r in records:
        r.split = split_of(r.paper_id, ratios, seed)
        out.append(r)
    return out


def leakage(records: Iterable[Record]) -> dict[str, set[str]]:
    """paper_ids that appear in more than one split (must be empty)."""
    seen: dict[str, set[str]] = defaultdict(set)
    for r in records:
        seen[r.paper_id].add(r.split or "?")
    return {p: s for p, s in seen.items() if len(s) > 1}
