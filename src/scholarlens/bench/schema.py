"""Benchmark records and JSONL storage.

Every text in the benchmark is a ``Record``. Derived texts (AI generations,
paraphrase attacks, Quill edits) point at their ``parent_id`` and inherit the
parent's ``paper_id``, so a whole family of texts always lands in the same split.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

LABEL_HUMAN, LABEL_AI = "human", "ai"


@dataclass
class Record:
    id: str
    paper_id: str
    text: str
    label: str  # "human" | "ai": who produced the underlying content
    source: str  # corpus, e.g. "arxiv"
    domain: str | None = None
    year: int | None = None
    title: str | None = None
    generator: str | None = None  # model that wrote the AI text
    transform: str = "none"  # e.g. "llm_paraphrase", "dipper_L60_O60", "quill_proofread"
    transform_model: str | None = None
    parent_id: str | None = None
    split: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def family(self) -> str:
        """Short tag used in reports: human, ai, human+quill_proofread, ai+dipper_L60_O60 …"""
        return self.label if self.transform == "none" else f"{self.label}+{self.transform}"

    def derive(self, text: str, *, transform: str, transform_model: str | None = None, **meta) -> Record:
        rid = stable_id(self.id, transform, transform_model or "")
        return Record(id=rid, paper_id=self.paper_id, text=text, label=self.label, source=self.source,
                      domain=self.domain, year=self.year, title=self.title, generator=self.generator,
                      transform=transform, transform_model=transform_model, parent_id=self.id,
                      split=self.split, meta=meta)


def stable_id(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def write_jsonl(records: Iterable[Record], path: str | Path, *, append: bool = False) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a" if append else "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[Record]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield Record(**json.loads(line))
