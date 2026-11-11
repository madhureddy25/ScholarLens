"""Human-written text sources.

All sources yield ``Record(label="human")`` and enforce a publication cut-off
(default: before 2022-01-01) so the human baseline is not contaminated by
LLM-assisted writing.

* ``ArxivAbstracts`` — Hugging Face dataset ``gfissore/arxiv-abstracts-2021``
  (2021 snapshot; year taken from the arXiv identifier).
* ``HFDatasetSource`` — any Hugging Face dataset with configurable fields.
* ``JsonlSource`` — local JSONL, e.g. S2ORC or PubMed Central exports you
  downloaded with your own API access.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from scholarlens.bench.schema import LABEL_HUMAN, Record, stable_id

_NEW_ID = re.compile(r"^(\d{2})(\d{2})\.\d{4,5}")
_OLD_ID = re.compile(r"/(\d{2})(\d{2})\d{3}")
_ARXIV_GROUPS = {
    "cs": "cs", "math": "math", "stat": "stat", "eess": "eess", "econ": "econ", "q-bio": "q-bio",
    "q-fin": "q-fin", "astro-ph": "physics", "cond-mat": "physics", "gr-qc": "physics", "hep": "physics",
    "math-ph": "physics", "nlin": "physics", "nucl": "physics", "physics": "physics", "quant-ph": "physics",
}


def arxiv_year(arxiv_id: str) -> int | None:
    m = _NEW_ID.match(arxiv_id) or _OLD_ID.search(arxiv_id)
    if not m:
        return None
    yy = int(m.group(1))
    return 1900 + yy if yy >= 90 else 2000 + yy


def arxiv_domain(categories: str) -> str | None:
    try:
        cats = ast.literal_eval(categories) if categories.startswith("[") else [categories]
    except (ValueError, SyntaxError):
        cats = [categories]
    first = " ".join(cats).split()[0] if cats and " ".join(cats).split() else ""
    for prefix, group in _ARXIV_GROUPS.items():
        if first.startswith(prefix):
            return group
    return first.split(".")[0] or None


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


@dataclass
class Filters:
    before_year: int = 2022
    min_words: int = 80
    max_words: int = 400

    def ok(self, text: str, year: int | None) -> bool:
        if year is None or year >= self.before_year:
            return False
        n = len(text.split())
        return self.min_words <= n <= self.max_words


class HFDatasetSource:
    def __init__(self, dataset: str, *, split: str = "train", config: str | None = None, text_field: str = "text",
                 id_field: str = "id", title_field: str | None = None, year_field: str | None = None,
                 domain_field: str | None = None, source_name: str | None = None, shuffle_buffer: int = 10_000,
                 seed: int = 13) -> None:
        self.dataset, self.split, self.config = dataset, split, config
        self.text_field, self.id_field, self.title_field = text_field, id_field, title_field
        self.year_field, self.domain_field = year_field, domain_field
        self.source_name = source_name or dataset.split("/")[-1]
        self.shuffle_buffer, self.seed = shuffle_buffer, seed

    def _rows(self) -> Iterator[dict]:
        from datasets import load_dataset

        import scholarlens._hub  # noqa: F401

        ds = load_dataset(self.dataset, self.config, split=self.split, streaming=True)
        if self.shuffle_buffer:
            ds = ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)
        yield from ds

    def year_of(self, row: dict) -> int | None:
        v = row.get(self.year_field) if self.year_field else None
        try:
            return int(str(v)[:4]) if v not in (None, "", "None") else None
        except ValueError:
            return None

    def domain_of(self, row: dict) -> str | None:
        return str(row.get(self.domain_field)) if self.domain_field and row.get(self.domain_field) else None

    def records(self, filters: Filters) -> Iterator[Record]:
        for row in self._rows():
            text = clean_text(str(row.get(self.text_field, "")))
            year = self.year_of(row)
            if not filters.ok(text, year):
                continue
            pid = f"{self.source_name}:{row.get(self.id_field)}"
            title = clean_text(str(row.get(self.title_field, ""))) if self.title_field else None
            yield Record(id=stable_id(pid, "human"), paper_id=pid, text=text, label=LABEL_HUMAN,
                         source=self.source_name, domain=self.domain_of(row), year=year, title=title or None)


class ArxivAbstracts(HFDatasetSource):
    def __init__(self, **kw) -> None:
        super().__init__("gfissore/arxiv-abstracts-2021", text_field="abstract", id_field="id",
                         title_field="title", domain_field="categories", source_name="arxiv", **kw)

    def year_of(self, row: dict) -> int | None:
        return arxiv_year(str(row.get("id", "")))

    def domain_of(self, row: dict) -> str | None:
        return arxiv_domain(str(row.get("categories", "")))


class JsonlSource:
    """Local JSONL with configurable field names (S2ORC, PMC OA, institutional corpora)."""

    def __init__(self, path: str | Path, *, text_field: str = "text", id_field: str = "id",
                 title_field: str | None = "title", year_field: str | None = "year",
                 domain_field: str | None = None, source_name: str | None = None) -> None:
        self.path = Path(path)
        self.text_field, self.id_field, self.title_field = text_field, id_field, title_field
        self.year_field, self.domain_field = year_field, domain_field
        self.source_name = source_name or self.path.stem

    def records(self, filters: Filters) -> Iterator[Record]:
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = clean_text(str(row.get(self.text_field, "")))
                try:
                    year = int(str(row.get(self.year_field))[:4]) if self.year_field and row.get(self.year_field) else None
                except ValueError:
                    year = None
                if not filters.ok(text, year):
                    continue
                pid = f"{self.source_name}:{row.get(self.id_field)}"
                yield Record(id=stable_id(pid, "human"), paper_id=pid, text=text, label=LABEL_HUMAN,
                             source=self.source_name,
                             domain=str(row[self.domain_field]) if self.domain_field and row.get(self.domain_field) else None,
                             year=year, title=clean_text(str(row.get(self.title_field, ""))) if self.title_field else None)


def make_source(spec: dict):
    spec = dict(spec)
    kind = spec.pop("type")
    if kind == "arxiv":
        return ArxivAbstracts(**spec)
    if kind == "hf":
        return HFDatasetSource(spec.pop("dataset"), **spec)
    if kind == "jsonl":
        return JsonlSource(spec.pop("path"), **spec)
    raise ValueError(f"unknown source type {kind!r}")
