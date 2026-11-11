"""Document-level analysis: prose extraction → masking → segment features.

This is the shared entry point for the detector (Lens) and for the editor's
before/after reports (Quill).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from scholarlens.core.document import Document
from scholarlens.core.text import split_sentences, word_windows, words
from scholarlens.features import extract_features
from scholarlens.masking import Masker, MaskResult


@dataclass
class ProseUnit:
    node_id: str
    section: str | None
    mask: MaskResult
    plain: str
    """Rendered for scoring: citations → "[1]", inline math → "x", LaTeX stripped."""
    locked: str | None = None


@dataclass
class Segment:
    index: int
    node_ids: list[str]
    text: str
    n_words: int
    features: dict[str, float] = field(default_factory=dict)


def prose_units(doc: Document, masker: Masker | None = None) -> list[ProseUnit]:
    masker = masker or Masker()
    units = []
    for node in doc.prose():
        m = masker.mask_node(node, doc.source_format)
        units.append(ProseUnit(node.id, node.section, m, m.render(), node.meta.get("locked")))
    return units


def segments(units: list[ProseUnit], size: int = 250, stride: int | None = None) -> list[Segment]:
    """Sentence-aligned ~``size``-word overlapping windows across the document's prose."""
    sents: list[str] = []
    owner: list[str] = []
    for u in units:
        for s in split_sentences(u.plain):
            sents.append(s)
            owner.append(u.node_id)
    out = []
    for i, (a, b) in enumerate(word_windows(sents, size, stride)):
        text = " ".join(sents[a:b])
        out.append(Segment(i, sorted(set(owner[a:b]), key=owner.index), text, len(words(text))))
    return out


def analyze(doc: Document, *, masker: Masker | None = None, embedder=None, lm=None,
            segment_size: int = 250) -> dict:
    units = prose_units(doc, masker)
    segs = segments(units, segment_size)
    for s in segs:
        s.features = extract_features(s.text, embedder=embedder, lm=lm)
    full = "\n\n".join(u.plain for u in units)
    mask_counts: dict[str, int] = {}
    for u in units:
        for k, v in u.mask.counts().items():
            mask_counts[k] = mask_counts.get(k, 0) + v
    return {
        "title": doc.meta.get("title"),
        "format": doc.source_format,
        "structure": doc.stats(),
        "protected_spans": mask_counts,
        "n_prose_units": len(units),
        "document_features": extract_features(full, embedder=embedder, lm=lm) if full.strip() else {},
        "segments": [{"index": s.index, "node_ids": s.node_ids, "n_words": s.n_words,
                      "features": s.features} for s in segs],
    }
