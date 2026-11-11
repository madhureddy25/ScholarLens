"""Disclosure manifest: a machine-readable record of what Quill changed and how.

Many journals ask authors to disclose AI assistance in preparing a manuscript.
The manifest records the model, settings, and every edited span, and
``disclosure_statement`` drafts a sentence the authors can adapt.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

from scholarlens import __version__
from scholarlens.edit.engine import EditSession, Status


def _sha256(path: str | Path | None) -> str | None:
    if path is None or not Path(path).is_file():
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_manifest(session: EditSession, *, input_path: str | Path | None = None,
                   output_path: str | Path | None = None) -> dict:
    doc = session.document
    accepted = [e for e in session.edits if e.status == Status.ACCEPTED]
    words_total = sum(e.stats.get("words_original", 0) for e in accepted)
    words_changed = sum(e.stats.get("words_removed", 0) + e.stats.get("words_added", 0) for e in accepted)
    nodes = []
    for e in session.edits:
        node = doc.get(e.node_id)
        nodes.append({
            "node_id": e.node_id,
            "section": e.section,
            "type": node.type.value,
            "span": [node.start, node.end] if node.has_span else None,
            "status": e.status.value,
            "reason": e.reason,
            "attempts": len(e.attempts),
            "similarity": e.similarity,
            "served_by": e.served_by,
            "change": e.stats or None,
            "warnings": e.warnings,
        })
    return {
        "tool": {"name": "ScholarLens Quill", "version": __version__},
        "purpose": "language editing of the authors' own text",
        "started_at": session.started_at,
        "finished_at": session.finished_at,
        "input": {"path": str(input_path) if input_path else doc.meta.get("path"),
                  "sha256": _sha256(input_path or doc.meta.get("path")), "format": doc.source_format,
                  "title": doc.meta.get("title")},
        "output": {"path": str(output_path) if output_path else None, "sha256": _sha256(output_path)},
        "model": session.model,
        "settings": {**asdict(session.config), "similarity_model": session.similarity_model},
        "guarantees": {
            "non_prose_unchanged": True,
            "protected_items_verified": ["citations", "cross-references", "math", "numbers", "URLs",
                                         "LaTeX commands", "domain terms"],
        },
        "summary": {**session.counts(), "prose_nodes": len(session.edits),
                    "words_in_edited_paragraphs": words_total, "words_changed": words_changed},
        "nodes": nodes,
        "statement_draft": disclosure_statement(session),
    }


def disclosure_statement(session: EditSession) -> str:
    n = sum(1 for e in session.edits if e.status == Status.ACCEPTED)
    total = len(session.edits)
    model = f"{session.model.get('provider')}:{session.model.get('model')}"
    if session.model.get("adapter"):
        model += f" with the fine-tuned adapter '{Path(str(session.model['adapter'])).name}'"
    served = {e.served_by for e in session.edits if e.status == Status.ACCEPTED and e.served_by}
    if served - {model}:
        model += f" (some requests served by {', '.join(sorted(served - {model}))})"
    date = session.finished_at[:10]
    return (f"The authors used ScholarLens Quill ({session.config.mode} mode) with the language model {model} "
            f"on {date} to suggest language edits to {n} of {total} prose paragraphs. Citations, equations, "
            f"numerical results and technical terms were protected from modification and verified "
            f"automatically. All suggested edits were reviewed by the authors, who take full responsibility "
            f"for the content.")
