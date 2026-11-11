"""Save a Quill session so the author can review it later (``scholarlens review``)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from scholarlens import __version__
from scholarlens.edit.engine import EditConfig, EditSession, NodeEdit, Status
from scholarlens.ingest import load
from scholarlens.masking import MaskConfig, Masker


def save_session(session: EditSession, path: str | Path, *, input_path: str | Path | None) -> Path:
    path = Path(path)
    edits = []
    for e in session.edits:
        if e.status != Status.ACCEPTED or e.mask is None or e.edited_masked is None:
            continue
        edits.append({"node_id": e.node_id, "section": e.section, "original": e.original, "edited": e.edited,
                      "masked": e.mask.masked, "edited_masked": e.edited_masked, "similarity": e.similarity,
                      "served_by": e.served_by})
    data = {"version": 1, "tool_version": __version__,
            "input": str(Path(input_path).resolve()) if input_path else session.document.meta.get("path"),
            "format": session.document.source_format, "model": session.model, "config": asdict(session.config),
            "similarity_model": session.similarity_model, "glossary": list(session.glossary),
            "started_at": session.started_at, "finished_at": session.finished_at, "edits": edits}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_session(path: str | Path, *, input_path: str | Path | None = None) -> tuple[EditSession, dict]:
    """Rebuild an EditSession (accepted edits only) against the original input file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    src = Path(input_path or data["input"])
    doc = load(src)
    masker = Masker(MaskConfig(glossary=tuple(data.get("glossary", []))))
    edits = []
    for item in data["edits"]:
        node = doc.get(item["node_id"])
        if node.text != item["original"]:
            raise ValueError(f"{src} changed since the Quill session (node {item['node_id']} differs)")
        mask = masker.mask_node(node, doc.source_format)
        if mask.masked != item["masked"]:
            raise ValueError(f"masking of {item['node_id']} differs from the session; was the glossary changed?")
        edits.append(NodeEdit(item["node_id"], item["section"], Status.ACCEPTED, item["original"],
                              edited=item["edited"], similarity=item.get("similarity"), served_by=item.get("served_by"),
                              mask=mask, edited_masked=item["edited_masked"]))
    cfg = EditConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in data["config"].items()})
    session = EditSession(doc, edits, data["model"], cfg, data.get("similarity_model"), data["started_at"],
                          data["finished_at"], tuple(data.get("glossary", [])))
    return session, data
