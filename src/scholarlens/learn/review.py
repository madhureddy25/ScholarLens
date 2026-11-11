"""Author review of a Quill session: accept or reject each edit.

Produces the final document containing only the edits the author accepted, an
updated disclosure manifest, and (unless disabled) local feedback entries that
can later improve Quill.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from scholarlens.edit.disclosure import build_manifest
from scholarlens.edit.engine import EditSession, NodeEdit
from scholarlens.edit.session_io import load_session
from scholarlens.learn import feedback

_EXT = {"latex": ".tex", "markdown": ".md", "text": ".txt"}


@dataclass
class ReviewResult:
    final: EditSession
    accepted: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    feedback_entries: list[dict] = field(default_factory=list)
    files: dict[str, Path] = field(default_factory=dict)


def review_session(session_path: str | Path, decide: Callable[[NodeEdit], str], *, out_dir: str | Path | None = None,
                   input_path: str | Path | None = None, record_feedback: bool = True) -> ReviewResult:
    session, data = load_session(session_path, input_path=input_path)
    res = ReviewResult(final=session)
    keep: list[NodeEdit] = []
    for e in session.edits:
        d = decide(e)
        if d not in ("accept", "reject", "skip"):
            raise ValueError(f"decision must be accept/reject/skip, got {d!r}")
        {"accept": res.accepted, "reject": res.rejected, "skip": res.skipped}[d].append(e.node_id)
        if d == "accept":
            keep.append(e)
        if d in ("accept", "reject"):
            res.feedback_entries.append(feedback.make_entry(
                masked_input=e.mask.masked, masked_output=e.edited_masked, decision=d, mode=session.config.mode,
                fmt=session.document.source_format, model=session.model, section=e.section, similarity=e.similarity))
    res.final = EditSession(session.document, keep, session.model, session.config, session.similarity_model,
                            session.started_at, session.finished_at, session.glossary)
    if record_feedback and res.feedback_entries:
        feedback.append(res.feedback_entries)

    src = Path(input_path or data["input"])
    out = Path(out_dir or Path(session_path).parent)
    out.mkdir(parents=True, exist_ok=True)
    doc = session.document
    if doc.source_format == "docx":
        from scholarlens.edit.docx_writer import write_docx

        final_path = out / f"{src.stem}.final.docx"
        write_docx(res.final, src, final_path, tracked=False)
    else:
        final_path = out / f"{src.stem}.final{_EXT.get(doc.source_format, '.md')}"
        final_path.write_text(res.final.output_text(), encoding="utf-8")
    manifest = build_manifest(res.final, input_path=src, output_path=final_path)
    manifest["author_review"] = {"accepted": res.accepted, "rejected": res.rejected, "skipped": res.skipped}
    manifest_path = out / f"{src.stem}.final.disclosure.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    res.files = {"final": final_path, "disclosure": manifest_path}
    return res
