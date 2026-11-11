"""Phase 1 — document ingestion. ``load(path)`` dispatches on file extension."""

from __future__ import annotations

from pathlib import Path

from scholarlens.core.document import Document
from scholarlens.ingest.latex import parse_latex
from scholarlens.ingest.markdown import parse_markdown, parse_text

__all__ = ["load", "parse_string", "parse_latex", "parse_markdown", "parse_text"]


def parse_string(text: str, fmt: str = "text") -> Document:
    fmt = fmt.lower()
    if fmt in ("latex", "tex"):
        return parse_latex(text)
    if fmt in ("markdown", "md"):
        return parse_markdown(text)
    if fmt in ("text", "txt", "plain"):
        return parse_text(text)
    if fmt in ("tei", "xml"):
        from scholarlens.ingest.tei import parse_tei

        return parse_tei(text)
    raise ValueError(f"unsupported string format: {fmt}")


def load(path: str | Path) -> Document:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".docx":
        from scholarlens.ingest.docx import parse_docx

        return parse_docx(path)
    if ext == ".pdf":
        from scholarlens.ingest.tei import parse_pdf

        return parse_pdf(path)
    text = path.read_text(encoding="utf-8")
    fmt = {".tex": "latex", ".md": "markdown", ".markdown": "markdown", ".txt": "text",
           ".xml": "tei", ".tei": "tei"}.get(ext)
    if fmt is None:
        raise ValueError(f"unsupported file type: {ext}")
    doc = parse_string(text, fmt)
    doc.meta.setdefault("path", str(path))
    return doc
