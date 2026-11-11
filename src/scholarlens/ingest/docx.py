"""DOCX → Document parser.

Beyond paragraph text, this records *protected spans* inside each paragraph:
runs with character formatting (italic species names, superscripts, sub-scripts),
hyperlinks, and reference-manager fields (Zotero / Mendeley / EndNote). The
masking engine turns these into sentinels so an editor can never strip the
formatting or break a live citation field.
"""

from __future__ import annotations

from pathlib import Path

import docx  # python-docx
from docx.table import Table
from docx.text.paragraph import Paragraph
from lxml import etree

from scholarlens.core.document import Document, NodeFactory, NodeType

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"

_FORMAT_TAGS = {"i": "italic", "b": "bold", "u": "underline", "smallCaps": "smallcaps",
                "caps": "caps", "strike": "strike"}
_BIB_HEADINGS = {"references", "bibliography", "works cited", "literature cited"}
_CITATION_FIELD_MARKERS = ("ZOTERO_ITEM", "ADDIN EN.CITE", "ADDIN CSL_CITATION", "MENDELEY", "CITATION ")


def _flag_on(el) -> bool:
    val = el.get(W + "val")
    return val is None or val.lower() not in ("0", "false", "none")


def _run_format(run) -> str | None:
    rpr = run.find(W + "rPr")
    if rpr is None:
        return None
    va = rpr.find(W + "vertAlign")
    if va is not None and va.get(W + "val") in ("superscript", "subscript"):
        return va.get(W + "val")
    for tag, name in _FORMAT_TAGS.items():
        el = rpr.find(W + tag)
        if el is not None and _flag_on(el):
            return name
    return None


def _run_text(run) -> str:
    parts = []
    for child in run:
        tag = etree.QName(child).localname
        if tag == "t":
            parts.append(child.text or "")
        elif tag == "tab":
            parts.append("\t")
        elif tag in ("br", "cr"):
            parts.append("\n")
        elif tag == "noBreakHyphen":
            parts.append("-")
    return "".join(parts)


def paragraph_content(p) -> tuple[str, list[dict], bool]:
    """Return (visible text, protected spans, contains_omml) for a ``w:p`` element."""
    text: list[str] = []
    spans: list[dict] = []
    pos = 0
    field_depth = 0
    field_is_citation = False
    has_math = p.find(".//" + M + "oMath") is not None

    for el in p.iter(W + "r"):
        # skip deleted text in tracked changes
        if any(etree.QName(a).localname == "del" for a in el.iterancestors()):
            continue
        fld = el.find(W + "fldChar")
        if fld is not None:
            kind = fld.get(W + "fldCharType")
            if kind == "begin":
                field_depth += 1
            elif kind == "end":
                field_depth = max(0, field_depth - 1)
                if field_depth == 0:
                    field_is_citation = False
            continue
        instr = el.find(W + "instrText")
        if instr is not None:
            if any(mk in (instr.text or "") for mk in _CITATION_FIELD_MARKERS):
                field_is_citation = True
            continue
        t = _run_text(el)
        if not t:
            continue
        reason = None
        ancestors = {etree.QName(a).localname for a in el.iterancestors()}
        if field_depth > 0 or "fldSimple" in ancestors:
            reason = "citation-field" if field_is_citation else "field"
        elif "hyperlink" in ancestors:
            reason = "hyperlink"
        else:
            reason = _run_format(el)
        if reason:
            if spans and spans[-1]["end"] == pos and spans[-1]["reason"] == reason:
                spans[-1]["end"] = pos + len(t)
            else:
                spans.append({"start": pos, "end": pos + len(t), "reason": reason})
        text.append(t)
        pos += len(t)
    return "".join(text), spans, has_math


def parse_docx(path: str | Path) -> Document:
    d = docx.Document(str(path))
    f = NodeFactory()
    nodes = []
    section: str | None = None
    in_bib = False
    title = None

    for block_index, child in enumerate(d.element.body.iterchildren()):
        tag = etree.QName(child).localname
        if tag == "tbl":
            cells = [" | ".join(c.text for c in row.cells) for row in Table(child, d).rows]
            nodes.append(f.make(NodeType.TABLE, "\n".join(cells), section=section, block_index=block_index))
            continue
        if tag != "p":
            continue
        para = Paragraph(child, d)
        style = (para.style.name if para.style is not None else "") or ""
        text, spans, has_math = paragraph_content(child)
        if not text.strip() and not has_math:
            continue
        meta = {"block_index": block_index, "style": style, "protected": spans}
        s = style.lower()
        if s == "title":
            type_ = NodeType.TITLE
            title = title or text.strip()
        elif s.startswith("heading"):
            type_ = NodeType.HEADING
            digits = "".join(ch for ch in s if ch.isdigit())
            meta["level"] = int(digits) if digits else 1
            section = text.strip()
            in_bib = section.lower().rstrip(":") in _BIB_HEADINGS
        elif s == "caption":
            type_ = NodeType.CAPTION
        elif in_bib or s in ("bibliography", "endnote text"):
            type_ = NodeType.BIBLIOGRAPHY
        elif has_math and len(text.strip()) < 3:
            type_ = NodeType.EQUATION
        elif s.startswith("list") or child.find(f"{W}pPr/{W}numPr") is not None:
            type_ = NodeType.LIST_ITEM
        else:
            type_ = NodeType.PARAGRAPH
        if has_math and type_ != NodeType.EQUATION:
            meta["locked"] = "inline-equation"  # editing would drop the OMML objects
        nodes.append(f.make(type_, text, section=section, **meta))

    return Document("docx", nodes, source=None, meta={"title": title, "path": str(path)})
