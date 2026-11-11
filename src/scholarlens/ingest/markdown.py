"""Markdown and plain-text → Document parser (line-based, span-preserving)."""

from __future__ import annotations

import re

from scholarlens.core.document import Document, NodeFactory, NodeType

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_LIST_RE = re.compile(r"^(\s*)([-*+]|\d{1,3}[.)])\s+")
_IMAGE_RE = re.compile(r"^\s*!\[[^\]]*\]\([^)]*\)\s*$")
_TABLE_RE = re.compile(r"^\s*\|")
_SECTION_NAMES = (
    "abstract", "introduction", "related work", "background", "preliminaries", "method",
    "methods", "methodology", "materials and methods", "approach", "experiments",
    "experimental setup", "evaluation", "results", "discussion", "results and discussion",
    "analysis", "conclusion", "conclusions", "limitations", "future work",
    "acknowledgments", "acknowledgements", "references", "bibliography", "works cited",
    "appendix", "supplementary material", "ethics statement",
)
_BIB_HEADINGS = {"references", "bibliography", "works cited", "literature cited"}
_PLAIN_HEADING_RE = re.compile(
    r"^(?P<num>(?:\d+(?:\.\d+)*\.?|[IVX]+\.)\s+)?(?P<title>[A-Z][A-Za-z ,&\-]{1,78})$"
)


def _plain_heading(line: str) -> str | None:
    """Recognise headings in plain text: "2.1 Methods", "REFERENCES", "Introduction"."""
    s = line.strip()
    m = _PLAIN_HEADING_RE.match(s)
    if not m:
        return None
    title = m.group("title").strip()
    if m.group("num") and len(title.split()) <= 8:
        return title
    if title.lower() in _SECTION_NAMES:
        return title
    return None


def parse_markdown(text: str, *, plain: bool = False) -> Document:
    """Parse Markdown (or, with ``plain=True``, plain text with heuristic headings)."""
    f = NodeFactory()
    nodes = []
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for ln in lines:
        offsets.append(offsets[-1] + len(ln))

    section: str | None = None
    in_bib = False
    para: list | None = None  # [type, start, end]

    def flush() -> None:
        nonlocal para
        if para is None:
            return
        type_, start, end = para
        para = None
        end = start + len(text[start:end].rstrip())
        if end > start:
            nodes.append(f.make(type_, text[start:end], start, end, section=section))

    def emit(type_: NodeType, start: int, end: int, **meta) -> None:
        end = start + len(text[start:end].rstrip())
        nodes.append(f.make(type_, text[start:end], start, end, section=section, **meta))

    i = 0
    if lines and lines[0].strip() == "---" and not plain:
        for j in range(1, len(lines)):
            if lines[j].strip() in ("---", "..."):
                emit(NodeType.METADATA, 0, offsets[j + 1], role="front-matter")
                i = j + 1
                break

    while i < len(lines):
        line = lines[i]
        start = offsets[i]
        stripped = line.strip()

        if not stripped:
            flush()
            i += 1
            continue

        if not plain and _FENCE_RE.match(line):
            flush()
            fence = _FENCE_RE.match(line).group(1)
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith(fence):
                j += 1
            end_line = min(j, len(lines) - 1)
            emit(NodeType.CODE, start, offsets[end_line + 1])
            i = end_line + 1
            continue

        if stripped.startswith("$$"):
            flush()
            j = i
            if not (len(stripped) > 2 and stripped.endswith("$$") and stripped != "$$"):
                j = i + 1
                while j < len(lines) and "$$" not in lines[j]:
                    j += 1
                j = min(j, len(lines) - 1)
            emit(NodeType.EQUATION, start, offsets[j + 1])
            i = j + 1
            continue

        if stripped.startswith("<!--"):
            flush()
            j = i
            while j < len(lines) and "-->" not in lines[j]:
                j += 1
            j = min(j, len(lines) - 1)
            emit(NodeType.COMMENT, start, offsets[j + 1])
            i = j + 1
            continue

        heading = None
        level = 1
        if not plain:
            hm = _HEADING_RE.match(line)
            if hm:
                heading, level = hm.group(2), len(hm.group(1))
        elif para is None:
            nxt_blank = i + 1 >= len(lines) or not lines[i + 1].strip()
            if nxt_blank:
                heading = _plain_heading(line)
        if heading is not None:
            flush()
            emit(NodeType.HEADING, start, offsets[i + 1], level=level, title=heading)
            section = heading
            in_bib = heading.strip().lower().rstrip(":") in _BIB_HEADINGS
            i += 1
            continue

        if not plain and _TABLE_RE.match(line):
            flush()
            j = i
            while j + 1 < len(lines) and _TABLE_RE.match(lines[j + 1]):
                j += 1
            emit(NodeType.TABLE, start, offsets[j + 1])
            i = j + 1
            continue

        if not plain and _IMAGE_RE.match(line):
            flush()
            emit(NodeType.FIGURE, start, offsets[i + 1])
            i += 1
            continue

        if not plain and stripped.startswith(">"):
            flush()
            emit(NodeType.RAW, start, offsets[i + 1], role="blockquote")
            i += 1
            continue

        lm = _LIST_RE.match(line)
        if lm and not in_bib:
            flush()
            para = [NodeType.LIST_ITEM, start + lm.end(), offsets[i + 1]]
            i += 1
            continue

        if para is None:
            type_ = NodeType.BIBLIOGRAPHY if in_bib else NodeType.PARAGRAPH
            lead = len(line) - len(line.lstrip())
            para = [type_, start + lead, offsets[i + 1]]
        else:
            para[2] = offsets[i + 1]
        i += 1

    flush()
    return Document("text" if plain else "markdown", nodes, source=text)


def parse_text(text: str) -> Document:
    return parse_markdown(text, plain=True)
