"""Write Quill edits back into a .docx: as Word revision marks (tracked) or applied.

Paragraphs are never rebuilt. Only plain-text runs are split at edit boundaries;
deleted pieces are wrapped in ``<w:del>`` and new text is added in ``<w:ins>``
runs that copy the neighbouring run's formatting. Protected content (italic or
superscript runs, hyperlinks, Zotero/EndNote fields, equations) is never touched.

A paragraph is left unchanged (and reported) when an edit would require moving a
protected item, or when the paragraph's run structure is unusual (existing
tracked changes, content controls, tabs inside an edited run).
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path

from lxml import etree

from scholarlens.edit.diff import tokenize
from scholarlens.edit.engine import EditSession, Status
from scholarlens.ingest.docx import W, _run_format, _run_text
from scholarlens.masking import SENTINEL_RE, MaskResult

XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"


@dataclass
class _Piece:
    run: etree._Element
    start: int
    end: int
    plain: bool  # a direct child run of the paragraph holding only rPr + one w:t


@dataclass
class _Edit:
    start: int  # original char offsets
    end: int
    new: str


class Unsupported(Exception):
    pass


def _token_ranges(mask: MaskResult) -> list[tuple[str, int, int]]:
    """Tokens of the masked text with their character ranges in the original text."""
    sentinels = {s.sentinel: s for s in mask.spans}
    out, orig = [], 0
    for tok in tokenize(mask.masked):
        span = sentinels.get(tok)
        length = (span.end - span.start) if span else len(tok)
        out.append((tok, orig, orig + length))
        orig += length
    return out


def edits_in_original_coords(mask: MaskResult, edited_masked: str) -> list[_Edit]:
    a = _token_ranges(mask)
    b = tokenize(edited_masked)
    sm = SequenceMatcher(a=[t for t, _, _ in a], b=b, autojunk=False)
    text_len = a[-1][2] if a else 0
    edits = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        old_toks, new_toks = [t for t, _, _ in a[i1:i2]], b[j1:j2]
        if any(SENTINEL_RE.fullmatch(t) for t in old_toks + new_toks):
            raise Unsupported("edit moves a protected item")
        start = a[i1][1] if i1 < len(a) else text_len
        end = a[i2 - 1][2] if i2 > i1 else start
        edits.append(_Edit(start, end, "".join(new_toks)))
    return edits


def _pieces(p: etree._Element) -> list[_Piece]:
    """Visible runs in the same order and with the same filtering as ``ingest.docx``."""
    if p.find(f".//{W}ins") is not None or p.find(f".//{W}del") is not None:
        raise Unsupported("paragraph already contains tracked changes")
    if p.find(f".//{W}sdt") is not None:
        raise Unsupported("paragraph contains content controls")
    pieces, pos, depth = [], 0, 0
    for r in p.iter(W + "r"):
        fld = r.find(W + "fldChar")
        if fld is not None:
            kind = fld.get(W + "fldCharType")
            depth += 1 if kind == "begin" else -1 if kind == "end" else 0
            continue
        if r.find(W + "instrText") is not None:
            continue
        t = _run_text(r)
        if not t:
            continue
        children = [etree.QName(c).localname for c in r]
        plain = (r.getparent() is p and depth == 0 and children.count("t") == 1
                 and set(children) <= {"rPr", "t"} and _run_format(r) is None)
        pieces.append(_Piece(r, pos, pos + len(t), plain))
        pos += len(t)
    return pieces


def _make_run(template: etree._Element | None, text: str, *, deleted: bool = False) -> etree._Element:
    r = etree.Element(W + "r")
    if template is not None and template.find(W + "rPr") is not None:
        r.append(copy.deepcopy(template.find(W + "rPr")))
    t = etree.SubElement(r, W + ("delText" if deleted else "t"))
    t.text = text
    t.set(XML_SPACE, "preserve")
    return r


class _Revisions:
    def __init__(self, author: str, start_id: int) -> None:
        self.author = author
        self.date = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.ids = itertools.count(start_id)

    def wrap(self, tag: str, run: etree._Element) -> etree._Element:
        el = etree.Element(W + tag)
        el.set(W + "id", str(next(self.ids)))
        el.set(W + "author", self.author)
        el.set(W + "date", self.date)
        el.append(run)
        return el


def _top(el: etree._Element, p: etree._Element) -> etree._Element:
    """The ancestor of ``el`` that is a direct child of paragraph ``p`` (e.g. a w:del wrapper)."""
    while el.getparent() is not None and el.getparent() is not p:
        el = el.getparent()
    return el


@dataclass
class _Seg:
    el: etree._Element
    a: int
    b: int
    plain: bool


def apply_to_paragraph(p: etree._Element, edits: list[_Edit], *, tracked: bool, rev: _Revisions) -> None:
    if not edits:
        return
    pieces = _pieces(p)
    for e in edits:
        if any(not pc.plain for pc in pieces if pc.start < e.end and pc.end > e.start):
            raise Unsupported("edit touches a formatted or protected run")
        if e.start == e.end and pieces:
            left = [pc for pc in pieces if pc.start < e.start <= pc.end]
            right = [pc for pc in pieces if pc.start <= e.start < pc.end]
            if not ((left and left[-1].plain) or (right and right[0].plain)):
                raise Unsupported("insertion point is next to protected content only")

    cuts = sorted({x for e in edits for x in (e.start, e.end)})
    segs: list[_Seg] = []
    for pc in pieces:
        inner = [c for c in cuts if pc.start < c < pc.end] if pc.plain else []
        if not inner:
            segs.append(_Seg(pc.run, pc.start, pc.end, pc.plain))
            continue
        text = pc.run.find(W + "t").text or ""
        bounds = [pc.start, *inner, pc.end]
        prev = pc.run
        for a, b in zip(bounds, bounds[1:], strict=False):
            piece = _make_run(pc.run, text[a - pc.start:b - pc.start])
            prev.addnext(piece)
            prev = piece
            segs.append(_Seg(piece, a, b, True))
        p.remove(pc.run)

    for e in sorted(edits, key=lambda e: e.start, reverse=True):
        doomed = [s for s in segs if s.plain and e.start <= s.a and s.b <= e.end and s.b > s.a]
        left = [s for s in segs if s.b <= e.start]
        right = [s for s in segs if s.a >= e.end]
        if e.new:
            if doomed:
                anchor, after, template = doomed[-1].el, True, doomed[0].el
            elif left and left[-1].plain:
                anchor, after, template = left[-1].el, True, left[-1].el
            elif right and right[0].plain:
                anchor, after, template = right[0].el, False, right[0].el
            elif not segs:
                anchor, after, template = None, True, None
            else:
                raise Unsupported("no plain run to anchor the insertion")
            run = _make_run(template, e.new)
            node = rev.wrap("ins", run) if tracked else run
            if anchor is None:
                p.append(node)
            elif after:
                _top(anchor, p).addnext(node)
            else:
                _top(anchor, p).addprevious(node)
        for s in doomed:
            if tracked:
                t = s.el.find(W + "t")
                t.tag = W + "delText"
                t.set(XML_SPACE, "preserve")
                holder = etree.Element(W + "tmp")
                s.el.addprevious(holder)
                p.remove(s.el)
                p.replace(holder, rev.wrap("del", s.el))
            else:
                p.remove(s.el)


def write_docx(session: EditSession, src: str | Path, dst: str | Path, *, tracked: bool = True,
               author: str = "ScholarLens Quill") -> dict[str, list[str]]:
    """Returns {"written": [node ids], "kept_original": ["node id: reason"]}."""
    import docx

    d = docx.Document(str(src))
    body = list(d.element.body.iterchildren())
    existing = [int(x) for x in d.element.xpath("//w:ins/@w:id | //w:del/@w:id")]
    rev = _Revisions(author, max(existing, default=0) + 1)
    written, kept = [], []
    for e in session.edits:
        if e.status != Status.ACCEPTED or e.mask is None or e.edited_masked is None:
            continue
        node = session.document.get(e.node_id)
        p = body[node.meta["block_index"]]
        snapshot = copy.deepcopy(p)
        try:
            apply_to_paragraph(p, edits_in_original_coords(e.mask, e.edited_masked), tracked=tracked, rev=rev)
            written.append(e.node_id)
        except Unsupported as why:
            p.getparent().replace(p, snapshot)
            body[node.meta["block_index"]] = snapshot
            kept.append(f"{e.node_id}: {why}")
    d.save(str(dst))
    return {"written": written, "kept_original": kept}

