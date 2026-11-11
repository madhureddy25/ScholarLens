"""PDF ingestion via GROBID, and GROBID TEI-XML → Document parsing.

GROBID (https://github.com/kermitt2/grobid) runs as a separate service, e.g.
``docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.1``. Its TEI output marks
inline citations as ``<ref type="bibr">`` and figure/table/formula references
as ``<ref type="figure|table|formula">``; these become protected spans.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from urllib import request

from lxml import etree

from scholarlens.core.document import Document, NodeFactory, NodeType

TEI = "{http://www.tei-c.org/ns/1.0}"
_REF_KIND = {"bibr": "citation", "figure": "figure-ref", "table": "table-ref", "formula": "equation-ref",
             "foot": "footnote-ref"}


def _inline_text(el) -> tuple[str, list[dict]]:
    """Flatten an element to text, recording spans of <ref> and <formula> children."""
    parts: list[str] = [el.text or ""]
    pos = len(parts[0])
    spans: list[dict] = []
    for child in el:
        tag = etree.QName(child).localname
        content = "".join(child.itertext())
        if tag in ("ref", "formula") and content:
            reason = _REF_KIND.get(child.get("type", ""), "reference") if tag == "ref" else "inline-math"
            spans.append({"start": pos, "end": pos + len(content), "reason": reason,
                          "target": child.get("target")})
        parts.append(content)
        pos += len(content)
        tail = child.tail or ""
        parts.append(tail)
        pos += len(tail)
    return "".join(parts), spans


def parse_tei(xml: str | bytes) -> Document:
    if isinstance(xml, str):
        xml = xml.encode("utf-8")
    root = etree.fromstring(xml)
    f = NodeFactory()
    nodes = []
    meta: dict = {}

    title_el = root.find(f".//{TEI}teiHeader//{TEI}titleStmt/{TEI}title")
    if title_el is not None and "".join(title_el.itertext()).strip():
        meta["title"] = "".join(title_el.itertext()).strip()
        nodes.append(f.make(NodeType.TITLE, meta["title"]))

    for p in root.findall(f".//{TEI}profileDesc/{TEI}abstract//{TEI}p"):
        text, spans = _inline_text(p)
        if text.strip():
            nodes.append(f.make(NodeType.PARAGRAPH, text.strip(), section="Abstract",
                                protected=_shift(spans, text)))

    body = root.find(f".//{TEI}text/{TEI}body")
    if body is not None:
        section = None
        for el in body.iter():
            tag = etree.QName(el).localname if isinstance(el.tag, str) else None
            if tag == "head" and etree.QName(el.getparent()).localname == "div":
                section = "".join(el.itertext()).strip()
                nodes.append(f.make(NodeType.HEADING, section, section=section, level=1, n=el.get("n")))
            elif tag == "p" and etree.QName(el.getparent()).localname == "div":
                text, spans = _inline_text(el)
                if text.strip():
                    nodes.append(f.make(NodeType.PARAGRAPH, text.strip(), section=section,
                                        protected=_shift(spans, text)))
            elif tag == "formula" and etree.QName(el.getparent()).localname == "div":
                nodes.append(f.make(NodeType.EQUATION, "".join(el.itertext()).strip(), section=section))
            elif tag == "figure":
                kind = NodeType.TABLE if el.get("type") == "table" else NodeType.FIGURE
                fig = f.make(kind, "".join(el.itertext()).strip(), section=section)
                nodes.append(fig)
                desc = el.find(f"{TEI}figDesc")
                if desc is not None:
                    text, spans = _inline_text(desc)
                    if text.strip():
                        nodes.append(f.make(NodeType.CAPTION, text.strip(), section=section,
                                            parent=fig.id, protected=_shift(spans, text)))

    for bib in root.findall(f".//{TEI}back//{TEI}listBibl/{TEI}biblStruct"):
        text = " ".join(" ".join(bib.itertext()).split())
        if text:
            nodes.append(f.make(NodeType.BIBLIOGRAPHY, text, section="References", xml_id=bib.get(
                "{http://www.w3.org/XML/1998/namespace}id")))

    return Document("tei", nodes, source=None, meta=meta)


def _shift(spans: list[dict], raw: str) -> list[dict]:
    """Adjust spans for the leading whitespace removed by ``.strip()``."""
    lead = len(raw) - len(raw.lstrip())
    return [{**s, "start": s["start"] - lead, "end": s["end"] - lead} for s in spans]


class GrobidClient:
    def __init__(self, url: str | None = None, timeout: float = 120.0) -> None:
        self.url = (url or os.environ.get("SCHOLARLENS_GROBID_URL", "http://localhost:8070")).rstrip("/")
        self.timeout = timeout

    def process_fulltext(self, pdf_path: str | Path) -> str:
        """POST a PDF to GROBID's processFulltextDocument and return TEI XML."""
        pdf = Path(pdf_path).read_bytes()
        boundary = uuid.uuid4().hex
        fields = [("consolidateHeader", "0"), ("segmentSentences", "0"),
                  ("teiCoordinates", "ref"), ("teiCoordinates", "formula")]
        body = b""
        for k, v in fields:
            body += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="input"; filename="{Path(pdf_path).name}"'
                 f"\r\nContent-Type: application/pdf\r\n\r\n").encode() + pdf + f"\r\n--{boundary}--\r\n".encode()
        req = request.Request(
            f"{self.url}/api/processFulltextDocument", data=body, method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "Accept": "application/xml"},
        )
        with request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 (configured URL)
            return resp.read().decode("utf-8")


def parse_pdf(path: str | Path, client: GrobidClient | None = None) -> Document:
    doc = parse_tei((client or GrobidClient()).process_fulltext(path))
    doc.source_format = "pdf"
    doc.meta["path"] = str(path)
    return doc
