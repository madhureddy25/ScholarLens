"""Format-independent document model.

Every ingested manuscript (LaTeX, Markdown, DOCX, GROBID-TEI) becomes a
``Document``: an ordered list of typed ``Node`` objects. Only *prose* nodes are
ever eligible for editing or prose-level analysis; everything else (equations,
tables, code, bibliography, preamble, unknown constructs) is carried through
untouched.

For text-based formats the document keeps the original source string and each
node records its character span. ``Document.reconstruct`` then rebuilds the file
by splicing replacement text into prose spans only, which guarantees that every
byte outside an edited prose span is identical to the input.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum


class NodeType(StrEnum):
    METADATA = "metadata"  # preamble, front matter, author/affiliation blocks
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    CAPTION = "caption"
    EQUATION = "equation"
    TABLE = "table"
    FIGURE = "figure"
    CODE = "code"
    BIBLIOGRAPHY = "bibliography"
    COMMENT = "comment"
    RAW = "raw"  # anything we do not understand is passed through verbatim


PROSE_TYPES = frozenset({NodeType.PARAGRAPH, NodeType.LIST_ITEM, NodeType.CAPTION})


@dataclass
class Node:
    id: str
    type: NodeType
    text: str
    """Node content exactly as it appears in the source (for span-based formats)."""
    start: int | None = None
    end: int | None = None
    section: str | None = None
    """Title of the enclosing section, e.g. "Methods" (useful for per-section analysis)."""
    meta: dict = field(default_factory=dict)

    @property
    def is_prose(self) -> bool:
        return self.type in PROSE_TYPES

    @property
    def has_span(self) -> bool:
        return self.start is not None and self.end is not None


class ReconstructionError(ValueError):
    pass


@dataclass
class Document:
    source_format: str
    nodes: list[Node]
    source: str | None = None
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ access
    def prose(self) -> Iterator[Node]:
        return (n for n in self.nodes if n.is_prose)

    def get(self, node_id: str) -> Node:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)

    def __iter__(self) -> Iterator[Node]:
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)

    def stats(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for n in self.nodes:
            counts[n.type.value] = counts.get(n.type.value, 0) + 1
        return counts

    # ---------------------------------------------------------- reconstruction
    def reconstruct(self, replacements: Mapping[str, str] | None = None) -> str:
        """Rebuild the source with the given prose nodes replaced.

        Raises ``ReconstructionError`` if a replacement targets a non-prose node,
        a node without a span, or if replacement spans overlap.
        """
        if self.source is None:
            raise ReconstructionError(
                f"{self.source_format!r} documents have no source text; use render_markdown()"
            )
        replacements = dict(replacements or {})
        targets: list[Node] = []
        for node_id in replacements:
            node = self.get(node_id)
            if not node.is_prose:
                raise ReconstructionError(f"refusing to modify non-prose node {node_id} ({node.type.value})")
            if not node.has_span:
                raise ReconstructionError(f"node {node_id} has no source span")
            targets.append(node)
        targets.sort(key=lambda n: n.start)  # type: ignore[arg-type, return-value]
        for a, b in zip(targets, targets[1:]):
            if b.start < a.end:  # type: ignore[operator]
                raise ReconstructionError(f"overlapping replacements: {a.id} and {b.id}")

        out: list[str] = []
        cursor = 0
        for node in targets:
            out.append(self.source[cursor : node.start])
            out.append(replacements[node.id])
            cursor = node.end  # type: ignore[assignment]
        out.append(self.source[cursor:])
        return "".join(out)

    def render_markdown(self, replacements: Mapping[str, str] | None = None) -> str:
        """Format-independent rendering, used for formats without a source string."""
        replacements = replacements or {}
        parts: list[str] = []
        for n in self.nodes:
            text = replacements.get(n.id, n.text)
            if n.type == NodeType.TITLE:
                parts.append(f"# {text}")
            elif n.type == NodeType.HEADING:
                level = int(n.meta.get("level", 1))
                parts.append(f"{'#' * min(level + 1, 6)} {text}")
            elif n.type == NodeType.LIST_ITEM:
                parts.append(f"- {text}")
            elif n.type == NodeType.EQUATION:
                parts.append(f"$$\n{text}\n$$")
            elif n.type == NodeType.CODE:
                parts.append(f"```\n{text}\n```")
            elif n.type == NodeType.CAPTION:
                parts.append(f"*{text}*")
            elif n.type == NodeType.COMMENT:
                continue
            else:
                parts.append(text)
        return "\n\n".join(p for p in parts if p.strip()) + "\n"


class NodeFactory:
    """Issues stable, sequential node ids (``n0001``, ``n0002``…)."""

    def __init__(self) -> None:
        self._i = 0

    def make(self, type_: NodeType, text: str, start: int | None = None, end: int | None = None,
             section: str | None = None, **meta) -> Node:
        self._i += 1
        return Node(id=f"n{self._i:04d}", type=type_, text=text, start=start, end=end,
                    section=section, meta=meta)
