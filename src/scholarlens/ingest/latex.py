"""LaTeX → Document parser.

Splits a LaTeX source into typed blocks with exact character spans, using
pylatexenc's tolerant walker. Inline constructs (citations, inline math,
formatting commands) stay inside the prose node and are handled later by the
masking engine; block constructs become their own non-prose nodes.

Conservative by default: an environment we do not recognise is emitted as a RAW
node and is never edited.
"""

from __future__ import annotations

import re

from pylatexenc.latexwalker import (
    LatexCharsNode,
    LatexCommentNode,
    LatexEnvironmentNode,
    LatexGroupNode,
    LatexMacroNode,
    LatexMathNode,
    LatexWalker,
)

from scholarlens.core.document import Document, NodeFactory, NodeType

VERBATIM_ENVS = ("verbatim", "verbatim*", "Verbatim", "lstlisting", "minted", "comment", "alltt")
EQUATION_ENVS = {
    "equation", "equation*", "align", "align*", "gather", "gather*", "multline", "multline*",
    "eqnarray", "eqnarray*", "displaymath", "flalign", "flalign*", "alignat", "alignat*",
    "dmath", "dmath*", "subequations",
}
FIGURE_ENVS = {"figure", "figure*", "wrapfigure", "subfigure", "SCfigure"}
TABLE_ENVS = {"table", "table*", "tabular", "tabular*", "tabularx", "longtable", "sidewaystable"}
CODE_ENVS = set(VERBATIM_ENVS) | {"algorithm", "algorithm*", "algorithmic", "algorithm2e", "program"}
BIB_ENVS = {"thebibliography"}
LIST_ENVS = {"itemize", "enumerate", "description", "compactitem", "compactenum", "inparaenum"}
PROSE_CONTAINER_ENVS = {"abstract", "quote", "quotation", "acknowledgments", "acknowledgements", "acks"}

HEADING_MACROS = {
    "part": 0, "chapter": 0, "section": 1, "subsection": 2, "subsubsection": 3,
    "paragraph": 4, "subparagraph": 5,
}
METADATA_MACROS = {
    "title", "author", "date", "maketitle", "affiliation", "affil", "thanks", "keywords",
    "email", "address", "institute", "tableofcontents", "listoffigures", "listoftables",
}
BIB_MACROS = {"bibliography", "bibliographystyle", "printbibliography", "addbibresource"}
BLOCK_MACROS = {
    "newpage", "clearpage", "cleardoublepage", "pagebreak", "appendix", "input", "include",
    "vspace", "vskip", "bigskip", "medskip", "smallskip", "FloatBarrier", "usepackage",
    "newcommand", "renewcommand", "def", "let", "setlength", "setcounter", "centering",
}
PARBREAK_MACROS = {"par"}
NO_ARG_MACROS = {
    "maketitle", "tableofcontents", "listoffigures", "listoftables", "newpage", "clearpage",
    "cleardoublepage", "pagebreak", "appendix", "bigskip", "medskip", "smallskip",
    "FloatBarrier", "centering", "printbibliography",
}

_PARBREAK_RE = re.compile(r"\n[ \t]*\n\s*")
_LEADING_NEWLINE_RE = re.compile(r"^[ \t]*\n")
_OPTARG_RE = re.compile(r"^\s*(\[[^\]]*\]\s*)*$")


def _blank_verbatim(src: str) -> str:
    """Replace the bodies of verbatim-like environments and ``\\verb`` with spaces.

    Offsets are preserved, so spans computed on the blanked text are valid on
    the original. This stops stray braces or ``%`` inside code from derailing
    the LaTeX walker.
    """
    names = "|".join(re.escape(n) for n in VERBATIM_ENVS)
    env_re = re.compile(
        r"(\\begin\{(" + names + r")\}(?:\[[^\]]*\])?(?:\{[^}]*\})?)(.*?)(\\end\{\2\})", re.S
    )

    def blank(m: re.Match) -> str:
        body = re.sub(r"[^\n]", " ", m.group(3))
        return m.group(1) + body + m.group(4)

    out = env_re.sub(blank, src)
    verb_re = re.compile(r"(\\verb\*?)([^a-zA-Z\s])(.*?)(\2)")
    return verb_re.sub(lambda m: m.group(1) + m.group(2) + "x" * len(m.group(3)) + m.group(4), out)


class _Builder:
    def __init__(self, src: str) -> None:
        self.src = src
        self.scan = _blank_verbatim(src)
        self.f = NodeFactory()
        self.nodes = []
        self.section: str | None = None
        self.prose_type = NodeType.PARAGRAPH
        self._para: list[int] | None = None
        self._comment_newline = False
        self.meta: dict = {}

    # ------------------------------------------------------------- emitting
    def emit(self, type_: NodeType, start: int, end: int, **meta):
        node = self.f.make(type_, self.src[start:end], start, end, section=self.section, **meta)
        self.nodes.append(node)
        return node

    def flush(self) -> None:
        if self._para is None:
            return
        start, end = self._para
        self._para = None
        text = self.src[start:end]
        end = start + len(text.rstrip())
        if end > start:
            self.emit(self.prose_type, start, end)

    def extend(self, a: int, b: int) -> None:
        if self._para is None:
            seg = self.src[a:b]
            stripped = seg.lstrip()
            if not stripped:
                return
            a += len(seg) - len(stripped)
            self._para = [a, b]
        else:
            self._para[1] = b

    # -------------------------------------------------------------- helpers
    def _text_of(self, group: LatexGroupNode) -> str:
        return self.src[group.pos + 1 : group.pos + group.len - 1].strip()

    @staticmethod
    def _last_group_arg(node: LatexMacroNode) -> LatexGroupNode | None:
        if node.nodeargd is None:
            return None
        groups = [a for a in node.nodeargd.argnlist if isinstance(a, LatexGroupNode)]
        return groups[-1] if groups else None

    def _trailing_group(self, nodelist, i: int) -> tuple[LatexGroupNode | None, int]:
        """For macros pylatexenc does not know the signature of, find the ``{…}``
        argument that follows. Returns (group, index of the group) or (None, i)."""
        j = i + 1
        while j < len(nodelist) and j <= i + 3:
            n = nodelist[j]
            if isinstance(n, LatexGroupNode):
                return n, j
            if isinstance(n, LatexCharsNode) and _OPTARG_RE.match(n.chars) and "\n\n" not in n.chars:
                j += 1
                continue
            break
        return None, i

    # ------------------------------------------------------------------ walk
    def build(self) -> Document:
        walker = LatexWalker(self.scan, tolerant_parsing=True)
        top, _, _ = walker.get_latex_nodes(pos=0)
        doc_env = next(
            (n for n in top if isinstance(n, LatexEnvironmentNode) and n.environmentname == "document"),
            None,
        )
        if doc_env is not None:
            begin = self.scan.index("\\begin{document}", doc_env.pos) + len("\\begin{document}")
            self.emit(NodeType.METADATA, 0, begin, role="preamble")
            self._collect_title(top)
            self.walk(doc_env.nodelist)
            self.flush()
            end_tag = doc_env.pos + doc_env.len
            if end_tag < len(self.src) and self.src[end_tag:].strip():
                self.emit(NodeType.RAW, end_tag, len(self.src), role="after-document")
        else:
            self._collect_title(top)
            self.walk(top)
            self.flush()
        self.nodes.sort(key=lambda n: (n.start, -(n.end or 0)))
        return Document("latex", self.nodes, source=self.src, meta=self.meta)

    def _collect_title(self, nodelist) -> None:
        for i, n in enumerate(nodelist):
            if isinstance(n, LatexMacroNode) and n.macroname == "title":
                g = self._last_group_arg(n) or self._trailing_group(nodelist, i)[0]
                if g is not None:
                    self.meta["title"] = self._text_of(g)
                return

    def walk(self, nodelist) -> None:
        i = 0
        while i < len(nodelist):
            n = nodelist[i]
            i += 1
            if n is None:
                continue
            if isinstance(n, LatexCharsNode):
                self._chars(n)
                continue
            self._comment_newline = False
            if isinstance(n, LatexCommentNode):
                if self._para is not None:
                    self.extend(n.pos, n.pos + n.len)  # masked later as a command
                else:
                    self.emit(NodeType.COMMENT, n.pos, n.pos + n.len)
                self._comment_newline = self.src[n.pos : n.pos + n.len].endswith("\n")
            elif isinstance(n, LatexMathNode):
                if n.displaytype == "inline":
                    self.extend(n.pos, n.pos + n.len)
                else:
                    self.flush()
                    self.emit(NodeType.EQUATION, n.pos, n.pos + n.len)
            elif isinstance(n, LatexEnvironmentNode):
                self._environment(n)
            elif isinstance(n, LatexMacroNode):
                i = self._macro(n, nodelist, i - 1) + 1
            else:  # groups, specials (~, --, ``)
                self.extend(n.pos, n.pos + n.len)

    def _chars(self, n: LatexCharsNode) -> None:
        text = self.scan[n.pos : n.pos + n.len]
        if self._comment_newline and _LEADING_NEWLINE_RE.match(text):
            self.flush()  # "% comment\n\n" — the blank line is split across nodes
        self._comment_newline = False
        pos = n.pos
        for m in _PARBREAK_RE.finditer(text):
            self.extend(pos, n.pos + m.start())
            self.flush()
            pos = n.pos + m.end()
        self.extend(pos, n.pos + n.len)

    def _environment(self, n: LatexEnvironmentNode) -> None:
        name = n.environmentname
        start, end = n.pos, n.pos + n.len
        if name in EQUATION_ENVS:
            self.flush()
            self.emit(NodeType.EQUATION, start, end, env=name)
        elif name in FIGURE_ENVS or name in TABLE_ENVS:
            self.flush()
            kind = NodeType.FIGURE if name in FIGURE_ENVS else NodeType.TABLE
            parent = self.emit(kind, start, end, env=name)
            self._captions(n.nodelist, parent.id)
        elif name in CODE_ENVS:
            self.flush()
            self.emit(NodeType.CODE, start, end, env=name)
        elif name in BIB_ENVS:
            self.flush()
            self.emit(NodeType.BIBLIOGRAPHY, start, end, env=name)
        elif name in LIST_ENVS:
            self.flush()
            outer = self.prose_type
            self.prose_type = NodeType.LIST_ITEM
            self.walk(n.nodelist)
            self.flush()
            self.prose_type = outer
        elif name in PROSE_CONTAINER_ENVS:
            self.flush()
            outer_section = self.section
            if name == "abstract":
                self.section = "Abstract"
            self.walk(n.nodelist)
            self.flush()
            self.section = outer_section if name == "abstract" else self.section
        else:
            self.flush()
            self.emit(NodeType.RAW, start, end, env=name)

    def _macro(self, n: LatexMacroNode, nodelist, i: int) -> int:
        """Handle a macro at index i; returns the index of the last node consumed."""
        name = n.macroname
        start, end = n.pos, n.pos + n.len
        if name in HEADING_MACROS:
            self.flush()
            group = self._last_group_arg(n)
            last = i
            if group is None:
                group, last = self._trailing_group(nodelist, i)
            if group is not None:
                end = group.pos + group.len
            title = self._text_of(group) if group is not None else ""
            self.section = title
            self.emit(NodeType.HEADING, start, end, level=HEADING_MACROS[name], title=title)
            return last
        if name == "item" or name in PARBREAK_MACROS:
            self.flush()
            return i
        if name in METADATA_MACROS or name in BIB_MACROS or name in BLOCK_MACROS:
            self.flush()
            last = i
            if name not in NO_ARG_MACROS and (n.nodeargd is None or not n.nodeargd.argnlist):
                group, last = self._trailing_group(nodelist, i)
                if group is not None:
                    end = group.pos + group.len
            if name in METADATA_MACROS:
                kind = NodeType.METADATA
            elif name in BIB_MACROS:
                kind = NodeType.BIBLIOGRAPHY
            else:
                kind = NodeType.RAW
            self.emit(kind, start, end, macro=name)
            return last
        self.extend(start, end)
        return i

    def _captions(self, nodelist, parent_id: str) -> None:
        for i, n in enumerate(nodelist or []):
            if isinstance(n, LatexMacroNode) and n.macroname == "caption":
                group = self._last_group_arg(n) or self._trailing_group(nodelist, i)[0]
                if group is not None and group.len > 2:
                    a, b = group.pos + 1, group.pos + group.len - 1
                    seg = self.src[a:b]
                    a += len(seg) - len(seg.lstrip())
                    b = a + len(self.src[a:b].rstrip())
                    if b > a:
                        self.emit(NodeType.CAPTION, a, b, parent=parent_id)
            elif isinstance(n, LatexEnvironmentNode):
                self._captions(n.nodelist, parent_id)


def parse_latex(src: str) -> Document:
    return _Builder(src).build()
