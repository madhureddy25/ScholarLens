"""Span detectors for protected content.

Each detector returns ``(start, end, kind)`` triples over a prose string. The
masker resolves overlaps by priority and length, so detectors are allowed to be
greedy and to overlap each other.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from scholarlens.masking.kinds import Kind

RawSpan = tuple[int, int, Kind]

# --------------------------------------------------------------------- LaTeX
_LATEX_COMMENT = re.compile(r"(?<!\\)%[^\n]*")
_LATEX_INLINE_MATH = re.compile(r"(?<![\\$])\$(?!\$)(?:\\.|[^$\\])+?\$|\\\((?:.|\n)*?\\\)")
_REF_WORDS = (
    r"(?i:fig(?:ure)?s?\.?|tables?|tabs?\.|eqs?\.|equations?|sections?|secs?\.|§§?|appendi(?:x|ces)|"
    r"algorithms?|alg\.|theorems?|lemmas?|definitions?|corollar(?:y|ies)|propositions?|chapters?|ch\.|"
    r"supplementary\s+(?:fig(?:ure)?s?\.?|tables?|notes?|material))"
)
_LATEX_CITE = re.compile(r"~?\\(?:[a-zA-Z]*cite[a-zA-Z]*|nocite)\*?(?:\s*\[[^\]]*\]){0,2}\s*\{[^}]*\}")
_REF_CMD = r"\\(?:ref|eqref|autoref|cref|Cref|pageref|nameref|vref|labelcref|label)\*?\{[^}]*\}"
_LATEX_REF = re.compile(r"(?:" + _REF_WORDS + r"\s*)?~?(?:\(" + _REF_CMD + r"\)|" + _REF_CMD + r")")
_LATEX_ACCENT_WORD = re.compile(
    r"[A-Za-z]*(?:\{?\\['\"`^~=.][A-Za-z]\}?|\{?\\[uvHcdbrkt]\{[A-Za-z]\}\}?)[A-Za-z]*"
)
_LATEX_CMD_START = re.compile(r"\\(?:[a-zA-Z@]+\*?|[^a-zA-Z@\s])")
_LATEX_BRACE_GROUP_CMD = re.compile(r"\{\\[a-zA-Z]+[^{}]*\}")
_CITE_OR_REF_NAME = re.compile(
    r"\\(?:[a-zA-Z]*cite[a-zA-Z]*|nocite|ref|eqref|autoref|cref|Cref|pageref|nameref|vref|labelcref|label)\*?$"
)


def _match_brace(text: str, i: int, open_: str = "{", close: str = "}") -> int:
    """Given text[i] == open_, return index just past the matching close (or -1)."""
    depth = 0
    j = i
    while j < len(text):
        c = text[j]
        if c == "\\":
            j += 2
            continue
        if c == open_:
            depth += 1
        elif c == close:
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return -1


def latex_spans(text: str) -> list[RawSpan]:
    spans: list[RawSpan] = []
    spans += [(m.start(), m.end(), Kind.CMD) for m in _LATEX_COMMENT.finditer(text)]
    spans += [(m.start(), m.end(), Kind.MATH) for m in _LATEX_INLINE_MATH.finditer(text)]
    spans += [(m.start(), m.end(), Kind.CITE) for m in _LATEX_CITE.finditer(text)]
    spans += [(m.start(), m.end(), Kind.REF) for m in _LATEX_REF.finditer(text)]
    spans += [(m.start(), m.end(), Kind.CMD) for m in _LATEX_ACCENT_WORD.finditer(text)]
    spans += [(m.start(), m.end(), Kind.CMD) for m in _LATEX_BRACE_GROUP_CMD.finditer(text)]
    # generic commands, consuming [optional] and {balanced} arguments
    for m in _LATEX_CMD_START.finditer(text):
        if _CITE_OR_REF_NAME.match(m.group(0)):
            continue  # handled by the dedicated citation / reference patterns
        if m.group(0) == "\\%" and m.start() > 0 and text[m.start() - 1].isdigit():
            continue  # part of a percentage, handled as a quantity
        end = m.end()
        if m.group(0)[1:2].isalpha() or m.group(0)[1:2] == "@":
            for _ in range(4):
                k = end
                while k < len(text) and text[k] in " \t":
                    k += 1
                if k < len(text) and text[k] == "{":
                    close = _match_brace(text, k)
                elif k < len(text) and text[k] == "[":
                    close = text.find("]", k) + 1 or -1
                else:
                    break
                if close == -1:
                    break
                end = close
        kind = Kind.URL if m.group(0) in ("\\url", "\\href") else Kind.CMD
        spans.append((m.start(), end, kind))
    return spans


# ----------------------------------------------------------- citations (text)
_YEAR = r"(?:1[89]|20)\d{2}[a-z]?"
_SURNAME = r"(?:(?:van|von|de|der|den|da|del|di|du|le|la)\s+)*[A-Z][A-Za-z'’\-]+"
_NUMERIC_CITE = re.compile(r"\[\s*\d{1,4}(?:\s*[-–—,;]\s*\d{1,4})*\s*\]")
_PAREN_CITE = re.compile(
    r"\((?:(?:e\.g\.|i\.e\.|see\s+also|see|cf\.)\s*,?\s*)?" + _SURNAME +
    r"[^()\n]{0,200}?\b(?:" + _YEAR + r"|n\.d\.|in\s+press)\b[^()\n]{0,200}?\)"
)
_NARRATIVE_CITE = re.compile(
    _SURNAME + r"(?:\s+et\s+al\.?|\s+(?:and|&)\s+" + _SURNAME + r")?\s*[\(\[](?:" + _YEAR +
    r"|n\.d\.|in\s+press)(?:\s*,\s*pp?\.\s*\d+(?:\s*[-–]\s*\d+)?)?[\)\]]"
)


def citation_spans(text: str) -> list[RawSpan]:
    out: list[RawSpan] = []
    for rx in (_NUMERIC_CITE, _PAREN_CITE, _NARRATIVE_CITE):
        out += [(m.start(), m.end(), Kind.CITE) for m in rx.finditer(text)]
    return out


# -------------------------------------------------------- cross-references
_REF_NUM = r"(?:\([A-Z]?S?\d+(?:\.\d+)*[a-z]?\)|[A-Z]?S?\d+(?:\.\d+)*[a-z]?|[A-Z](?:\.\d+)*\b|[IVX]+\b)"
_TEXT_REF = re.compile(
    r"\b" + _REF_WORDS + r"\s*~?\s*" + _REF_NUM +
    r"(?:\s*(?:[-–,]|and|&)\s*" + _REF_NUM + r")*"
)


def reference_spans(text: str) -> list[RawSpan]:
    return [(m.start(), m.end(), Kind.REF) for m in _TEXT_REF.finditer(text)]


# ---------------------------------------------------------------- URLs etc.
_URL = re.compile(r"https?://[^\s)\]}>,;\"']*[^\s)\]}>,;.\"']")
_DOI = re.compile(r"\b(?:doi:\s*)?10\.\d{4,9}/[^\s,;)\]]*[^\s,;)\].]", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")


def url_spans(text: str) -> list[RawSpan]:
    out: list[RawSpan] = []
    for rx in (_URL, _DOI, _EMAIL):
        out += [(m.start(), m.end(), Kind.URL) for m in rx.finditer(text)]
    return out


# --------------------------------------------------------------- quantities
_STAT = re.compile(
    r"(?<![\w])(?:[pPnNrRtFdkzZ]|SD|SE|SEM|OR|HR|RR|AUC|CI)\s*(?:\([^)\n]{1,12}\))?\s*"
    r"(?:[<>=≤≥≈]|&lt;|&gt;|\\leq?|\\geq?|\\approx)\s*[-−]?(?:\d{1,3}(?:,\d{3})+|\d*\.?\d+)"
    r"(?:\s*[×x]\s*10\^?\{?[-−]?\d+\}?)?"
)
_PLUSMINUS = re.compile(r"[-−]?\d+(?:[.,]\d+)?\s*(?:±|\+/-|\\pm|\$\\pm\$)\s*\d+(?:[.,]\d+)?")
_NUMBER = re.compile(
    r"(?<![\w.])(?<![A-Za-z0-9][-–])[-−+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?(?:\s*(?:%|\\%|‰))?(?![\w])"
)


def quantity_spans(text: str) -> list[RawSpan]:
    out: list[RawSpan] = []
    for rx in (_STAT, _PLUSMINUS, _NUMBER):
        out += [(m.start(), m.end(), Kind.NUM) for m in rx.finditer(text)]
    return out


def literal_sentinel_char_spans(text: str, chars: Iterable[str]) -> list[RawSpan]:
    """Protect any literal sentinel delimiter already present in the source."""
    return [(i, i + 1, Kind.CMD) for i, c in enumerate(text) if c in set(chars)]
