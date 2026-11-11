from __future__ import annotations

from enum import Enum


class Kind(str, Enum):
    """Categories of protected content. Values are the sentinel prefixes."""

    PROT = "PROT"  # pre-declared by the parser (DOCX formatting runs, TEI <ref>)
    MATH = "MATH"
    CITE = "CITE"
    CMD = "CMD"  # LaTeX commands, comments, accents
    REF = "REF"  # figure / table / equation / section cross-references
    URL = "URL"
    NUM = "NUM"  # numbers, statistics, measurements
    TERM = "TERM"  # domain terms: gene symbols, acronyms, formulas, glossary entries


PRIORITY = {
    Kind.PROT: 100,
    Kind.MATH: 95,
    Kind.CITE: 90,
    Kind.CMD: 85,
    Kind.REF: 80,
    Kind.URL: 75,
    Kind.NUM: 60,
    Kind.TERM: 50,
}

# How protected kinds read when a masked text is rendered for statistical
# scoring: neutral surfaces stop markup from skewing perplexity / burstiness.
RENDER_SURFACE = {Kind.CITE: "[1]", Kind.URL: "URL"}

PROTECTED_REASON_KIND = {
    "citation": Kind.CITE,
    "citation-field": Kind.CITE,
    "figure-ref": Kind.REF,
    "table-ref": Kind.REF,
    "equation-ref": Kind.REF,
    "footnote-ref": Kind.REF,
    "reference": Kind.REF,
    "inline-math": Kind.MATH,
    "hyperlink": Kind.URL,
}
