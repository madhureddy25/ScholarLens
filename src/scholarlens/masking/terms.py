"""Domain-term protection.

Two tiers:

* **Hard terms** are masked: tokens that an editor must never alter and that
  are identifiable with high precision — gene/protein symbols (BRCA1, IL-6),
  alphanumeric identifiers (CRISPR-Cas9, SARS-CoV-2, GPT-4), acronyms and
  CamelCase names (DNA, mRNA, SciBERT), chemical formulas (H2O, NaCl), Greek
  letters, abbreviated binomials (E. coli), and any user glossary entry.
* **Soft terms** are only *verified*: broader domain concepts found by a NER
  model such as SciSpacy (``en_core_sci_lg``), e.g. "double-strand breaks".
  Masking these would make text unreadable to an editor; instead the integrity
  check confirms each one still appears after editing.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Protocol

from scholarlens.masking.kinds import Kind

RawSpan = tuple[int, int, Kind]

_GREEK = "α-ωΑ-Ω"
_TOKEN = re.compile(
    rf"[A-Za-z0-9{_GREEK}][A-Za-z0-9{_GREEK}'’]*(?:[-–/+][A-Za-z0-9{_GREEK}]+)*[+\-]?(?![A-Za-z0-9])"
)
_ORDINAL = re.compile(r"^\d+(?:st|nd|rd|th|s)$")
_PURE_NUMBER = re.compile(r"^[\d.,]+$")
_BINOMIAL = re.compile(r"\b[A-Z]\.\s?[a-z]{3,}\b")
_ROMAN = re.compile(r"^[IVXLC]+$")


def _is_hard_term(tok: str) -> bool:
    if len(tok) < 2 or _PURE_NUMBER.match(tok) or _ORDINAL.match(tok):
        return False
    has_digit = any(c.isdigit() for c in tok)
    has_alpha = any(c.isalpha() for c in tok)
    if has_digit and has_alpha:
        return True
    if any("Ͱ" <= c <= "Ͽ" for c in tok):
        return True
    uppers = sum(1 for c in tok if c.isupper())
    if uppers >= 2 and not _ROMAN.match(tok):
        return True
    # lowercase-led mixed case: mRNA, siRNA, iPhone, pH
    return len(tok) >= 2 and tok[0].islower() and uppers >= 1


def hard_term_spans(text: str, glossary: Iterable[str] = ()) -> list[RawSpan]:
    spans: list[RawSpan] = []
    for term in sorted(set(glossary), key=len, reverse=True):
        if term:
            for m in re.finditer(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text):
                spans.append((m.start(), m.end(), Kind.TERM))
    spans += [(m.start(), m.end(), Kind.TERM) for m in _BINOMIAL.finditer(text)]
    for m in _TOKEN.finditer(text):
        if _is_hard_term(m.group(0)):
            spans.append((m.start(), m.end(), Kind.TERM))
    return spans


class SoftTermDetector(Protocol):
    def __call__(self, text: str) -> list[str]: ...


class SpacyTermDetector:
    """Wraps a spaCy / SciSpacy pipeline; returns entity strings to verify.

    Install: ``pip install scispacy`` and the ``en_core_sci_lg`` model wheel
    from the SciSpacy releases page. Any spaCy model with NER works.
    """

    def __init__(self, model: str = "en_core_sci_lg", min_len: int = 4) -> None:
        import spacy  # optional dependency

        self.nlp = spacy.load(model, disable=["parser", "lemmatizer"])
        self.min_len = min_len

    def __call__(self, text: str) -> list[str]:
        seen: dict[str, None] = {}
        for ent in self.nlp(text).ents:
            s = ent.text.strip()
            if len(s) >= self.min_len and "⟦" not in s and "⟧" not in s:
                seen.setdefault(s, None)
        return list(seen)
