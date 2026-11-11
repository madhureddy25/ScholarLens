"""Phase 2 — mask protected content behind sentinels, verify, and restore.

    masker = Masker()
    m = masker.mask("Prior work (Smith et al., 2024) used CRISPR-Cas9 [3].")
    m.masked   -> "Prior work ⟦CITE_1⟧ used ⟦TERM_1⟧ ⟦CITE_2⟧."
    ...edit m.masked...
    report = m.verify(edited)        # every sentinel exactly once, none invented
    restored = m.unmask(edited)      # raises if the report is not ok

Sentinels use mathematical white square brackets (U+27E6/U+27E7). They never
occur in normal prose, survive tokenisation by hosted LLM APIs (where custom
vocabulary tokens are not available), and are trivially machine-checkable.
"""

from __future__ import annotations

import re
from bisect import bisect_left
from dataclasses import dataclass, field

from scholarlens.masking import patterns
from scholarlens.masking.kinds import PRIORITY, PROTECTED_REASON_KIND, RENDER_SURFACE, Kind
from scholarlens.masking.terms import SoftTermDetector, hard_term_spans

OPEN, CLOSE = "⟦", "⟧"
SENTINEL_RE = re.compile(OPEN + r"([A-Z]+)_(\d+)" + CLOSE)
_KIND_NAMES = "|".join(k.value for k in Kind)
# Damaged sentinel forms an LLM might produce: bare, ASCII-bracketed, spaced.
_DAMAGED_RE = re.compile(
    r"(?:(?:\[\[|\[|<|⟦)\s*)?\b(" + _KIND_NAMES + r")[ _](\d+)\b(?:\s*(?:\]\]|\]|>|⟧))?"
)


class IntegrityError(ValueError):
    def __init__(self, report: IntegrityReport) -> None:
        super().__init__(report.summary())
        self.report = report


@dataclass
class Span:
    start: int
    end: int
    kind: Kind
    text: str
    sentinel: str = ""


@dataclass
class MaskConfig:
    citations: bool = True
    references: bool = True
    math: bool = True
    urls: bool = True
    quantities: bool = True
    terms: bool = True
    latex_commands: bool = True
    glossary: tuple[str, ...] = ()


@dataclass
class IntegrityReport:
    missing: list[str] = field(default_factory=list)
    duplicated: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    malformed: list[str] = field(default_factory=list)
    soft_terms_missing: list[str] = field(default_factory=list)
    citation_order_changed: bool = False
    brace_imbalance: int = 0

    @property
    def ok(self) -> bool:
        """Hard guarantees only; soft-term loss and reordering are warnings."""
        return not (self.missing or self.duplicated or self.unknown or self.malformed or self.brace_imbalance)

    @property
    def warnings(self) -> list[str]:
        w = []
        if self.citation_order_changed:
            w.append("citation order changed")
        if self.soft_terms_missing:
            w.append(f"terms not found after edit: {', '.join(self.soft_terms_missing)}")
        return w

    def summary(self) -> str:
        if self.ok and not self.warnings:
            return "integrity ok"
        parts = []
        for name in ("missing", "duplicated", "unknown", "malformed"):
            vals = getattr(self, name)
            if vals:
                parts.append(f"{name}: {', '.join(vals)}")
        if self.brace_imbalance:
            parts.append(f"unbalanced LaTeX braces ({self.brace_imbalance:+d})")
        parts += self.warnings
        return "; ".join(parts)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "missing": self.missing, "duplicated": self.duplicated,
                "unknown": self.unknown, "malformed": self.malformed,
                "soft_terms_missing": self.soft_terms_missing,
                "citation_order_changed": self.citation_order_changed,
                "brace_imbalance": self.brace_imbalance, "warnings": self.warnings}


def _brace_balance(text: str) -> int:
    t = re.sub(r"\\[{}]", "", text)
    return t.count("{") - t.count("}")


@dataclass
class MaskResult:
    original: str
    masked: str
    spans: list[Span]
    fmt: str = "text"
    soft_terms: list[str] = field(default_factory=list)

    @property
    def symbols(self) -> dict[str, str]:
        return {s.sentinel: s.text for s in self.spans}

    def counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for s in self.spans:
            c[s.kind.value] = c.get(s.kind.value, 0) + 1
        return c

    # ------------------------------------------------------------ verification
    def repair(self, edited: str) -> tuple[str, list[str]]:
        """Normalise damaged sentinels (``[CITE_1]``, ``CITE 1``) back to canonical form,
        but only for sentinels that are otherwise absent. Returns (text, repairs)."""
        known = self.symbols
        present = set(SENTINEL_RE.findall(edited))
        present_s = {f"{OPEN}{k}_{n}{CLOSE}" for k, n in present}
        repairs: list[str] = []

        def fix(m: re.Match) -> str:
            canon = f"{OPEN}{m.group(1)}_{m.group(2)}{CLOSE}"
            if m.group(0) == canon or canon not in known or canon in present_s:
                return m.group(0)
            repairs.append(f"{m.group(0)!r} -> {canon}")
            present_s.add(canon)
            return canon

        return _DAMAGED_RE.sub(fix, edited), repairs

    def verify(self, edited: str) -> IntegrityReport:
        r = IntegrityReport()
        expected = [s.sentinel for s in self.spans]
        found = [f"{OPEN}{k}_{n}{CLOSE}" for k, n in SENTINEL_RE.findall(edited)]
        exp_set = set(expected)
        for s in expected:
            n = found.count(s)
            if n == 0:
                r.missing.append(s)
            elif n > 1:
                r.duplicated.append(s)
        r.unknown = sorted({s for s in found if s not in exp_set})
        residue = SENTINEL_RE.sub("", edited)
        if OPEN in residue or CLOSE in residue:
            r.malformed.append("stray sentinel bracket")
        for m in _DAMAGED_RE.finditer(residue):
            canon = f"{OPEN}{m.group(1)}_{m.group(2)}{CLOSE}"
            if canon in exp_set:
                r.malformed.append(m.group(0).strip())
        cites_before = [s.sentinel for s in self.spans if s.kind == Kind.CITE]
        cites_after = [s for s in found if s in set(cites_before)]
        r.citation_order_changed = cites_after != cites_before and not r.missing
        if self.fmt == "latex":
            r.brace_imbalance = _brace_balance(residue) - _brace_balance(SENTINEL_RE.sub("", self.masked))
        if self.soft_terms:
            restored_lower = self._substitute(edited).lower()
            r.soft_terms_missing = [t for t in self.soft_terms if t.lower() not in restored_lower]
        return r

    # --------------------------------------------------------------- restoring
    def _substitute(self, text: str, surface=None) -> str:
        sym = self.symbols
        kinds = {s.sentinel: s for s in self.spans}

        def sub(m: re.Match) -> str:
            key = m.group(0)
            if key not in sym:
                return key
            return surface(kinds[key]) if surface else sym[key]

        return SENTINEL_RE.sub(sub, text)

    def unmask(self, edited: str | None = None, *, strict: bool = True) -> str:
        text = self.masked if edited is None else edited
        if strict:
            report = self.verify(text)
            if not report.ok:
                raise IntegrityError(report)
        return self._substitute(text)

    def render(self, text: str | None = None) -> str:
        """Plain-prose rendering for statistical scoring (no LaTeX markup)."""
        out = self._substitute(self.masked if text is None else text, _surface)
        if self.fmt == "latex":
            out = re.sub(r"(?<!\\)~", " ", out).replace("``", "\"").replace("''", "\"")
        return _clean_spacing(out)


def _surface(span: Span) -> str:
    # a LaTeX non-breaking space glued to the span ("grew~\\cite{x}") still separates words
    lead = " " if span.text.startswith("~") else ""
    return lead + _surface_core(span)


def _surface_core(span: Span) -> str:
    if span.kind in RENDER_SURFACE:
        return RENDER_SURFACE[span.kind]
    t = span.text
    if span.kind == Kind.MATH:
        return "x"
    if span.kind == Kind.REF and "\\" in t:
        return "Figure 1"
    if span.kind == Kind.CMD:
        return "" if t.lstrip().startswith("%") else _latex_to_text(t)
    return t


def _latex_to_text(snippet: str) -> str:
    try:
        from pylatexenc.latex2text import LatexNodes2Text

        return LatexNodes2Text(math_mode="text").latex_to_text(snippet).strip()
    except Exception:  # pragma: no cover - defensive
        return re.sub(r"\\[a-zA-Z]+\*?|[{}]", "", snippet)


def _clean_spacing(text: str) -> str:
    text = re.sub(r"[ \t]+([,.;:!?)])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


class Masker:
    def __init__(self, config: MaskConfig | None = None, soft_terms: SoftTermDetector | None = None) -> None:
        self.config = config or MaskConfig()
        self.soft_terms = soft_terms

    def find_spans(self, text: str, fmt: str = "text", protected: list[dict] | None = None) -> list[Span]:
        c = self.config
        raw: list[patterns.RawSpan] = patterns.literal_sentinel_char_spans(text, (OPEN, CLOSE))
        for p in protected or []:
            kind = PROTECTED_REASON_KIND.get(p.get("reason", ""), Kind.PROT)
            raw.append((p["start"], p["end"], kind))
        if fmt == "latex" and c.latex_commands:
            raw += [s for s in patterns.latex_spans(text)
                    if (c.math or s[2] != Kind.MATH) and (c.citations or s[2] != Kind.CITE)
                    and (c.references or s[2] != Kind.REF)]
        if c.citations:
            raw += patterns.citation_spans(text)
        if c.references:
            raw += patterns.reference_spans(text)
        if c.urls:
            raw += patterns.url_spans(text)
        if c.quantities:
            raw += patterns.quantity_spans(text)
        if c.terms:
            raw += hard_term_spans(text, c.glossary)
        return self._resolve(text, raw)

    @staticmethod
    def _resolve(text: str, raw: list[patterns.RawSpan]) -> list[Span]:
        """Greedy overlap resolution: higher priority first, then longer spans."""
        raw = [(a, b, k) for a, b, k in raw if 0 <= a < b <= len(text)]
        raw.sort(key=lambda s: (-PRIORITY[s[2]], -(s[1] - s[0]), s[0]))
        starts: list[int] = []
        accepted: list[tuple[int, int, Kind]] = []
        for a, b, k in raw:
            i = bisect_left(starts, a)
            if i > 0 and accepted[i - 1][1] > a:
                continue
            if i < len(accepted) and accepted[i][0] < b:
                continue
            starts.insert(i, a)
            accepted.insert(i, (a, b, k))
        return [Span(a, b, k, text[a:b]) for a, b, k in accepted]

    def mask(self, text: str, fmt: str = "text", protected: list[dict] | None = None) -> MaskResult:
        spans = self.find_spans(text, fmt, protected)
        counters: dict[Kind, int] = {}
        out: list[str] = []
        cursor = 0
        for s in spans:
            counters[s.kind] = counters.get(s.kind, 0) + 1
            s.sentinel = f"{OPEN}{s.kind.value}_{counters[s.kind]}{CLOSE}"
            out.append(text[cursor : s.start])
            out.append(s.sentinel)
            cursor = s.end
        out.append(text[cursor:])
        masked = "".join(out)
        soft = []
        if self.soft_terms is not None:
            soft = self.soft_terms(SENTINEL_RE.sub(" ", masked))
        return MaskResult(original=text, masked=masked, spans=spans, fmt=fmt, soft_terms=soft)

    def mask_node(self, node, fmt: str) -> MaskResult:
        return self.mask(node.text, fmt=fmt, protected=node.meta.get("protected"))
