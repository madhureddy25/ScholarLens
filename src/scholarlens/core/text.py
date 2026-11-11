"""Sentence splitting, word tokenisation and windowing for academic prose.

Runs on plain or masked text. Masking first makes splitting markedly more
reliable, because citations ("Smith et al. (2020)") and cross-references
("Fig. 3") are already collapsed into single sentinel tokens.
"""

from __future__ import annotations

import re

_ABBREVIATIONS = {
    "e.g", "i.e", "al", "et al", "fig", "figs", "eq", "eqs", "vs", "cf", "approx", "resp", "dr",
    "prof", "mr", "mrs", "ms", "no", "nos", "vol", "vols", "pp", "p", "ca", "ref", "refs", "sec",
    "secs", "tab", "inc", "ltd", "co", "corp", "jr", "sr", "ph.d", "u.s", "u.k", "viz", "ibid",
    "chap", "ch", "eds", "ed", "est", "suppl", "max", "min", "avg", "std", "var",
    "dept", "univ", "lab", "e.g.,", "i.e.,", "s.t", "w.r.t", "a.k.a",
}
_BOUNDARY = re.compile(r"([.!?]+)([\"'”’)\]]*)\s+(?=[\"'“‘(\[]?[A-Z0-9⟦])")
_LAST_WORD = re.compile(r"([A-Za-z][A-Za-z.]*)$")
_WORD = re.compile(r"⟦[A-Z]+_\d+⟧|[A-Za-z0-9Ͱ-Ͽ]+(?:[-'’][A-Za-z0-9Ͱ-Ͽ]+)*")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _BOUNDARY.finditer(text):
        if m.group(1) == ".":
            last = _LAST_WORD.search(text[start : m.start(1)])
            if last and last.group(1).lower().rstrip(".") in _ABBREVIATIONS:
                continue
        end = m.end(2)
        a = start + (len(text[start:end]) - len(text[start:end].lstrip()))
        if end > a:
            spans.append((a, end))
        start = m.end()
    tail = text[start:]
    if tail.strip():
        a = start + (len(tail) - len(tail.lstrip()))
        spans.append((a, a + len(tail.strip())))
    return spans


def split_sentences(text: str) -> list[str]:
    return [text[a:b] for a, b in sentence_spans(text)]


def words(text: str) -> list[str]:
    return _WORD.findall(text)


def word_windows(sentences: list[str], size: int = 250, stride: int | None = None) -> list[tuple[int, int]]:
    """Sentence-aligned windows of roughly ``size`` words (as sentence index ranges).

    Commercial detectors score ~250-word overlapping segments; aligning windows
    to sentence boundaries keeps every window grammatical.
    """
    stride = stride or size // 2
    lengths = [len(words(s)) for s in sentences]
    if not sentences:
        return []
    windows: list[tuple[int, int]] = []
    i = 0
    while i < len(sentences):
        total, j = 0, i
        while j < len(sentences) and total < size:
            total += lengths[j]
            j += 1
        windows.append((i, j))
        if j >= len(sentences):
            break
        advanced, k = 0, i
        while k < j - 1 and advanced < stride:
            advanced += lengths[k]
            k += 1
        i = max(k, i + 1)
    return windows
