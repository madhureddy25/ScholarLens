"""Word-level diffs between an original and an edited paragraph.

Diffs run on the *masked* texts, so every protected item (citation, equation,
number, command) is a single atomic token that is either kept or moved, never
partially changed. Rendering unmasks tokens back to the author's original text.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from scholarlens.masking import MaskResult

_TOKEN = re.compile(r"⟦[A-Z]+_\d+⟧|\s+|\w+(?:[-'’]\w+)*|[^\w\s]")
_WORD = re.compile(r"\w")
_LATEX_UNSAFE = re.compile(r"[{}\\%$#&^_]|⟦")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text)


@dataclass
class Op:
    tag: str  # equal | delete | insert | replace
    old: list[str]
    new: list[str]


def diff(old_masked: str, new_masked: str) -> list[Op]:
    a, b = tokenize(old_masked), tokenize(new_masked)
    sm = SequenceMatcher(a=a, b=b, autojunk=False)
    return [Op(tag, a[i1:i2], b[j1:j2]) for tag, i1, i2, j1, j2 in sm.get_opcodes()]


def change_stats(ops: list[Op]) -> dict[str, float]:
    def words(toks: list[str]) -> int:
        return sum(1 for t in toks if _WORD.search(t))

    total = sum(words(o.old) for o in ops)
    removed = sum(words(o.old) for o in ops if o.tag in ("delete", "replace"))
    added = sum(words(o.new) for o in ops if o.tag in ("insert", "replace"))
    return {"words_original": total, "words_removed": removed, "words_added": added,
            "change_ratio": round((removed + added) / max(2 * total, 1), 4)}


def _plain(tokens: list[str], mask: MaskResult) -> str:
    return mask._substitute("".join(tokens))


def render_html(ops: list[Op], mask: MaskResult) -> str:
    out = []
    for o in ops:
        old = html.escape(_plain(o.old, mask))
        new = html.escape(_plain(o.new, mask))
        if o.tag == "equal":
            out.append(old)
        if o.tag in ("delete", "replace") and old.strip():
            out.append(f"<del>{old}</del>")
        elif o.tag in ("delete", "replace"):
            out.append(old)
        if o.tag in ("insert", "replace") and new.strip():
            out.append(f"<ins>{new}</ins>")
        elif o.tag in ("insert", "replace"):
            out.append(new)
    return "".join(out)


def render_latex_tracked(ops: list[Op], mask: MaskResult) -> str:
    r"""LaTeX with ``\DIFdel{}``/``\DIFadd{}`` marks (latexdiff-style macro names).

    Chunks containing protected items or LaTeX-special characters are emitted in
    their new form without marks, so the review copy always compiles.
    """
    out = []
    for o in ops:
        old, new = "".join(o.old), "".join(o.new)
        if o.tag == "equal":
            out.append(_plain(o.old, mask))
            continue
        if _LATEX_UNSAFE.search(old) or _LATEX_UNSAFE.search(new):
            out.append(_plain(o.new, mask))
            continue
        lead_o, core_o, trail_o = _split_ws(old)
        lead_n, core_n, trail_n = _split_ws(new)
        if not core_o and not core_n:
            out.append(new)
            continue
        marks = ([f"\\DIFdel{{{core_o}}}"] if core_o else []) + ([f"\\DIFadd{{{core_n}}}"] if core_n else [])
        out.append((lead_o or lead_n) + " ".join(marks) + (trail_n if core_n else trail_o))
    return "".join(out)


def _split_ws(s: str) -> tuple[str, str, str]:
    core = s.strip()
    if not core:
        return s, "", ""
    lead = s[: len(s) - len(s.lstrip())]
    return lead, core, s[len(lead) + len(core):]


LATEX_DIFF_PREAMBLE = r"""
% --- ScholarLens Quill tracked changes (review copy) ---
\usepackage[normalem]{ulem}
\usepackage{xcolor}
\providecommand{\DIFadd}[1]{{\color{blue}\uwave{#1}}}
\providecommand{\DIFdel}[1]{{\color{red}\sout{#1}}}
"""


def add_latex_diff_preamble(source: str) -> str:
    marker = "\\begin{document}"
    i = source.find(marker)
    if i == -1:
        return LATEX_DIFF_PREAMBLE.lstrip() + "\n" + source
    return source[:i] + LATEX_DIFF_PREAMBLE.lstrip() + source[i:]
