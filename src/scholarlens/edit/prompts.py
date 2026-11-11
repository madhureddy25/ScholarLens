"""Prompts for Quill. Instructions are about language quality and fidelity only."""

from __future__ import annotations

import re

SYSTEM = """\
You are a copy editor for academic manuscripts. An author has asked for help with the \
language of their own draft. You receive one paragraph at a time and return the revised paragraph.

The paragraph contains placeholders such as ⟦CITE_1⟧, ⟦NUM_2⟧, ⟦MATH_1⟧, ⟦REF_1⟧ and ⟦TERM_3⟧. \
Each one stands for content that must reach the reader exactly as the author wrote it: a citation, \
a number or statistic, an equation, a cross-reference, a technical term, or markup. Copy every \
placeholder exactly, use each one exactly once, and never create new ones. You may move a \
placeholder within its sentence when the grammar requires it, but keep it attached to the claim it \
supports.

Edit the language, not the science. Do not add, remove, strengthen or soften claims, hedges, results \
or citations. Keep the author's technical terminology and voice. If a sentence is already clear and \
correct, leave it as it is; if the whole paragraph is fine, return it unchanged.

{mode_instructions}{format_instructions}

Return only the revised paragraph, with no preamble, notes, quotation marks or code fences."""

MODES = {
    "proofread": "Mode: proofread. Fix spelling, grammar, punctuation, agreement and article usage. "
                 "Make the smallest change that fixes each problem and do not rephrase sentences that "
                 "are already correct.",
    "clarity": "Mode: clarity. Fix errors and improve readability: untangle convoluted sentences, make "
               "vague wording precise and smooth the flow between sentences, while keeping the meaning, "
               "claims and level of certainty identical.",
    "concise": "Mode: concise. Tighten the prose by removing redundancy, filler and needless words, "
               "without dropping any information, qualifier or claim.",
}

FORMATS = {
    "latex": "\n\nThe paragraph is LaTeX source, so the result must remain valid LaTeX. Escape any %, & or _ "
             "you introduce, and keep LaTeX spacing and dash conventions (~, --, ---) as the author uses them.",
    "markdown": "\n\nThe paragraph is Markdown; keep its inline markup intact.",
}


def system_prompt(mode: str, fmt: str) -> str:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {', '.join(MODES)}")
    return SYSTEM.format(mode_instructions=MODES[mode], format_instructions=FORMATS.get(fmt, ""))


def user_prompt(masked: str) -> str:
    return f"<paragraph>\n{masked}\n</paragraph>"


def retry_prompt(problems: list[str]) -> str:
    bullet = "\n".join(f"- {p}" for p in problems)
    return ("Your revision could not be accepted:\n" + bullet +
            "\nRevise the original paragraph again, following all of the rules, and return only the paragraph.")


_FENCE = re.compile(r"^```[a-zA-Z]*\n(.*?)\n```$", re.S)
_TAGGED = re.compile(r"<paragraph>\s*(.*?)\s*</paragraph>", re.S)
_PREAMBLE = re.compile(r"^(?:here is|here's|revised|edited|corrected)[^\n:]{0,60}:\s*\n+", re.I)


def extract_paragraph(output: str) -> str:
    """Strip wrappers models sometimes add despite instructions."""
    text = output.strip()
    m = _TAGGED.search(text)
    if m:
        text = m.group(1)
    text = re.sub(r"^\s*<paragraph>\s*", "", text)
    text = re.sub(r"\s*</paragraph>\s*$", "", text)
    m = _FENCE.match(text)
    if m:
        text = m.group(1)
    text = _PREAMBLE.sub("", text).strip()
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'“”":
        text = text[1:-1].strip()
    return text
