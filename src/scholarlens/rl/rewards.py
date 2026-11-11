"""Verifiable rewards for training Quill with reinforcement learning.

Quill is rewarded for being a good, trustworthy copy editor and for nothing else:

    integrity   every placeholder kept exactly once, LaTeX balanced  (hard gate)
    fidelity    meaning preserved: semantic similarity ≥ threshold
    quality     grammar/spelling errors actually fixed (LanguageTool), and/or an
                LLM judge's rating of clarity and faithfulness
    economy     no more change than the mode calls for (keeps the author's voice)
    format      returns only the paragraph (no chat wrappers)

DESIGN RULE: this package never uses an AI-text detector, perplexity/burstiness
targets, or any other detection statistic as a reward. Quill-edited human text
is the honest-editing baseline in Bench's fairness study; optimising Quill
against a detector would make that measurement circular and turn an editor into
a detector-evasion tool. ``tests/test_rl.py`` fails if this package imports
``scholarlens.detect``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol
from urllib import parse, request

from scholarlens.edit.diff import change_stats, diff
from scholarlens.edit.prompts import extract_paragraph
from scholarlens.masking import SENTINEL_RE, Kind, MaskResult, Span

# Neutral stand-ins used when only masked text is available (contributions):
NEUTRAL_SURFACE = {Kind.CITE: "[1]", Kind.REF: "Figure 1", Kind.MATH: "x", Kind.NUM: "3", Kind.TERM: "X",
                   Kind.URL: "URL", Kind.CMD: "", Kind.PROT: "word"}

# How much change each mode legitimately calls for (fraction of words touched).
MODE_ALLOWANCE = {"proofread": 0.06, "clarity": 0.25, "concise": 0.35}


def mask_from_masked(masked: str) -> MaskResult:
    """A MaskResult for text that is already masked (originals unknown): placeholders map to
    neutral surfaces, so integrity checks and similarity work without the private originals."""
    spans, parts, pos, cursor = [], [], 0, 0
    for m in SENTINEL_RE.finditer(masked):
        parts.append(masked[cursor:m.start()])
        pos += m.start() - cursor
        surface = NEUTRAL_SURFACE.get(Kind(m.group(1)), "X") if m.group(1) in Kind.__members__ else "X"
        spans.append(Span(pos, pos + len(surface), Kind(m.group(1)) if m.group(1) in Kind.__members__ else Kind.TERM,
                          surface, m.group(0)))
        parts.append(surface)
        pos += len(surface)
        cursor = m.end()
    parts.append(masked[cursor:])
    return MaskResult(original="".join(parts), masked=masked, spans=spans)


class GrammarChecker(Protocol):
    def count(self, text: str) -> int: ...


class LanguageToolChecker:
    """Counts grammar/spelling/punctuation issues with a LanguageTool server
    (``docker run -p 8010:8010 erikvl87/languagetool``). Style and typography hints are ignored."""

    CATEGORIES = {"TYPOS", "GRAMMAR", "PUNCTUATION", "CASING", "CONFUSED_WORDS", "COLLOCATIONS", "SEMANTICS"}

    def __init__(self, url: str = "http://localhost:8010", language: str = "en-US", timeout: float = 20.0) -> None:
        self.url, self.language, self.timeout = url.rstrip("/"), language, timeout
        self._cache: dict[str, int] = {}

    def count(self, text: str) -> int:
        if text not in self._cache:
            data = parse.urlencode({"text": text, "language": self.language}).encode()
            with request.urlopen(request.Request(f"{self.url}/v2/check", data=data), timeout=self.timeout) as r:  # noqa: S310
                matches = json.loads(r.read())["matches"]
            self._cache[text] = sum(1 for m in matches if m.get("rule", {}).get("category", {}).get("id") in self.CATEGORIES)
        return self._cache[text]


class LLMJudge:
    """Rates an edit's clarity and faithfulness with any chat model (via the switchable model layer).
    The judge is asked about language quality and meaning only."""

    SYSTEM = ("You are an expert academic copy editor reviewing a proposed revision of a paragraph. Placeholders "
              "like ⟦CITE_1⟧ stand for protected content. Judge only language quality and faithfulness.")
    PROMPT = ("Original:\n{a}\n\nRevision:\n{b}\n\nScore the revision from 1 to 5, where 5 means clearer and more "
              "correct than the original with exactly the same meaning and claims, 3 means no better or worse, and "
              "1 means worse or meaning changed. Reply with the number only.")

    def __init__(self, model) -> None:
        self.model = model

    def score(self, original: str, revision: str) -> float:
        gen = self.model.generate(self.SYSTEM, [{"role": "user", "content": self.PROMPT.format(a=original, b=revision)}],
                                  max_tokens=16)
        m = re.search(r"[1-5]", gen.text or "")
        return (int(m.group(0)) - 3) / 2 if m else 0.0  # → [-1, 1]


@dataclass
class RewardWeights:
    fidelity: float = 0.3
    quality: float = 0.35
    economy: float = 0.35
    format_penalty: float = 0.3
    integrity_fail: float = -1.0


@dataclass
class RewardBreakdown:
    total: float
    integrity_ok: bool
    format_ok: bool
    similarity: float | None = None
    quality: float = 0.0
    economy: float = 0.0
    change_ratio: float = 0.0
    errors_before: int | None = None
    errors_after: int | None = None
    notes: list[str] = field(default_factory=list)


class QuillReward:
    def __init__(self, *, similarity=None, grammar: GrammarChecker | None = None, judge: LLMJudge | None = None,
                 weights: RewardWeights | None = None, min_similarity: float = 0.94, drift_floor: float = 0.80) -> None:
        self.similarity, self.grammar, self.judge = similarity, grammar, judge
        self.w = weights or RewardWeights()
        self.min_similarity, self.drift_floor = min_similarity, drift_floor

    def score(self, masked_input: str, output: str, *, mode: str = "proofread", mask: MaskResult | None = None
              ) -> RewardBreakdown:
        mask = mask or mask_from_masked(masked_input)
        candidate = extract_paragraph(output)
        format_ok = candidate.strip() == output.strip()
        fmt_pen = 0.0 if format_ok else self.w.format_penalty
        report = mask.verify(candidate)
        if not candidate.strip() or not report.ok:
            return RewardBreakdown(self.w.integrity_fail - fmt_pen, False, format_ok, notes=[report.summary()])

        before, after = mask.render(), mask.render(candidate)
        r = RewardBreakdown(0.0, True, format_ok)
        # fidelity: full credit at/above the threshold, falling to -1 at the drift floor
        if self.similarity is not None:
            r.similarity = float(self.similarity(before, after))
            gap = (self.min_similarity - r.similarity) / max(self.min_similarity - self.drift_floor, 1e-6)
            fidelity = 1.0 if r.similarity >= self.min_similarity else max(-1.0, -gap)
        else:
            fidelity = 1.0
        # quality: errors fixed (relative), plus optional judge
        q_parts = []
        if self.grammar is not None:
            r.errors_before, r.errors_after = self.grammar.count(before), self.grammar.count(after)
            if r.errors_before or r.errors_after:
                q_parts.append(max(-1.0, min(1.0, (r.errors_before - r.errors_after) / max(r.errors_before, 1))))
            else:
                q_parts.append(0.0)
        if self.judge is not None and candidate.strip() != masked_input.strip():
            q_parts.append(self.judge.score(masked_input, candidate))
        r.quality = sum(q_parts) / len(q_parts) if q_parts else 0.0
        # economy: penalise change beyond what the mode (and the errors present) call for
        r.change_ratio = change_stats(diff(mask.masked, candidate))["change_ratio"]
        allowance = MODE_ALLOWANCE.get(mode, 0.25) + 0.03 * (r.errors_before or 0)
        # steep enough that rewriting a whole paragraph (which trivially "fixes" every error)
        # always scores below leaving it alone — otherwise RL learns to rewrite everything
        r.economy = -min(2.0, 3.0 * max(0.0, r.change_ratio - allowance))
        r.total = self.w.fidelity * fidelity + self.w.quality * r.quality + self.w.economy * r.economy - fmt_pen
        return r
