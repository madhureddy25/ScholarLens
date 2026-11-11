"""Quill: citation-safe language editing of an author's own manuscript.

Per prose node:

    mask → prompt model → extract → repair sentinels → verify integrity
         → unmask → semantic-similarity gate → accept
                         ↑ on failure: retry with specific feedback (max_attempts)

An edit that cannot pass every gate is discarded and the author's original
paragraph is kept. Nothing outside prose nodes is ever sent to the model or
modified. Quill's objective is the author's clarity and fidelity only; it takes
no detector signal as input.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from scholarlens.core.document import Document
from scholarlens.core.text import words
from scholarlens.edit import prompts
from scholarlens.edit.diff import change_stats, diff
from scholarlens.llm.base import ChatModel, ModelConfigError, ModelError
from scholarlens.masking import SENTINEL_RE, Masker, MaskResult


class Status(StrEnum):
    ACCEPTED = "accepted"
    UNCHANGED = "unchanged"
    REJECTED = "rejected"  # every attempt failed a gate; original kept
    SKIPPED = "skipped"  # not sent to the model (locked, too short, filtered out)
    REFUSED = "refused"  # model declined
    ERROR = "error"  # backend error after retries


@dataclass
class EditConfig:
    mode: str = "clarity"
    max_attempts: int = 3
    min_similarity: float | None = 0.94
    max_tokens: int | None = None
    min_words: int = 6
    sections: tuple[str, ...] | None = None
    """Only edit prose in these sections (case-insensitive); None = all."""
    node_ids: tuple[str, ...] | None = None
    """Only edit these node ids; None = all prose nodes."""


@dataclass
class Attempt:
    index: int
    output: str
    repairs: list[str]
    integrity: dict
    similarity: float | None
    problems: list[str]
    seconds: float


@dataclass
class NodeEdit:
    node_id: str
    section: str | None
    status: Status
    original: str
    edited: str | None = None
    similarity: float | None = None
    stats: dict = field(default_factory=dict)
    attempts: list[Attempt] = field(default_factory=list)
    served_by: str | None = None
    reason: str | None = None
    warnings: list[str] = field(default_factory=list)
    mask: MaskResult | None = field(default=None, repr=False)
    edited_masked: str | None = field(default=None, repr=False)

    @property
    def final_text(self) -> str:
        return self.edited if self.status == Status.ACCEPTED and self.edited is not None else self.original

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k not in ("mask", "edited_masked")}
        d["status"] = self.status.value
        return d


@dataclass
class EditSession:
    document: Document
    edits: list[NodeEdit]
    model: dict
    config: EditConfig
    similarity_model: str | None
    started_at: str
    finished_at: str
    glossary: tuple[str, ...] = ()

    def replacements(self) -> dict[str, str]:
        return {e.node_id: e.edited for e in self.edits if e.status == Status.ACCEPTED and e.edited is not None}

    def output_text(self) -> str:
        if self.document.source is not None:
            return self.document.reconstruct(self.replacements())
        return self.document.render_markdown(self.replacements())

    def counts(self) -> dict[str, int]:
        c = {s.value: 0 for s in Status}
        for e in self.edits:
            c[e.status.value] += 1
        return c


class Quill:
    def __init__(self, model: ChatModel, *, masker: Masker | None = None,
                 similarity: Callable[[str, str], float] | None = None,
                 config: EditConfig | None = None) -> None:
        self.model = model
        self.masker = masker or Masker()
        self.similarity = similarity
        self.config = config or EditConfig()
        if self.config.mode not in prompts.MODES:
            raise ValueError(f"unknown mode {self.config.mode!r}")

    # ---------------------------------------------------------------- single
    def edit_text(self, text: str, fmt: str = "text", *, protected: list[dict] | None = None,
                  node_id: str = "text", section: str | None = None) -> NodeEdit:
        mask = self.masker.mask(text, fmt=fmt, protected=protected)
        result = NodeEdit(node_id, section, Status.SKIPPED, text, mask=mask)
        if len([w for w in words(SENTINEL_RE.sub(" ", mask.masked))]) < self.config.min_words:
            result.reason = f"fewer than {self.config.min_words} editable words"
            return result

        system = prompts.system_prompt(self.config.mode, fmt)
        messages = [{"role": "user", "content": prompts.user_prompt(mask.masked)}]
        original_plain = mask.render()

        for i in range(1, self.config.max_attempts + 1):
            t0 = time.monotonic()
            try:
                gen = self.model.generate(system, messages, max_tokens=self.config.max_tokens)
            except ModelConfigError:
                raise  # misconfiguration aborts the whole session
            except ModelError as e:
                result.status, result.reason = Status.ERROR, str(e)
                return result
            result.served_by = f"{gen.provider}:{gen.model}"
            if gen.refused:
                result.status, result.reason = Status.REFUSED, f"model declined (stop_reason={gen.stop_reason})"
                return result

            candidate = prompts.extract_paragraph(gen.text)
            candidate, repairs = mask.repair(candidate)
            report = mask.verify(candidate)
            problems: list[str] = []
            if gen.stop_reason in ("max_tokens", "length"):
                problems.append("the output was cut off by the length limit")
            if report.missing:
                problems.append(f"placeholders missing: {', '.join(report.missing)}")
            if report.duplicated:
                problems.append(f"placeholders used more than once: {', '.join(report.duplicated)}")
            if report.unknown:
                problems.append(f"placeholders that do not exist in the original: {', '.join(report.unknown)}")
            if report.malformed:
                problems.append("some placeholders were altered; copy them exactly as ⟦KIND_n⟧")
            if report.brace_imbalance:
                problems.append("LaTeX braces are unbalanced")
            if not candidate.strip():
                problems.append("the output was empty")
            if "<paragraph" in candidate or "</paragraph" in candidate:
                problems.append("the output still contains <paragraph> tags; return only the text")

            sim = None
            if not problems and self.similarity is not None and self.config.min_similarity is not None:
                sim = round(float(self.similarity(original_plain, mask.render(candidate))), 4)
                if sim < self.config.min_similarity:
                    problems.append(
                        f"the meaning moved too far from the original (similarity {sim:.3f} < "
                        f"{self.config.min_similarity}); stay closer to the author's wording")

            result.attempts.append(Attempt(i, gen.text, repairs, report.to_dict(), sim, problems,
                                           round(time.monotonic() - t0, 3)))
            if not problems:
                edited = mask.unmask(candidate)
                result.similarity, result.warnings = sim, report.warnings
                result.edited_masked = candidate
                if edited.strip() == text.strip():
                    result.status, result.edited = Status.UNCHANGED, None
                else:
                    result.status, result.edited = Status.ACCEPTED, edited
                    result.stats = change_stats(diff(mask.masked, candidate))
                return result
            messages += [{"role": "assistant", "content": gen.text},
                         {"role": "user", "content": prompts.retry_prompt(problems)}]

        result.status = Status.REJECTED
        result.reason = "; ".join(result.attempts[-1].problems) if result.attempts else "no attempts"
        return result

    # -------------------------------------------------------------- document
    def edit_document(self, doc: Document, *, on_progress: Callable[[int, int, NodeEdit], None] | None = None
                      ) -> EditSession:
        started = datetime.now(UTC).isoformat(timespec="seconds")
        cfg = self.config
        targets = list(doc.prose())
        edits: list[NodeEdit] = []
        for k, node in enumerate(targets, 1):
            reason = None
            if node.meta.get("locked"):
                reason = f"locked ({node.meta['locked']})"
            elif cfg.node_ids is not None and node.id not in cfg.node_ids:
                reason = "not selected"
            elif cfg.sections is not None and (node.section or "").lower() not in {s.lower() for s in cfg.sections}:
                reason = "section not selected"
            if reason:
                e = NodeEdit(node.id, node.section, Status.SKIPPED, node.text, reason=reason)
            else:
                e = self.edit_text(node.text, doc.source_format, protected=node.meta.get("protected"),
                                   node_id=node.id, section=node.section)
            edits.append(e)
            if on_progress:
                on_progress(k, len(targets), e)
        sim_name = getattr(self.similarity, "name", None) if self.similarity else None
        return EditSession(doc, edits, self.model.describe(), cfg, sim_name, started,
                           datetime.now(UTC).isoformat(timespec="seconds"), tuple(self.masker.config.glossary))
