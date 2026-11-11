"""AI-written counterparts for human records.

Each human abstract gets a machine-written abstract for the *same paper*
(same title, same field, similar length), so detectors are compared on matched
topics rather than learning topic shortcuts. Several generator models are used
so the detector does not learn one model's fingerprint.
"""

from __future__ import annotations

import random
import re

from scholarlens.bench.schema import LABEL_AI, Record, stable_id
from scholarlens.core.text import split_sentences
from scholarlens.llm.base import ChatModel, ModelError

SYSTEM = "You are a researcher writing material for a scientific paper."

PROMPTS = {
    "title": ("Write the abstract of a {domain}paper titled \"{title}\". Write a single paragraph of about "
              "{n} words in the style of a published abstract. Return only the abstract."),
    "title_first_sentence": ("Write the abstract of a {domain}paper titled \"{title}\". It begins with this "
                             "sentence:\n\n{first}\n\nContinue from that sentence (do not repeat it) to complete a "
                             "single-paragraph abstract of about {n} words in total. Return only the continuation."),
}


_HEADER = re.compile(r"^(?:#+\s*)?(?:\*\*|__)?\s*(?:abstract|summary|paraphrase|title)\s*(?:\*\*|__)?\s*[:.\-–—]?\s*",
                     re.I)
_LEAD_IN = re.compile(r"^(?:here(?:'s| is)|sure[,!]?|certainly[,!]?)[^:\n]{0,80}:\s*", re.I)
_QUOTES = "\"'“”‘’"


def clean_generation(text: str) -> str:
    """Remove formatting artifacts that would let a detector key on format instead of writing:
    "**Abstract**" headers, "Here is…:" lead-ins, wrapping quotes, markdown emphasis, stray ellipses."""
    t = " ".join(text.split())
    for _ in range(3):
        before = t
        t = _LEAD_IN.sub("", t)
        t = _HEADER.sub("", t)
        t = t.strip().lstrip(".…").strip()
        if len(t) > 1 and t[0] in _QUOTES and t[-1] in _QUOTES:
            t = t[1:-1].strip()
        elif t and t[0] in _QUOTES and t.count(t[0]) == 1:
            t = t[1:].strip()
        if t == before:
            break
    t = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: m.group(1) or m.group(2), t)
    t = re.sub(r"\s*\.\.\.\s*", " ", t) if t.count("...") == 1 and "..." in t[:200] else t
    return " ".join(t.split())


def generate_counterpart(human: Record, model: ChatModel, *, style: str = "title",
                         max_tokens: int = 2048) -> Record | None:
    n = round(len(human.text.split()) / 10) * 10 or 150
    domain = f"{human.domain} " if human.domain else ""
    first = split_sentences(human.text)[0] if style == "title_first_sentence" else ""
    prompt = PROMPTS[style].format(domain=domain, title=human.title or "Untitled", n=n, first=first)
    try:
        gen = model.generate(SYSTEM, [{"role": "user", "content": prompt}], max_tokens=max_tokens)
    except ModelError:
        return None
    text = clean_generation(gen.text)
    if gen.refused or not text:
        return None
    if style == "title_first_sentence":
        text = f"{first} {text}"
    return Record(id=stable_id(human.id, "ai", model.name, style), paper_id=human.paper_id, text=text,
                  label=LABEL_AI, source=human.source, domain=human.domain, year=None, title=human.title,
                  generator=gen.model and f"{gen.provider}:{gen.model}", parent_id=human.id, split=human.split,
                  meta={"prompt_style": style})


def generate_all(humans: list[Record], models: list[ChatModel], *, styles: tuple[str, ...] = ("title",),
                 seed: int = 13, max_tokens: int = 2048, concurrency: int = 1, on_progress=None) -> list[Record]:
    """One AI counterpart per human record, round-robin over generators and prompt styles.

    ``concurrency`` > 1 sends requests in parallel (useful for API providers and vLLM/Ollama
    servers, which batch concurrent requests on the GPU). Assignment of generator and prompt
    style to each record is deterministic regardless of concurrency.
    """
    rng = random.Random(seed)
    jobs = [(h, models[i % len(models)], rng.choice(styles)) for i, h in enumerate(humans)]
    out: list[Record] = []
    if concurrency <= 1:
        for k, (h, m, st) in enumerate(jobs, 1):
            rec = generate_counterpart(h, m, style=st, max_tokens=max_tokens)
            if rec is not None:
                out.append(rec)
            if on_progress:
                on_progress(k, len(jobs), rec)
        return out
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(generate_counterpart, h, m, style=st, max_tokens=max_tokens) for h, m, st in jobs]
        for k, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            if rec is not None:
                out.append(rec)
            if on_progress:
                on_progress(k, len(jobs), rec)
    return out
