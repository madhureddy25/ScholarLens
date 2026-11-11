"""AI-written counterparts for human records.

Each human abstract gets a machine-written abstract for the *same paper*
(same title, same field, similar length), so detectors are compared on matched
topics rather than learning topic shortcuts. Several generator models are used
so the detector does not learn one model's fingerprint.
"""

from __future__ import annotations

import random

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
    text = " ".join(gen.text.split())
    if gen.refused or not text:
        return None
    if style == "title_first_sentence":
        text = f"{first} {text}"
    return Record(id=stable_id(human.id, "ai", model.name, style), paper_id=human.paper_id, text=text,
                  label=LABEL_AI, source=human.source, domain=human.domain, year=None, title=human.title,
                  generator=gen.model and f"{gen.provider}:{gen.model}", parent_id=human.id, split=human.split,
                  meta={"prompt_style": style})


def generate_all(humans: list[Record], models: list[ChatModel], *, styles: tuple[str, ...] = ("title",),
                 seed: int = 13, max_tokens: int = 2048, on_progress=None) -> list[Record]:
    """One AI counterpart per human record, round-robin over generators and prompt styles."""
    rng = random.Random(seed)
    out: list[Record] = []
    for i, h in enumerate(humans):
        model = models[i % len(models)]
        rec = generate_counterpart(h, model, style=rng.choice(styles), max_tokens=max_tokens)
        if rec is not None:
            out.append(rec)
        if on_progress:
            on_progress(i + 1, len(humans), rec)
    return out
