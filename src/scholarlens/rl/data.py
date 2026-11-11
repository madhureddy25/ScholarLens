"""Training data for Quill-RL.

* GRPO prompts: public, pre-2022 human paragraphs (Bench human records) and the
  masked inputs of community contributions. No labels are needed: rewards are
  computed automatically.
* DPO pairs: author decisions from reviews. An accepted edit is preferred over
  leaving the paragraph unchanged; for a rejected edit, unchanged is preferred.
"""

from __future__ import annotations

from scholarlens.bench.schema import LABEL_HUMAN, Record
from scholarlens.edit.prompts import system_prompt, user_prompt
from scholarlens.masking import SENTINEL_RE, Masker


def _prompt(masked: str, mode: str, fmt: str) -> list[dict]:
    return [{"role": "system", "content": system_prompt(mode, fmt)}, {"role": "user", "content": user_prompt(masked)}]


def prompts_from_records(records: list[Record], *, mode: str, splits: tuple[str, ...] = ("train",),
                         limit: int | None = None) -> list[dict]:
    masker, out = Masker(), []
    for r in records:
        if r.label != LABEL_HUMAN or r.transform != "none" or r.split not in splits:
            continue
        masked = masker.mask(r.text).masked
        out.append({"prompt": _prompt(masked, mode, "text"), "masked": masked, "mode": mode, "source": r.id})
        if limit and len(out) >= limit:
            break
    return out


def prompts_from_contributions(entries: list[dict], *, mode: str | None = None) -> list[dict]:
    out, seen = [], set()
    for e in entries:
        m = mode or e["mode"]
        key = (e["masked_input"], m)
        if key in seen:
            continue
        seen.add(key)
        out.append({"prompt": _prompt(e["masked_input"], m, e.get("format") or "text"), "masked": e["masked_input"],
                    "mode": m, "source": e["id"]})
    return out


def dpo_pairs(entries: list[dict]) -> list[dict]:
    pairs = []
    for e in entries:
        if e["masked_input"].strip() == e["masked_output"].strip():
            continue
        if e["decision"] == "accept" and sorted(SENTINEL_RE.findall(e["masked_input"])) != sorted(
                SENTINEL_RE.findall(e["masked_output"])):
            continue
        edit = [{"role": "assistant", "content": e["masked_output"]}]
        keep = [{"role": "assistant", "content": e["masked_input"]}]
        chosen, rejected = (edit, keep) if e["decision"] == "accept" else (keep, edit)
        pairs.append({"prompt": _prompt(e["masked_input"], e["mode"], e.get("format") or "text"),
                      "chosen": chosen, "rejected": rejected})
    return pairs
