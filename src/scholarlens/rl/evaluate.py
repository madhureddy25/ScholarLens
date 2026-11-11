"""Evaluate a Quill policy on a frozen set of paragraphs and decide whether a new
adapter may replace the current one. No detector is involved."""

from __future__ import annotations

import numpy as np

from scholarlens.edit.prompts import system_prompt, user_prompt
from scholarlens.llm.base import ModelError
from scholarlens.rl.rewards import QuillReward


def evaluate_policy(model, examples: list[dict], reward: QuillReward, *, max_tokens: int = 400) -> dict:
    """Single attempt per paragraph (no retries, no repair): measures the raw policy."""
    rows = []
    for ex in examples:
        try:
            gen = model.generate(system_prompt(ex["mode"], "text"), [{"role": "user", "content": user_prompt(ex["masked"])}],
                                 max_tokens=max_tokens)
            text = gen.text
        except ModelError:
            text = ""
        r = reward.score(ex["masked"], text, mode=ex["mode"])
        rows.append(r)
    sims = [r.similarity for r in rows if r.similarity is not None]
    ok = [r for r in rows if r.integrity_ok]
    return {
        "n": len(rows),
        "integrity_rate": float(np.mean([r.integrity_ok for r in rows])) if rows else 0.0,
        "format_rate": float(np.mean([r.format_ok for r in rows])) if rows else 0.0,
        "mean_reward": float(np.mean([r.total for r in rows])) if rows else 0.0,
        "mean_similarity": float(np.mean(sims)) if sims else None,
        "mean_quality": float(np.mean([r.quality for r in ok])) if ok else 0.0,
        "mean_change": float(np.mean([r.change_ratio for r in ok])) if ok else 0.0,
    }


def gate(new: dict, current: dict, *, tol: float = 0.01) -> tuple[bool, list[str]]:
    """Accept only if integrity does not drop, reward improves, and meaning is no less preserved."""
    reasons = []
    if new["integrity_rate"] < current["integrity_rate"] - 1e-9:
        reasons.append(f"integrity {new['integrity_rate']:.3f} < current {current['integrity_rate']:.3f}")
    if new["mean_reward"] <= current["mean_reward"]:
        reasons.append(f"mean reward {new['mean_reward']:.3f} ≤ current {current['mean_reward']:.3f}")
    if new.get("mean_similarity") is not None and current.get("mean_similarity") is not None \
            and new["mean_similarity"] < current["mean_similarity"] - tol:
        reasons.append(f"similarity {new['mean_similarity']:.3f} < current {current['mean_similarity']:.3f} - {tol}")
    return not reasons, reasons
