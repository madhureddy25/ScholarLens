"""Local feedback store: the author's accept/reject decisions on Quill edits.

Stored on the author's own machine (``$SCHOLARLENS_HOME``, default ``~/.scholarlens``).
Nothing leaves the machine unless the author runs ``scholarlens contribute``.
Only masked text is stored: citations, numbers, equations, references, URLs and
domain terms are already replaced by ⟦KIND_n⟧ placeholders.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

from scholarlens import __version__


def home() -> Path:
    return Path(os.environ.get("SCHOLARLENS_HOME", Path.home() / ".scholarlens"))


def feedback_path() -> Path:
    return home() / "feedback.jsonl"


def make_entry(*, masked_input: str, masked_output: str, decision: str, mode: str, fmt: str, model: dict,
               section: str | None, similarity: float | None) -> dict:
    if decision not in ("accept", "reject"):
        raise ValueError(decision)
    return {"id": uuid.uuid4().hex, "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "tool_version": __version__, "mode": mode, "format": fmt, "model": model, "section": section,
            "masked_input": masked_input, "masked_output": masked_output, "decision": decision,
            "similarity": similarity, "contributed": False}


def append(entries: list[dict]) -> Path:
    path = feedback_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return path


def load_all() -> list[dict]:
    path = feedback_path()
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mark_contributed(ids: set[str]) -> None:
    entries = load_all()
    for e in entries:
        if e["id"] in ids:
            e["contributed"] = True
    path = feedback_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries), encoding="utf-8")
    tmp.replace(path)
