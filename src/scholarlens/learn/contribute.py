"""Opt-in sharing of review feedback with the public ScholarLens repository.

Contributions are *public*. Only masked text is ever shared (citations, numbers,
equations, references, URLs and domain terms are placeholders), personal
details are screened out, and model endpoints/credentials are never included.
The same validation runs again in CI before any contribution is used.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

from scholarlens.edit.prompts import MODES
from scholarlens.masking import SENTINEL_RE

CONTRIB_DIR = "contributions"
MAX_PER_FILE = 2000
_ALLOWED_SECTIONS = {
    "abstract", "introduction", "related work", "background", "method", "methods", "methodology",
    "materials and methods", "experiments", "results", "discussion", "conclusion", "conclusions", "limitations",
}
_PII = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "url": re.compile(r"https?://|www\."),
    "phone": re.compile(r"\+?\d[\d ()-]{8,}\d"),
    "orcid": re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b"),
    "id_number": re.compile(r"\b\d{6,}\b"),
}
_SENTINEL_LIKE = re.compile(r"⟦[^⟧]*⟧")


def entry_hash(e: dict) -> str:
    return hashlib.sha256((e["masked_input"] + "\x1f" + e["masked_output"]).encode()).hexdigest()[:20]


def sanitize(e: dict) -> dict:
    """Keep only shareable fields (no base URLs, file names, titles or timestamps finer than a day)."""
    model = e.get("model") or {}
    section = (e.get("section") or "").strip().lower()
    return {"id": entry_hash(e), "tool_version": e.get("tool_version"), "date": (e.get("created_at") or e.get("date") or "")[:10],
            "mode": e.get("mode"), "format": e.get("format"),
            "section": section if section in _ALLOWED_SECTIONS else None,
            "model": {"provider": model.get("provider"), "model": model.get("model")},
            "masked_input": e["masked_input"], "masked_output": e["masked_output"], "decision": e["decision"]}


def problems(e: dict) -> list[str]:
    out = []
    if e.get("decision") not in ("accept", "reject"):
        out.append("bad decision")
    if e.get("mode") not in MODES:
        out.append("bad mode")
    for key in ("masked_input", "masked_output"):
        text = e.get(key)
        if not isinstance(text, str) or not 20 <= len(text) <= 4000:
            out.append(f"{key} length")
            continue
        if any(not SENTINEL_RE.fullmatch(m) for m in _SENTINEL_LIKE.findall(text)):
            out.append(f"{key} has malformed placeholders")
        for name, rx in _PII.items():
            if rx.search(SENTINEL_RE.sub(" ", text)):
                out.append(f"{key} may contain personal data ({name})")
    if not out and e["decision"] == "accept":
        if sorted(SENTINEL_RE.findall(e["masked_input"])) != sorted(SENTINEL_RE.findall(e["masked_output"])):
            out.append("accepted edit does not preserve placeholders")
    return out


def prepare(entries: list[dict]) -> tuple[list[dict], list[tuple[str, list[str]]]]:
    ok, bad, seen = [], [], set()
    for raw in entries:
        e = sanitize(raw)
        why = problems(e)
        if e["id"] in seen:
            why.append("duplicate")
        seen.add(e["id"])
        (bad.append((raw.get("id", e["id"]), why)) if why else ok.append(e))
    return ok, bad


def write_contribution(entries: list[dict], repo_dir: str | Path) -> Path:
    if not entries:
        raise ValueError("nothing to contribute")
    if len(entries) > MAX_PER_FILE:
        raise ValueError(f"at most {MAX_PER_FILE} entries per contribution")
    d = Path(repo_dir) / CONTRIB_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{datetime.now(UTC):%Y%m%d}-{uuid.uuid4().hex[:8]}.jsonl"
    path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries), encoding="utf-8")
    return path


def load_contributions(repo_dir: str | Path) -> tuple[list[dict], list[str]]:
    """All valid, de-duplicated contribution entries in the repository (+ messages about rejected ones)."""
    d = Path(repo_dir) / CONTRIB_DIR
    entries, notes = [], []
    if not d.is_dir():
        return [], []
    for f in sorted(d.glob("*.jsonl")):
        rows = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) > MAX_PER_FILE:
            notes.append(f"{f.name}: more than {MAX_PER_FILE} entries, skipped")
            continue
        ok, bad = prepare(rows)
        entries += ok
        notes += [f"{f.name}:{rid}: {', '.join(why)}" for rid, why in bad]
    seen, unique = set(), []
    for e in entries:
        if e["id"] not in seen:
            seen.add(e["id"])
            unique.append(e)
    return unique, notes
