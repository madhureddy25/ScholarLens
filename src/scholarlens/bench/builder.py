"""Build a benchmark from a TOML config (see ``configs/bench.example.toml``).

Steps, each cached to the output directory so an interrupted build resumes:

    human.jsonl          sampled, filtered (pre-cutoff), paper-level splits
    ai.jsonl             one matched AI counterpart per human record
    transforms/<t>.jsonl attacks on AI text / legitimate edits of human text
    bench.jsonl          everything merged
    manifest.json        config, counts per split × family, leakage check
"""

from __future__ import annotations

import json
import tomllib
from collections import Counter
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path

from scholarlens.bench.generate import generate_all
from scholarlens.bench.schema import LABEL_AI, LABEL_HUMAN, Record, read_jsonl, write_jsonl
from scholarlens.bench.sources import Filters, make_source
from scholarlens.bench.splits import assign_splits, leakage
from scholarlens.bench.transforms import make_transform


def load_config(path: str | Path) -> dict:
    return tomllib.loads(Path(path).read_text())


def _cached(path: Path) -> list[Record] | None:
    return list(read_jsonl(path)) if path.is_file() else None


def build(config: dict, *, log=print, load_model=None) -> dict:
    if load_model is None:
        from scholarlens.llm import load_model
    b = config["bench"]
    out = Path(b["out"])
    out.mkdir(parents=True, exist_ok=True)
    seed = int(b.get("seed", 13))
    ratios = b.get("splits", {"train": 0.8, "val": 0.1, "test": 0.1})

    # 1. human text
    human_path = out / "human.jsonl"
    humans = _cached(human_path)
    if humans is None:
        filters = Filters(before_year=int(b.get("before_year", 2022)), min_words=int(b.get("min_words", 80)),
                          max_words=int(b.get("max_words", 400)))
        n_per = int(b["n_human"])
        humans = []
        for spec in config.get("sources", []):
            src = make_source({**spec, **({"seed": seed} if spec["type"] in ("arxiv", "hf") else {})})
            got = list(islice(src.records(filters), int(spec.get("limit", n_per)) if "limit" in spec else n_per))
            log(f"source {spec['type']}: {len(got)} human records")
            humans += got
        humans = assign_splits(humans, ratios, seed)
        write_jsonl(humans, human_path)
    log(f"human: {len(humans)}")

    # 2. AI counterparts
    ai_path = out / "ai.jsonl"
    ais = _cached(ai_path)
    if ais is None:
        gens = config.get("generators", [])
        if not gens:
            raise ValueError("config needs at least one [[generators]] entry")
        models = [load_model(g["model"], **g.get("options", {})) for g in gens]
        styles = tuple(b.get("prompt_styles", ["title"]))
        ais = generate_all(humans, models, styles=styles, seed=seed, max_tokens=int(b.get("max_tokens", 2048)),
                           on_progress=lambda k, n, r: log(f"  generated {k}/{n}") if k % 25 == 0 or k == n else None)
        write_jsonl(ais, ai_path)
    log(f"ai: {len(ais)}")

    # 3. transforms (resumable per record)
    derived: list[Record] = []
    for spec in config.get("transforms", []):
        t = make_transform(spec, load_model)
        tpath = out / "transforms" / f"{t.name}.jsonl"
        done = {r.parent_id for r in read_jsonl(tpath)} if tpath.is_file() else set()
        pool = {"human": humans, "ai": ais, "both": humans + ais}[t.apply_to]
        limit = spec.get("limit")
        todo = [r for r in pool if r.id not in done][: (int(limit) - len(done)) if limit else None]
        log(f"transform {t.name} on {t.apply_to}: {len(done)} cached, {len(todo)} to do")
        for i, rec in enumerate(todo, 1):
            res = t.apply(rec)
            if res is not None:
                write_jsonl([res], tpath, append=True)
            if i % 25 == 0:
                log(f"  {t.name}: {i}/{len(todo)}")
        derived += list(read_jsonl(tpath)) if tpath.is_file() else []

    allrecs = humans + ais + derived
    write_jsonl(allrecs, out / "bench.jsonl")
    manifest = summarize(allrecs)
    manifest.update({"built_at": datetime.now(UTC).isoformat(timespec="seconds"), "config": config})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    if manifest["leakage"]:
        raise RuntimeError(f"paper-level leakage across splits: {list(manifest['leakage'])[:5]}")
    log(f"wrote {out / 'bench.jsonl'} ({len(allrecs)} records)")
    return manifest


def summarize(records: list[Record]) -> dict:
    by = Counter((r.split, r.family) for r in records)
    table: dict[str, dict[str, int]] = {}
    for (split, family), n in sorted(by.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        table.setdefault(family, {})[str(split)] = n
    return {
        "n_records": len(records),
        "n_papers": len({r.paper_id for r in records}),
        "by_family_and_split": table,
        "generators": dict(Counter(r.generator for r in records if r.label == LABEL_AI and r.transform == "none")),
        "domains": dict(Counter(r.domain for r in records if r.label == LABEL_HUMAN and r.transform == "none")),
        "leakage": {k: sorted(v) for k, v in leakage(records).items()},
    }
