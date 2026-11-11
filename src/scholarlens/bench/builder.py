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
import random
import tomllib
from collections import Counter
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path

from scholarlens.bench.generate import generate_counterpart
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

    # 2. AI counterparts — one generator loaded at a time (fits a single GPU), each record's
    #    generator and prompt style fixed by its position, written as produced (resumable)
    ai_path = out / "ai.jsonl"
    ais = _cached(ai_path) or []
    done = {r.parent_id for r in ais}
    gens = config.get("generators", [])
    if not gens:
        raise ValueError("config needs at least one [[generators]] entry")
    styles = tuple(b.get("prompt_styles", ["title"]))
    rng = random.Random(seed)
    plan = [(i % len(gens), rng.choice(styles), h) for i, h in enumerate(humans)]
    for gi, g in enumerate(gens):
        jobs = [(h, st) for k, st, h in plan if k == gi and h.id not in done]
        if not jobs:
            continue
        log(f"generator {g['model']}: {len(jobs)} to generate ({len(ais)} cached overall)")
        model = load_model(g["model"], **g.get("options", {}))
        concurrency = int(g.get("concurrency", b.get("concurrency", 1)))

        def keep(k, n, rec):
            if rec is not None:
                write_jsonl([rec], ai_path, append=True)
                ais.append(rec)
            if k % 25 == 0 or k == n:
                log(f"  generated {k}/{n}")

        _run_jobs(jobs, model, max_tokens=int(b.get("max_tokens", 2048)), concurrency=concurrency, on_progress=keep)
        del model
        _release_gpu()
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
        workers = int(spec.get("concurrency", b.get("concurrency", 1))) if spec["type"] in ("llm_paraphrase", "quill") else 1
        if workers <= 1:
            results = (t.apply(rec) for rec in todo)
        else:
            from concurrent.futures import ThreadPoolExecutor

            pool = ThreadPoolExecutor(max_workers=workers)
            results = pool.map(t.apply, todo)
        for i, res in enumerate(results, 1):
            if res is not None:
                write_jsonl([res], tpath, append=True)
            if i % 25 == 0:
                log(f"  {t.name}: {i}/{len(todo)}")
        del t
        _release_gpu()

    # merge every transform cached in this benchmark, including ones produced by an earlier
    # config (e.g. DIPPER run separately on a bigger GPU)
    tdir = out / "transforms"
    derived = [r for p in sorted(tdir.glob("*.jsonl")) for r in read_jsonl(p)] if tdir.is_dir() else []

    # generated text is cleaned again at assembly (idempotent), so caches built before a
    # cleaning fix are corrected too; human text is never altered here
    from scholarlens.bench.generate import clean_generation

    for r in ais + derived:
        if r.label == LABEL_AI or r.transform in ("llm_paraphrase",):
            r.text = clean_generation(r.text)
    allrecs = humans + ais + derived
    write_jsonl(allrecs, out / "bench.jsonl")
    manifest = summarize(allrecs)
    manifest.update({"built_at": datetime.now(UTC).isoformat(timespec="seconds"), "config": config})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    if manifest["leakage"]:
        raise RuntimeError(f"paper-level leakage across splits: {list(manifest['leakage'])[:5]}")
    log(f"wrote {out / 'bench.jsonl'} ({len(allrecs)} records)")
    return manifest


def _run_jobs(jobs, model, *, max_tokens: int, concurrency: int, on_progress) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if concurrency <= 1:
        for k, (h, st) in enumerate(jobs, 1):
            on_progress(k, len(jobs), generate_counterpart(h, model, style=st, max_tokens=max_tokens))
        return
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(generate_counterpart, h, model, style=st, max_tokens=max_tokens) for h, st in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            on_progress(k, len(jobs), f.result())


def _release_gpu() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


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
