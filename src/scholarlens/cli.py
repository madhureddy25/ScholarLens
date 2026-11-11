"""Command-line interface: ``scholarlens --help``."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from scholarlens.analysis import analyze, prose_units
from scholarlens.ingest import load
from scholarlens.masking import MaskConfig, Masker

app = typer.Typer(add_completion=False, help="ScholarLens — integrity-first tooling for academic text.")
console = Console()


@app.command()
def parse(path: Path, full: bool = typer.Option(False, help="Show full node text.")) -> None:
    """Show the document structure (Phase 1)."""
    doc = load(path)
    t = Table(title=f"{path.name} — {doc.source_format}", show_lines=False)
    for col in ("id", "type", "section", "span", "text"):
        t.add_column(col, overflow="fold")
    for n in doc.nodes:
        text = n.text if full else (n.text[:80] + ("…" if len(n.text) > 80 else ""))
        span = f"{n.start}-{n.end}" if n.has_span else ""
        style = "green" if n.is_prose else "dim"
        t.add_row(n.id, n.type.value, n.section or "", span, text.replace("\n", "⏎"), style=style)
    console.print(t)
    console.print(f"[bold]{sum(1 for _ in doc.prose())}[/] prose nodes · {doc.stats()}")


@app.command()
def mask(path: Path, node: str = typer.Option(None, help="Only this node id."),
         glossary: Path = typer.Option(None, help="File with one protected term per line.")) -> None:
    """Show masked prose and the symbol map (Phase 2)."""
    doc = load(path)
    terms = tuple(t.strip() for t in glossary.read_text(encoding="utf-8").splitlines() if t.strip()) if glossary else ()
    units = prose_units(doc, Masker(MaskConfig(glossary=terms)))
    for u in units:
        if node and u.node_id != node:
            continue
        console.rule(f"{u.node_id} · {u.section or ''} · {u.mask.counts()}")
        console.print(u.mask.masked, markup=False, highlight=False)
        for k, v in u.mask.symbols.items():
            console.print(f"  [cyan]{k}[/] = {v!r}", highlight=False)


@app.command()
def roundtrip(path: Path) -> None:
    """Verify that parse → reconstruct reproduces the file byte-for-byte."""
    doc = load(path)
    if doc.source is None:
        console.print(f"[yellow]{doc.source_format} has no span reconstruction (rendered output only)[/]")
        raise typer.Exit(0)
    ok = doc.reconstruct({}) == doc.source
    masker = Masker()
    bad = [u.node_id for u in prose_units(doc, masker) if u.mask.unmask() != doc.get(u.node_id).text]
    console.print(f"reconstruct identity: {'[green]ok' if ok else '[red]FAILED'}[/]")
    console.print(f"mask/unmask identity: {'[green]ok' if not bad else '[red]FAILED ' + str(bad)}[/]")
    raise typer.Exit(0 if ok and not bad else 1)


@app.command()
def features(path: Path, lm: str = typer.Option(None, help="HF causal LM for PPL/entropy, e.g. gpt2."),
             embedder: str = typer.Option(None, help="sentence-transformers model for transitions."),
             segment_size: int = 250, output: Path = typer.Option(None, help="Write JSON here.")) -> None:
    """Compute detection signals per ~250-word segment."""
    doc = load(path)
    lm_scorer = emb = None
    if lm:
        from scholarlens.features.lm import CausalLMScorer

        lm_scorer = CausalLMScorer(lm)
    if embedder:
        from scholarlens.features.transitions import SentenceTransformerEmbedder

        emb = SentenceTransformerEmbedder(embedder)
    report = analyze(doc, embedder=emb, lm=lm_scorer, segment_size=segment_size)
    if output:
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        console.print(f"wrote {output}")
        return
    df = report["document_features"]
    t = Table(title="document features")
    t.add_column("feature")
    t.add_column("value", justify="right")
    for k, v in df.items():
        t.add_row(k, f"{v:.4f}")
    console.print(t)
    console.print(f"{len(report['segments'])} segments · protected spans {report['protected_spans']}")


@app.command()
def models(config: Path = typer.Option(None, help="Profiles file (default: ./scholarlens.toml).")) -> None:
    """List model providers and configured profiles."""
    from scholarlens.llm import available_providers, load_profiles

    console.print("[bold]Providers[/] (use as --model provider:model): " + ", ".join(available_providers()))
    profiles, default = load_profiles(config)
    if not profiles:
        console.print("No profiles found. Copy configs/models.example.toml to ./scholarlens.toml to add some.")
        return
    t = Table(title="Profiles")
    for col in ("name", "provider", "model", "options"):
        t.add_column(col)
    for name, p in profiles.items():
        opts = {k: v for k, v in p.items() if k not in ("provider", "model")}
        t.add_row(name + (" (default)" if name == default else ""), str(p.get("provider", "")),
                  str(p.get("model", "")), ", ".join(f"{k}={v}" for k, v in opts.items()))
    console.print(t)


@app.command()
def edit(
    path: Path,
    model: str = typer.Option(None, "--model", "-m", help="provider:model or profile name."),
    opt: list[str] = typer.Option(None, "--opt", help="Model option key=value (repeatable), e.g. effort=low."),
    mode: str = typer.Option("clarity", help="proofread | clarity | concise"),
    out: Path = typer.Option(None, help="Output directory (default: <file>_quill/)."),
    sections: str = typer.Option(None, help="Comma-separated section titles to edit."),
    nodes: str = typer.Option(None, help="Comma-separated node ids to edit (see `scholarlens parse`)."),
    min_similarity: float = typer.Option(0.94, help="Semantic-similarity gate; 0 disables."),
    similarity_model: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2",
                                         help="sentence-transformers model for the similarity gate."),
    max_attempts: int = typer.Option(3, help="Attempts per paragraph before keeping the original."),
    glossary: Path = typer.Option(None, help="File with one protected term per line."),
    config: Path = typer.Option(None, help="Model profiles file."),
    dry_run: bool = typer.Option(False, help="Show what would be sent to the model, without calling it."),
) -> None:
    """Quill: citation-safe language editing of your own manuscript."""
    from scholarlens.edit import EditConfig, Quill, write_outputs
    from scholarlens.edit.prompts import system_prompt, user_prompt
    from scholarlens.llm import ModelConfigError, load_model
    from scholarlens.llm.registry import parse_options

    doc = load(path)
    terms = tuple(t.strip() for t in glossary.read_text(encoding="utf-8").splitlines() if t.strip()) if glossary else ()
    masker = Masker(MaskConfig(glossary=terms))
    cfg = EditConfig(mode=mode, max_attempts=max_attempts, min_similarity=min_similarity or None,
                     sections=tuple(s.strip() for s in sections.split(",")) if sections else None,
                     node_ids=tuple(n.strip() for n in nodes.split(",")) if nodes else None)
    selected = [n for n in doc.prose()
                if (cfg.node_ids is None or n.id in cfg.node_ids)
                and (cfg.sections is None or (n.section or "").lower() in {s.lower() for s in cfg.sections})]
    n_words = sum(len(n.text.split()) for n in selected)
    console.print(f"{len(selected)} prose paragraphs selected (~{n_words} words) from {path.name}")
    if dry_run:
        console.rule("system prompt")
        console.print(system_prompt(mode, doc.source_format), markup=False, highlight=False)
        for n in selected[:3]:
            console.rule(f"user message for {n.id}")
            console.print(user_prompt(masker.mask_node(n, doc.source_format).masked), markup=False, highlight=False)
        if len(selected) > 3:
            console.print(f"… and {len(selected) - 3} more")
        return

    try:
        chat = load_model(model, config=config, **parse_options(opt))
    except ModelConfigError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(2) from e
    sim = None
    if cfg.min_similarity:
        from scholarlens.edit import sentence_transformer_similarity

        sim = sentence_transformer_similarity(similarity_model)
    console.print(f"model: [bold]{chat.name}[/] · mode: {mode} · similarity gate: "
                  f"{cfg.min_similarity if sim else 'off'}")
    quill = Quill(chat, masker=masker, similarity=sim, config=cfg)

    def progress(k: int, total: int, e) -> None:
        if e.status.value != "skipped" or e.reason not in ("not selected", "section not selected"):
            console.print(f"  [{k}/{total}] {e.node_id} {e.status.value}"
                          + (f" (sim {e.similarity:.3f})" if e.similarity is not None else "")
                          + (f" — {e.reason}" if e.reason and e.status.value != "unchanged" else ""))

    try:
        session = quill.edit_document(doc, on_progress=progress)
    except ModelConfigError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(2) from e
    files = write_outputs(session, out or path.with_name(path.stem + "_quill"), input_path=path)
    console.print(f"[green]done[/] {session.counts()}")
    for k, f in files.items():
        console.print(f"  {k:10} {f}")


bench_app = typer.Typer(help="Bench: build and inspect detection benchmarks.")
app.add_typer(bench_app, name="bench")


@bench_app.command("build")
def bench_build(config: Path, n_human: int = typer.Option(None, help="Override [bench].n_human."),
                out: Path = typer.Option(None, help="Override [bench].out.")) -> None:
    """Build (or resume) a benchmark from a TOML config."""
    from scholarlens.bench.builder import build, load_config

    cfg = load_config(config)
    if n_human:
        cfg["bench"]["n_human"] = n_human
    if out:
        cfg["bench"]["out"] = str(out)
    manifest = build(cfg, log=lambda m: console.print(m, highlight=False))
    _print_bench_table(manifest)


@bench_app.command("stats")
def bench_stats(path: Path) -> None:
    """Counts per family × split for a bench.jsonl file."""
    from scholarlens.bench.builder import summarize
    from scholarlens.bench.schema import read_jsonl

    _print_bench_table(summarize(list(read_jsonl(path))))


def _print_bench_table(manifest: dict) -> None:
    t = Table(title=f"{manifest['n_records']} records · {manifest['n_papers']} papers")
    splits = sorted({s for row in manifest["by_family_and_split"].values() for s in row})
    t.add_column("family")
    for s in splits:
        t.add_column(s, justify="right")
    for fam, row in manifest["by_family_and_split"].items():
        t.add_row(fam, *[str(row.get(s, 0)) for s in splits])
    console.print(t)
    console.print(f"generators: {manifest['generators']}")
    console.print(f"leakage: {'[green]none[/]' if not manifest['leakage'] else '[red]' + str(manifest['leakage'])}")


lens_app = typer.Typer(help="Lens: train and evaluate AI-text detectors on a Bench file.")
app.add_typer(lens_app, name="lens")


@lens_app.command("train")
def lens_train(
    bench: Path,
    out: Path = typer.Option(Path("models/lens"), help="Output directory for models and report."),
    lm: str = typer.Option("gpt2", help="Causal LM for perplexity/entropy features ('none' to skip)."),
    embedder: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2", help="Sentence embedder."),
    gnn: bool = typer.Option(False, help="Also train the sentence-graph GNN."),
    binoculars: bool = typer.Option(False, help="Also run the Binoculars zero-shot baseline (gpt2/distilgpt2)."),
    exclude: str = typer.Option(None, help="Comma-separated families held out of training (unseen-attack test)."),
    estimator: str = typer.Option("hgb", help="hgb (scikit-learn) or lightgbm."),
    fpr: float = typer.Option(0.01, help="Operating false-positive rate on clean human text."),
) -> None:
    """Train Lens on the train split, calibrate on val, report robustness and fairness on test."""
    from scholarlens.detect.runner import train_and_evaluate

    lm_scorer = emb = bino = None
    if lm and lm != "none":
        from scholarlens.features.lm import CausalLMScorer

        lm_scorer = CausalLMScorer(lm)
    if embedder and embedder != "none":
        from scholarlens.features.transitions import SentenceTransformerEmbedder

        emb = SentenceTransformerEmbedder(embedder)
    if binoculars:
        from scholarlens.detect.zero_shot import Binoculars

        bino = Binoculars()
    train_and_evaluate(bench, out, lm=lm_scorer, embedder=emb, gnn=gnn, binoculars=bino,
                                exclude_families=tuple(e.strip() for e in exclude.split(",")) if exclude else (),
                                estimator=estimator, fpr=fpr, log=lambda m: console.print(m, highlight=False))
    console.print((out / "report.md").read_text(encoding="utf-8"), markup=False, highlight=False)
    console.print(f"[green]saved[/] {out}/fusion, {out}/report.json, {out}/report.md")


@app.command()
def detect(path: Path, model: Path = typer.Option(None, help="Directory from `lens train` (default: installed release)."),
           segment_size: int = 250) -> None:
    """Score a manuscript with a trained Lens model (per ~250-word segment)."""
    import json as _json

    from scholarlens.learn.release import installed

    model = model or installed("lens") or Path("models/lens")
    if not (model / "fusion" / "meta.json").is_file():
        console.print("no Lens model found: run `scholarlens update lens` or `scholarlens lens train`")
        raise typer.Exit(1)

    from scholarlens.analysis import prose_units, segments
    from scholarlens.detect import load_featurizer_from_signature
    from scholarlens.detect.fusion import FusionDetector

    meta = _json.loads((model / "fusion" / "meta.json").read_text(encoding="utf-8"))
    det = FusionDetector.load(model / "fusion", load_featurizer_from_signature(meta["featurizer"]))
    segs = segments(prose_units(load(path)), segment_size)
    if not segs:
        console.print("no prose found")
        raise typer.Exit(1)
    probs = det.score_texts([s.text for s in segs], ids=[f"{path}:{s.index}" for s in segs])
    t = Table(title=f"Lens — {path.name}")
    for col in ("segment", "nodes", "words", "P(AI)"):
        t.add_column(col, justify="right" if col != "nodes" else "left")
    for s, p in zip(segs, probs, strict=True):
        t.add_row(str(s.index), ",".join(s.node_ids[:4]) + ("…" if len(s.node_ids) > 4 else ""), str(s.n_words),
                  f"{p:.2f}")
    console.print(t)
    console.print(f"document mean P(AI) = {float(probs.mean()):.2f} · max = {float(probs.max()):.2f}")
    console.print("[dim]Probabilities are calibrated on the Bench data this model was trained on and are evidence, "
                  "not a verdict. Never use them alone to make decisions about an author.[/]")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the HTTP API (see scholarlens.api.app for configuration)."""
    import uvicorn

    uvicorn.run("scholarlens.api.app:app", host=host, port=port)


@app.command()
def review(
    session_file: Path,
    accept_all: bool = typer.Option(False, help="Accept every edit without prompting."),
    reject_all: bool = typer.Option(False, help="Reject every edit without prompting."),
    decisions: Path = typer.Option(None, help='JSON file {"n0005": "accept", "n0006": "reject"}.'),
    out: Path = typer.Option(None, help="Output directory (default: next to the session file)."),
    no_feedback: bool = typer.Option(False, "--no-feedback", help="Do not record decisions locally."),
) -> None:
    """Accept or reject each Quill edit; writes the final document with only accepted edits."""
    import json as _json

    from rich.text import Text

    from scholarlens.edit.diff import diff
    from scholarlens.learn.review import review_session

    fixed = _json.loads(decisions.read_text(encoding="utf-8")) if decisions else {}

    def show(e) -> None:
        t = Text()
        for o in diff(e.mask.masked, e.edited_masked):
            old, new = e.mask._substitute("".join(o.old)), e.mask._substitute("".join(o.new))
            if o.tag == "equal":
                t.append(old)
                continue
            if old.strip():
                t.append(old, style="red strike")
            if new.strip():
                t.append(new, style="green")
        console.rule(f"{e.node_id} · {e.section or ''} · similarity {e.similarity}")
        console.print(t)

    def decide(e) -> str:
        if e.node_id in fixed:
            return fixed[e.node_id]
        if accept_all or reject_all:
            return "accept" if accept_all else "reject"
        show(e)
        answer = typer.prompt("[a]ccept / [r]eject / [s]kip", default="a").strip().lower()[:1]
        return {"a": "accept", "r": "reject", "s": "skip"}.get(answer, "skip")

    res = review_session(session_file, decide, out_dir=out, record_feedback=not no_feedback)
    console.print(f"accepted {len(res.accepted)} · rejected {len(res.rejected)} · skipped {len(res.skipped)}")
    for k, f in res.files.items():
        console.print(f"  {k:10} {f}")
    if res.feedback_entries and not no_feedback:
        console.print(f"[dim]{len(res.feedback_entries)} decisions saved locally (masked text only). "
                      "Run `scholarlens contribute` to share them and help improve Quill.[/]")


@app.command()
def contribute(
    repo_dir: Path = typer.Option(Path("."), help="Your local clone of the ScholarLens repository."),
    yes: bool = typer.Option(False, "--yes", help="Confirm without prompting."),
) -> None:
    """Share your review decisions (masked text only) as a public contribution file for a pull request."""
    from scholarlens.learn import feedback
    from scholarlens.learn.contribute import prepare, write_contribution

    pending = [e for e in feedback.load_all() if not e.get("contributed")]
    ok, bad = prepare(pending)
    console.print(f"{len(pending)} unshared decisions · {len(ok)} shareable · {len(bad)} held back")
    for rid, why in bad[:10]:
        console.print(f"  [yellow]held back[/] {rid[:8]}: {', '.join(why)}")
    if not ok:
        return
    console.rule("example of what will be shared")
    console.print(_json_preview(ok[0]), markup=False, highlight=False)
    console.print("[bold]Contributions are public[/] and become part of the repository history. Only masked text is "
                  "included, but share only text you have the right to share.")
    if not yes and not typer.confirm(f"Write {len(ok)} entries to {repo_dir / 'contributions'}?"):
        raise typer.Exit(1)
    path = write_contribution(ok, repo_dir)
    feedback.mark_contributed({e["id"] for e in pending} - {rid for rid, _ in bad})
    console.print(f"[green]wrote[/] {path}")
    console.print("Open a pull request with it:\n"
                  f"  git checkout -b contrib-{path.stem}\n  git add {path.relative_to(repo_dir)}\n"
                  f'  git commit -m "Feedback contribution {path.stem}"\n  git push -u origin contrib-{path.stem}\n'
                  "  gh pr create --fill")


def _json_preview(entry: dict) -> str:
    import json as _json

    return _json.dumps({k: entry[k] for k in ("mode", "decision", "masked_input", "masked_output")}, indent=2,
                       ensure_ascii=False)


learn_app = typer.Typer(help="Continuous learning: improve Quill and Lens from data and community feedback.")
app.add_typer(learn_app, name="learn")


@learn_app.command("quill")
def learn_quill(
    bench: Path = typer.Option(..., help="bench.jsonl (public human paragraphs; test split is the frozen eval set)."),
    repo_dir: Path = typer.Option(Path("."), help="Repository checkout containing contributions/."),
    base: str = typer.Option("Qwen/Qwen2.5-0.5B-Instruct", help="Base instruct model."),
    current: str = typer.Option(None, help="Current adapter to beat (directory); default: the base model alone."),
    mode: str = typer.Option("proofread", help="proofread | clarity | concise"),
    out: Path = typer.Option(Path("models/quill-rl"), help="Where to write the new adapter and report."),
    steps: int = typer.Option(200, help="GRPO/DPO optimisation steps."),
    group: int = typer.Option(4, help="GRPO group size (candidates per paragraph)."),
    load_in_4bit: bool = typer.Option(False, "--4bit", help="QLoRA (needed for 1.5B+ on a 6 GB GPU)."),
    cpu: bool = typer.Option(False, help="Train on CPU (slow; for testing)."),
    grammar_url: str = typer.Option(None, help="LanguageTool server for the quality reward."),
    judge: str = typer.Option(None, help="Model spec for the LLM-judge quality reward (e.g. ollama:qwen2.5:3b)."),
    similarity_model: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2"),
) -> None:
    """Train a better Quill adapter and keep it only if it beats the current one on a frozen test set."""
    from scholarlens.edit import sentence_transformer_similarity
    from scholarlens.learn.quill import improve_quill
    from scholarlens.llm import load_model
    from scholarlens.rl.rewards import LanguageToolChecker, LLMJudge, QuillReward
    from scholarlens.rl.train import RLConfig

    reward = QuillReward(similarity=sentence_transformer_similarity(similarity_model),
                         grammar=LanguageToolChecker(grammar_url) if grammar_url else None,
                         judge=LLMJudge(load_model(judge)) if judge else None)
    if not grammar_url and not judge:
        console.print("[yellow]No --grammar-url or --judge: the quality reward is off, so training mainly teaches "
                      "integrity, fidelity and restraint.[/]")
    cfg = RLConfig(base_model=base, out_dir=str(out), max_steps=steps, num_generations=group,
                   per_device_train_batch_size=group, load_in_4bit=load_in_4bit, use_cpu=cpu)
    rep = improve_quill(bench=bench, repo_dir=repo_dir, cfg=cfg, reward=reward, mode=mode, current_adapter=current,
                        log=lambda m: console.print(m, highlight=False))
    t = Table(title="Quill: candidate vs current (frozen eval set)")
    for col in ("metric", "current", "candidate"):
        t.add_column(col)
    for k in ("integrity_rate", "format_rate", "mean_reward", "mean_similarity", "mean_quality", "mean_change"):
        a, b = rep["current"].get(k), rep["candidate"].get(k)
        t.add_row(k, "—" if a is None else f"{a:.3f}", "—" if b is None else f"{b:.3f}")
    console.print(t)
    if rep["accepted"]:
        console.print(f"[green]accepted[/]: publish with `scholarlens learn publish quill {out}`")
    else:
        console.print(f"[yellow]not accepted[/]: {'; '.join(rep['reasons'])}")


@learn_app.command("lens")
def learn_lens(
    bench: Path = typer.Option(..., help="Base bench.jsonl (its test split is the frozen yardstick)."),
    repo_dir: Path = typer.Option(Path("."), help="Repository checkout containing contributions/."),
    current: Path = typer.Option(None, help="Current Lens directory (default: the installed release, if any)."),
    out: Path = typer.Option(Path("models/lens-candidate")),
    lm: str = typer.Option("gpt2"),
    embedder: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2"),
) -> None:
    """Retrain Lens with contributions and hard examples; accept only if it beats the current version."""
    from scholarlens.detect.featurize import Featurizer
    from scholarlens.features.lm import CausalLMScorer
    from scholarlens.features.transitions import SentenceTransformerEmbedder
    from scholarlens.learn.lens import improve_lens
    from scholarlens.learn.release import installed

    featurizer = Featurizer(lm=CausalLMScorer(lm) if lm != "none" else None,
                            embedder=SentenceTransformerEmbedder(embedder), cache=out / "features.jsonl")
    rep = improve_lens(bench=bench, repo_dir=repo_dir, out_dir=out, featurizer=featurizer,
                       current_dir=current or installed("lens"), log=lambda m: console.print(m, highlight=False))
    t = Table(title="Lens candidate vs current (frozen test split)")
    for col in ("family", "current AUROC", "candidate AUROC"):
        t.add_column(col)
    for fam, row in rep["candidate"]["attacks"].items():
        cur = rep["current"]["attacks"].get(fam, {}).get("auroc") if rep["current"] else None
        t.add_row(fam, "—" if cur is None else f"{cur:.4f}", f"{row['auroc']:.4f}")
    console.print(t)
    if rep["accepted"]:
        console.print(f"[green]accepted[/]: publish with `scholarlens learn publish lens {out}`")
    else:
        console.print(f"[yellow]not accepted[/]: {'; '.join(rep['reasons'])}")
        raise typer.Exit(3)


@learn_app.command("validate")
def learn_validate(repo_dir: Path = typer.Option(Path(".")),
                   strict: bool = typer.Option(False, help="Exit non-zero if any entry is invalid (for PR checks).")) -> None:
    """Check contribution files (schema, placeholders, personal data, duplicates)."""
    from scholarlens.learn.contribute import load_contributions

    entries, notes = load_contributions(repo_dir)
    console.print(f"{len(entries)} valid contribution entries")
    for n in notes:
        console.print(f"  [yellow]invalid[/] {n}")
    if strict and notes:
        raise typer.Exit(1)


@learn_app.command("publish")
def learn_publish(kind: str, directory: Path, force: bool = typer.Option(False, help="Publish even if not accepted.")) -> None:
    """Publish a model (lens | quill) or benchmark (bench) directory as a GitHub Release (needs `gh auth login`)."""
    from scholarlens.learn.release import publish

    console.print(f"[green]published[/] {publish(kind, directory, force=force)}")


@app.command()
def update(kind: str = typer.Argument("all", help="lens | quill | all")) -> None:
    """Install the latest community-improved Lens and/or Quill from GitHub Releases."""
    from scholarlens.learn.release import install, latest_release

    for k in (("lens", "quill") if kind == "all" else (kind,)):
        rel = latest_release(k)
        if rel is None:
            console.print(f"{k}: no release yet")
            continue
        console.print(f"{k}: installed {rel['tag_name']} → {install(k, rel)}")


if __name__ == "__main__":
    app()
