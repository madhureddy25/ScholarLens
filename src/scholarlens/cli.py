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
    terms = tuple(t.strip() for t in glossary.read_text().splitlines() if t.strip()) if glossary else ()
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
        output.write_text(json.dumps(report, indent=2))
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
    terms = tuple(t.strip() for t in glossary.read_text().splitlines() if t.strip()) if glossary else ()
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
    console.print((out / "report.md").read_text(), markup=False, highlight=False)
    console.print(f"[green]saved[/] {out}/fusion, {out}/report.json, {out}/report.md")


@app.command()
def detect(path: Path, model: Path = typer.Option(Path("models/lens"), help="Directory from `lens train`."),
           segment_size: int = 250) -> None:
    """Score a manuscript with a trained Lens model (per ~250-word segment)."""
    import json as _json

    from scholarlens.analysis import prose_units, segments
    from scholarlens.detect import load_featurizer_from_signature
    from scholarlens.detect.fusion import FusionDetector

    meta = _json.loads((model / "fusion" / "meta.json").read_text())
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


if __name__ == "__main__":
    app()
