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


if __name__ == "__main__":
    app()
