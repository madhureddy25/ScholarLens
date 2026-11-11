# ScholarLens

Integrity-first tooling for academic text:

- **Lens**: AI-text detection for scientific prose that stays robust to paraphrase attacks and is audited for false positives on honest authors.
- **Quill**: a citation-safe editing assistant for an author's own drafts. Citations, equations, numbers and domain terms are guaranteed untouched, and every session produces a disclosure log.

Both share one core: a structure-aware document parser and a masking engine that protects everything that must not change.
See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design, evaluation targets and roadmap.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[ml,dev]"          # core + PyTorch/transformers + tests
```

CPU-only PyTorch (smaller download):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Command line

```bash
scholarlens parse paper.tex            # typed node structure (prose in green)
scholarlens mask paper.tex             # masked prose + symbol map
scholarlens roundtrip paper.tex        # proves parse→reconstruct and mask→unmask are lossless
scholarlens features paper.tex --lm gpt2 --embedder sentence-transformers/all-MiniLM-L6-v2
```

Supported inputs: `.tex`, `.md`, `.txt`, `.docx`, GROBID TEI `.xml`, and `.pdf`. PDFs need a GROBID server: `docker run -p 8070:8070 lfoppiano/grobid:0.8.1`, and set `SCHOLARLENS_GROBID_URL` if it isn't on localhost.

## Python

```python
from scholarlens.ingest import load
from scholarlens.masking import Masker

doc = load("paper.tex")
masker = Masker()
for node in doc.prose():
    m = masker.mask_node(node, doc.source_format)
    print(m.masked)            # "Prior work ⟦CITE_1⟧ used ⟦TERM_1⟧ with ⟦NUM_1⟧."
    report = m.verify(m.masked)  # every sentinel exactly once, none invented
    assert m.unmask() == node.text
```

## Tests

```bash
pytest -q                 # full suite (downloads GPT-2 once for the LM test)
pytest -q -m "not slow"   # offline
```
