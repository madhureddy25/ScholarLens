# ScholarLens

Integrity-first tooling for academic text:

- **Lens**: AI-text detection for scientific prose that stays robust to paraphrase attacks and is audited for false positives on honest authors.
- **Quill**: a citation-safe editing assistant for an author's own drafts. Citations, equations, numbers and domain terms are guaranteed untouched, and every session produces a disclosure log.

Both share one core: a structure-aware document parser and a masking engine that protects everything that must not change.
See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design, evaluation targets and roadmap.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[all,dev]"         # core + ML + API clients + tests
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

## Quill: editing your own draft

```bash
scholarlens edit paper.tex -m anthropic:claude-opus-5-5 --mode clarity          # Claude (ANTHROPIC_API_KEY)
scholarlens edit paper.tex -m ollama:llama3.1 --mode proofread                   # local Ollama server
scholarlens edit paper.tex -m hf:Qwen/Qwen2.5-1.5B-Instruct                      # fully offline, in-process
scholarlens edit paper.tex -m openai-compatible:my-model --opt base_url=http://gpu-box:8000/v1
scholarlens edit paper.tex --dry-run                                             # see exactly what would be sent
```

Modes: `proofread` (minimal fixes), `clarity`, `concise`. Restrict with `--sections Introduction,Discussion` or `--nodes n0005,n0012`.
Outputs go to `paper_quill/`:

| File | Contents |
|---|---|
| `paper.edited.tex` | Your file with accepted edits. Bytes outside edited paragraphs are identical. |
| `paper.tracked.tex` | Review copy with `\DIFdel{}`/`\DIFadd{}` marks (compiles with `ulem` + `xcolor`). |
| `paper.review.html` | Side-by-side review: every paragraph, its status, similarity score, and diff. |
| `paper.disclosure.json` | Model, settings, per-paragraph record, and a draft AI-use disclosure statement. |

Every edit must pass three gates or the original paragraph is kept: (1) every protected item comes back exactly once, (2) LaTeX stays balanced, (3) semantic similarity to your text is at least `--min-similarity` (default 0.94).

## Models: bring your own

Any model can be plugged in. Use a spec string `provider:model`, or a named profile from `scholarlens.toml` (template: `configs/models.example.toml`):

| Provider | Example | Credentials |
|---|---|---|
| `anthropic` | `anthropic:claude-opus-5-5` | `ANTHROPIC_API_KEY` or `ant auth login` |
| `openai` | `openai:<model>` | `OPENAI_API_KEY` |
| `ollama`, `vllm`, `lmstudio`, `llamacpp` | `ollama:llama3.1` | none (local server) |
| `openrouter`, `groq`, `together`, `deepseek`, `mistral`, `gemini` | `openrouter:<org>/<model>` | provider's API key env var |
| `openai-compatible` | `--opt base_url=http://host:8000/v1` | optional `api_key_env` |
| `hf` | `hf:Qwen/Qwen2.5-1.5B-Instruct` | none (runs in-process) |

Pass model options with `--opt key=value` (e.g. `--opt effort=low`, `--opt temperature=0.2`). `scholarlens models` lists providers and profiles.
To add your own backend, subclass `scholarlens.llm.ChatModel` and either call `register_provider("name", factory)` or expose it under the `scholarlens.providers` entry-point group in your package.

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
