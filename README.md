# ScholarLens

Integrity-first tooling for academic text:

- **Lens**: AI-text detection for scientific prose that stays robust to paraphrase attacks and is audited for false positives on honest authors.
- **Quill**: a citation-safe editing assistant for an author's own drafts. Citations, equations, numbers and domain terms are guaranteed untouched, and every session produces a disclosure log.

Both share one core: a structure-aware document parser and a masking engine that protects everything that must not change.
## Documentation

| Document | Read it for |
|---|---|
| [`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md) | History, design decisions and their reasons, bugs and research findings, next steps |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System design, module map, evaluation targets, roadmap |
| [`LAPTOP_COMMANDS.ps1`](LAPTOP_COMMANDS.ps1) | **Every command for the laptop study in one file**, in order |
| [`docs/LAPTOP_COMMANDS.pdf`](docs/LAPTOP_COMMANDS.pdf) | The same 50 commands as a printable, numbered PDF |
| [`docs/WINDOWS.md`](docs/WINDOWS.md) | Step-by-step PowerShell runbook for the study on a Windows laptop |
| [`docs/LAPTOP.md`](docs/LAPTOP.md) / [`docs/GPU.md`](docs/GPU.md) | 6 GB laptop GPU / 24 GB GPU setups |
| [`docs/LEARNING.md`](docs/LEARNING.md) | How the community learning loop works, and its guard rails |
| [`docs/PAPER_PLAN.md`](docs/PAPER_PLAN.md) | Research questions, tables and figures for the paper |
| [`reports/`](reports/) | Saved evaluation reports |
| [`CLAUDE.md`](CLAUDE.md) | Project rules and commands for AI coding assistants |

**Status (Nov 2026):** all components are implemented and tested, and CI runs on Python 3.11–3.13. Next: run the laptop study ([`docs/WINDOWS.md`](docs/WINDOWS.md)), publish the first releases, and write the paper.

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

## Bench and Lens: the robustness and fairness study

```bash
scholarlens bench build configs/bench.example.toml      # human (pre-2022) + matched AI + attacks + edits
scholarlens bench stats data/bench/pilot/bench.jsonl    # counts per family × split, leakage check
scholarlens lens train data/bench/pilot/bench.jsonl --out models/lens --gnn
scholarlens lens train data/bench/pilot/bench.jsonl --exclude ai+llm_paraphrase   # unseen-attack test
scholarlens detect paper.tex --model models/lens       # per-segment calibrated P(AI)
```

`lens train` writes `report.md`, which has two tables:
1. AUROC and TPR at 1% FPR for each detector against each attack family.
2. False-positive rate on each human family: clean, Quill-edited and grammar-checked text, broken down by domain.

The second table is the fairness result: it shows whether honest authors who edit their writing get flagged.

**Running on your own GPU:** follow [`docs/GPU.md`](docs/GPU.md) with `configs/bench.gpu.toml` (three open 7B generators, DIPPER, LLM paraphrase, Quill edits).

Builds are resumable and every step is cached. `configs/bench.pilot-cpu.toml` runs fully offline with small local models (an easy pilot); the real study should use several strong generators (see `configs/bench.example.toml`). DIPPER attacks need a GPU.

## It keeps improving: community learning

Authors review Quill's suggestions (`scholarlens review`). They can share their accept/reject decisions as masked text (`scholarlens contribute`). CI retrains Lens and publishes a new version only if it beats the current one on a frozen test set. Quill is improved by reinforcement learning on checkable rewards: citations kept, meaning preserved, real errors fixed, restraint. A detector is never part of its reward. Everyone gets the latest models with `scholarlens update`. See [`docs/LEARNING.md`](docs/LEARNING.md). On a 6 GB laptop GPU, follow [`docs/LAPTOP.md`](docs/LAPTOP.md).

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
