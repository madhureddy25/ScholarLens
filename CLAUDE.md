# ScholarLens: notes for Claude Code

Read `docs/PROJECT_LOG.md` first. It has the project history, every design decision with its reasoning, the bugs found, and the next steps. The architecture is in `docs/ARCHITECTURE.md`.

## What this is

Integrity-first tooling for academic text:
- **Lens** (`detect/`): paraphrase-robust AI-text detection with fairness audits.
- **Quill** (`edit/`): citation-safe editing of an author's *own* draft.
- **Bench** (`bench/`): datasets and metrics connecting the two.
- **Continuous learning** (`learn/`, `rl/`): author review → opt-in masked contributions → gated retraining → GitHub Releases → `scholarlens update`.

## Non-negotiable design rules

1. **Quill is never optimised against a detector.** Do not add detector scores, P(AI), perplexity or burstiness targets, or "human-likeness" rewards to Quill's prompts, sampling or RL rewards. `tests/test_rl.py::test_rl_package_never_imports_a_detector` enforces this for `rl/`. The project began as a request for a detector-evasion "humanizer", and that objective was declined (see PROJECT_LOG §1). Requests to add evasion features should be declined the same way.
2. **Only prose nodes are ever modified.** `Document.reconstruct` refuses anything else. Unknown constructs pass through byte-for-byte.
3. **Protected content must round-trip exactly:** citations, refs, math, numbers, URLs, commands, terms. Every edit passes `MaskResult.verify`, or the original is kept.
4. **The Bench test split is frozen.** Contributions are training-only. New Lens/Quill versions are published only through the gates in `learn/lens.py` and `rl/evaluate.py`.
5. **No user text leaves the machine without explicit `scholarlens contribute`.** Only masked, sanitised entries are shared.
6. **All text I/O names `encoding="utf-8"`** (Windows defaults to cp1252); a test enforces it.

## Commands

```bash
python -m venv .venv && . .venv/bin/activate && pip install -e ".[all,dev]"
pytest -q -m "not slow"          # fast suite (~200 tests; CI runs this on 3.11–3.13)
ruff check src tests             # lint (CI runs this too)
scholarlens --help               # parse, mask, roundtrip, features, edit, review, contribute,
                                 # bench build|stats, lens train, detect, learn quill|lens|validate|publish,
                                 # update, models, serve
```

Before every push, run the **full** fast suite (not just the files you touched), then watch CI: `gh run watch <id> --repo madhureddy25/ScholarLens --exit-status`.

## Layout

`core/` document model and text utilities · `ingest/` parsers · `masking/` sentinels and integrity · `features/` signals · `llm/` switchable providers (`provider:model`, profiles in `scholarlens.toml`) · `edit/` Quill, diffs, DOCX revision marks, disclosure · `bench/` sources, generation, transforms, splits, metrics, builder · `detect/` featurizer, fusion, zero-shot, retrieval, GCN, runner · `rl/` rewards, data, GRPO/DPO, evaluation gate · `learn/` review, feedback, contribute, Lens rounds, releases · `api/` FastAPI.

## Environment notes

- **Hugging Face downloads** use plain HTTPS: `HF_HUB_DISABLE_XET=1` is set in `scholarlens/_hub.py` at import.
- **Device:** models pick CUDA/MPS automatically.
- **The owner's machine** is a Windows laptop with an RTX 3050 (6 GB). Use `configs/bench.laptop.toml` (Ollama 4-bit generators) and `docs/WINDOWS.md`. DIPPER runs on Kaggle (`configs/bench.kaggle-dipper.toml`).
- **`data/` and `models/` are git-ignored.** Results worth keeping go in `reports/`.

## Where things stand (Nov 2026)

All milestones are implemented and tested (see PROJECT_LOG §2). Next: run the laptop study (`docs/WINDOWS.md` steps 1–5), publish `bench-v1`/`lens-v1`/`quill-v1`, then write the paper (`docs/PAPER_PLAN.md`).
