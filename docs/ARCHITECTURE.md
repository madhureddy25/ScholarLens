# ScholarLens — System Architecture

ScholarLens is one system with two products on a shared, integrity-preserving core:

| Product | Purpose | Who it serves |
|---|---|---|
| **Lens** (detect) | AI-text detection for academic prose that stays accurate under paraphrase attacks and does not penalise honest authors | Journals, reviewers, research-integrity teams, detector researchers |
| **Quill** (edit) | Citation-safe language editing of an author's *own* draft, with tracked changes and an AI-assistance disclosure log | Researchers, especially non-native English writers |
| **Bench** (research) | Benchmarks that connect the two: detector robustness under attack, and false-positive rates on human text before/after legitimate editing | The paper we publish |

## Design principles

1. **Never touch what we don't understand.** Only prose nodes are editable. Equations, tables, code, bibliography, preamble and *unknown* LaTeX environments pass through byte-for-byte. This is enforced in code (`Document.reconstruct` refuses non-prose replacements) and by tests.
2. **Facts are immutable.** Citations, cross-references, math, numbers/statistics, URLs and hard domain terms are masked behind sentinels before any model sees the text, and every edit is verified before it is accepted.
3. **Quill and Lens are independent.** Quill's objective is clarity and fidelity; it never takes a detector score as an input, reward, or stopping criterion, and Lens is never in Quill's loop. Lens evaluates Quill output only inside Bench, to *measure* whether legitimate editing gets honest authors falsely flagged.
4. **Disclosure is a feature, not an afterthought.** Every Quill session produces a machine-readable log of what was changed and by which model, suitable for a journal's AI-use statement.
5. **CPU-first, models pluggable.** Everything runs on a laptop CPU with small defaults (GPT-2, MiniLM). Larger models (Llama-3.x, SPECTER2, SciBERT) are drop-in via constructor arguments.

## Pipeline

```
                ┌──────────────────────── shared core ────────────────────────┐
 .tex .md .txt  │  Phase 1  ingest/        Phase 2  masking/                   │
 .docx .pdf ───▶│  parse → Document AST ─▶ mask → MaskResult (sentinels+map)   │
 (PDF via GROBID)│  typed nodes + spans     verify / repair / unmask           │
                └───────────────┬───────────────────────────┬─────────────────┘
                                │ prose units (masked)      │
              ┌─────────────────▼──────────┐   ┌────────────▼────────────────────┐
              │ Lens  features/ + detect/  │   │ Quill  edit/                     │
              │ • PPL, entropy, rank stats │   │ • rewrite masked prose (LLM)     │
              │ • Fast-DetectGPT score     │   │ • verify integrity, retry/repair │
              │ • burstiness, stylometry   │   │ • semantic-similarity gate       │
              │ • transition graph (δ_ij)  │   │ • tracked changes (DOCX/LaTeX)   │
              │ • fusion classifier / GNN  │   │ • disclosure log                 │
              │ • retrieval defence        │   │                                   │
              └─────────────┬──────────────┘   └────────────┬────────────────────┘
                            └──────────────┬────────────────┘
                                   ┌───────▼────────┐
                                   │ Bench  bench/  │  robustness + fairness reports
                                   └────────────────┘
```

## Module map

| Module | Status | Contents |
|---|---|---|
| `core/document.py` | ✅ | `Document`, `Node`, `NodeType`; span-based byte-exact reconstruction |
| `core/text.py` | ✅ | Academic-aware sentence splitter, word tokeniser, 250-word sentence-aligned windows |
| `ingest/latex.py` | ✅ | pylatexenc walker; sections, abstract, lists (nested), floats + captions, verbatim blanking, comments |
| `ingest/markdown.py` | ✅ | Markdown and plain text (heuristic headings, references detection) |
| `ingest/docx.py` | ✅ read | Paragraph styles, tables, OMML detection, protected runs (italic/sub/superscript, hyperlinks, Zotero/EndNote fields) |
| `ingest/tei.py` | ✅ | GROBID client + TEI parser (`<ref>` → protected spans) |
| `masking/` | ✅ | Sentinel masking for CITE/REF/MATH/NUM/URL/CMD/TERM, glossary, SciSpacy soft-term verification, integrity report, damaged-sentinel repair, scoring render |
| `features/stylometry.py` | ✅ | Burstiness, sentence-length stats, MATTR, connective and marker rates |
| `features/transitions.py` | ✅ | T(D), δ_ij, Var(T(D)) features, sentence-relation graph |
| `features/lm.py` | ✅ | Sliding-window PPL, entropy dynamics, rank stats, Fast-DetectGPT |
| `analysis.py` / `cli.py` | ✅ | Document → segments → features; `scholarlens parse/mask/roundtrip/features` |
| `bench/` | ⏳ M2 | Dataset builder (human / AI / AI-paraphrased / human-edited), splits, metrics |
| `detect/` | ⏳ M3 | Feature-fusion classifier, fine-tuned encoder, GNN on sentence graphs, retrieval defence, calibration |
| `llm/` | ✅ | Provider-neutral `ChatModel`; Anthropic (official SDK, refusal fallbacks), any OpenAI-compatible endpoint (OpenAI, Ollama, vLLM, LM Studio, OpenRouter, Groq, …), in-process Hugging Face; TOML profiles; plugin entry points |
| `edit/` | ✅ | Quill engine (mask → generate → repair → verify → similarity gate → retry), LaTeX tracked changes, HTML review, disclosure manifest, `scholarlens edit` |
| `edit/` DOCX writer | ⏳ | Write accepted edits back into .docx as Word revision marks (w:ins / w:del) |
| `api/` | ⏳ M5 | FastAPI service + minimal web UI |

## Changes relative to the original blueprint

| Blueprint | ScholarLens | Why |
|---|---|---|
| Mask every SciSpacy entity | Two tiers: **hard-mask** high-precision terms (gene symbols, acronyms, formulas, glossary); **verify-only** for broad NER concepts | Masking "double-strand breaks" makes prose unreadable for an editor; verification keeps the 100% term-fidelity guarantee without wrecking fluency |
| Citations + entities only | Also **numbers and statistics** (`p < 0.05`, `n = 1,200`, `3.2 ± 0.4`), URLs/DOIs, LaTeX commands, accents, comments | In science a changed number is worse than a changed word |
| Sentinels as new tokenizer vocabulary | `⟦KIND_n⟧` Unicode sentinels + verifier + repair | Works with hosted LLM APIs where vocabulary cannot be extended; machine-checkable |
| "AST reconstruction" | Span splicing over the original source | Guarantees non-prose bytes are identical (property-tested) |
| — | DOCX formatting runs and reference-manager fields protected | Italic species names, superscripts and live Zotero citations survive editing |
| Success = low scores on commercial detectors | Lens: AUROC and TPR@1%FPR on clean **and attacked** text, FPR on human subgroups. Quill: integrity, fidelity, fluency | Measures what matters for each product; no evasion objective |

## Sampling policy

Quill uses each backend's fixed, conservative settings (low temperature where a backend exposes one; current Claude models choose their own sampling and take an `effort` level instead). Retries add specific feedback about what failed. They do not escalate temperature. The blueprint's entropy-targeted dynamic temperature (raising temperature when local entropy is low) is deliberately **not** implemented: its purpose was to shift detector statistics, which is outside Quill's objective.

## Evaluation targets

**Lens (detector)**
- AUROC and TPR @ 1% FPR on held-out human vs. AI academic text, per domain (CS, bio, physics, social science).
- Same metrics under attack: DIPPER (L/O ∈ {20, 40, 60}), recursive paraphrasing, and off-the-shelf rewriting tools. Report the drop, not just the clean number.
- **False-positive rate on human text**, sliced by author L1 proxy and by whether the text was language-edited (Quill, Grammarly-style tools). Target: no subgroup above 1% FPR at the operating threshold.
- Calibration (ECE) of segment and document scores; scores are reported with uncertainty, never as a verdict.

**Quill (editor)**
- Citation, reference, math and number integrity: 100% (hard gate; edits that fail are retried, then dropped).
- Term fidelity: 100% for hard terms; soft-term retention reported.
- Semantic similarity to the author's text ≥ 0.94 (SPECTER2 / SciBERT cosine) and BERTScore-F1.
- Human evaluation: fluency and faithfulness ratings by domain experts on a sample.

## Data plan (M2)

- **Human text:** abstracts and paragraphs from S2ORC / arXiv / PubMed Central **published before 2022**, to avoid contamination by LLM-assisted writing.
- **AI text:** generations from several model families conditioned on the same titles/outlines, with varied prompts and temperatures.
- **Attacked AI text:** DIPPER and other paraphrasers applied to the AI text (threat model for robustness).
- **Edited human text:** Quill and other grammar tools applied to human text (for the false-positive study).
- Splits by *paper*, never by paragraph, so no document leaks across train/test.

## Roadmap

| Milestone | Scope | Status |
|---|---|---|
| M0 | Repo, packaging, CI (GitHub Actions), tests | ✅ |
| M1 | Shared core: ingestion (LaTeX/MD/TXT/DOCX/TEI) + masking + integrity verification + feature extraction | ✅ |
| M2 | Bench data builder + metrics | next |
| M3 | Lens detectors + robustness study | |
| M4 | Quill editor + tracked changes + disclosure | ✅ (DOCX revision marks pending) |
| M5 | API, web UI, docs | |
| M6 | Paper write-up | |
