# ScholarLens: project log

This is the record of how ScholarLens was designed and built: what was decided and why, what went wrong and how it was fixed, and what comes next. It's written so the project can continue without the original conversation. Built with Claude Code (Claude Opus 5.5) in November 2026.

## 1. Origin and scope decision

The project started from a research blueprint, *"System Architecture and Implementation Roadmap for Academic-Grade AI Text Humanization"*. It proposed a six-stage pipeline to rewrite AI-generated scientific text so that it passes Turnitin, GPTZero and Pangram as human-written. Its targets were a Turnitin AI score under 5–15% and a Turnitin "AI Bypasser" pass rate over 98%. It also proposed a Sem-DPO stage rewarding "surrogate AI score < 10%" and a closed loop that re-paraphrases until every detector is beaten.

**That evasion objective was not built.** Put together, it makes a tool for passing off AI-written papers as human-written to universities and journals. Two specific mechanisms were explicitly left out:
- **Detector-score rewards** for the rewriter.
- **Entropy-targeted "dynamic temperature" sampling,** whose only purpose was to shift detector statistics.

A later proposal to train Quill with RL against Lens (reward = 1 − P_AI, Quill-vs-Lens self-play) was declined for the same reason. It would also make the fairness study circular, because Quill-edited human text is the honest-editing baseline.

Two of the blueprint's citations are worth flagging:
- **Sem-DPO (refs 10/15)** is a prompt-engineering paper, and applying it here was the blueprint's own extrapolation.
- **Refs 6 and 8,** which back most of its Turnitin claims, are an evasion vendor's blog and a GitHub discussion thread.

**What was built instead** reuses the blueprint's sound technical parts (document parsing, citation/entity masking, the statistical and structural signals) in three products:
- **Lens:** AI-text detection for academic prose that is robust to paraphrase attacks and audited for false positives on honest authors.
- **Quill:** a citation-safe language editor for an author's own draft, with tracked changes and an AI-use disclosure record.
- **Bench:** the benchmark that connects them. Its central research question is whether legitimate language editing gets honest authors, especially non-native English writers, falsely flagged. Detectors are known to be biased against non-native writers (Liang et al., 2023).

## 2. Timeline

| Commit | Milestone |
|---|---|
| `7938186` | Shared core: LaTeX/Markdown/DOCX/GROBID-TEI parsing with byte-exact reconstruction; masking of citations, references, math, numbers, URLs, commands and domain terms; integrity verification; features (PPL, entropy, burstiness, transitions, Fast-DetectGPT) |
| `81b450a` | Switchable model layer (Anthropic, any OpenAI-compatible endpoint, local Hugging Face) and Quill (M4) |
| `beac160` | Bench (M2): pre-2022 arXiv source, matched AI generation, attacks, paper-level splits, metrics |
| `b5fd000` | Lens (M3): calibrated fusion classifier, zero-shot baselines, retrieval defence, sentence-graph GCN |
| `2953852` | Quill writes `.docx` edits as Word revision marks |
| `5628b60` | HTTP API (FastAPI) and Dockerfile (M5) |
| `81addae` | Generated-text cleaning (label-leak fix, see §4) |
| `4ba65b8` | Provider-registry bug fix (found by CI) |
| `fc8e37f` | GPU study config, single-GPU generator scheduling, length-feature fix |
| `d515c1f` | Author review, local feedback store, opt-in contributions |
| `d46610e` | Quill-RL: GRPO with checkable rewards, DPO from author decisions |
| `0b776fd` | Continuous learning: gated Lens retraining, GitHub Releases, `update`, `learn.yml` |
| `9f06791` | 6 GB laptop path (Ollama), Kaggle DIPPER, learning docs |
| `27eabad` | Windows UTF-8 robustness, Kaggle notebook fix |

## 3. Key design decisions

The full table of changes from the blueprint is in `ARCHITECTURE.md`. In short:

1. **Byte-exact span reconstruction** instead of a custom AST round-trip. Only prose nodes can be replaced, and every byte outside an edited paragraph is identical to the original. Unknown LaTeX environments pass through untouched.
2. **Numbers and statistics are masked too.** A changed number is worse than a changed word.
3. **Two-tier terms.** High-precision terms (gene symbols, acronyms, formulas, glossary entries) are masked. Broad NER concepts are only *verified* after editing, because masking them would make the prose unreadable to an editor.
4. **Unicode sentinels `⟦KIND_n⟧` with a verifier and repair step.** These work with hosted LLM APIs, where custom vocabulary tokens aren't available.
5. **Quill gates every edit.** Every placeholder must come back exactly once, LaTeX braces must balance, and semantic similarity must be ≥ 0.94. Otherwise Quill retries with specific feedback (up to 3 attempts), then keeps the original. Retries never raise temperature.
6. **Models are switchable** with one string (`anthropic:claude-opus-5-5`, `ollama:qwen2.5:7b`, `hf:...`, `quill:latest`), with TOML profiles and plugin entry points. On Claude, server-side refusal fallbacks are on by default (`--opt fallbacks=false` turns them off).
7. **Bench splits by paper,** never by paragraph, and checks for leakage. Human text must predate 2022.
8. **Lens drops length-dependent features,** so the classifier can't shortcut on "AI abstracts are shorter".
9. **Quill-RL rewards are checkable and detector-free:** integrity gate, fidelity, grammar errors actually fixed (LanguageTool), an optional LLM judge for clarity, edit economy, and format. `tests/test_rl.py` fails if `rl/` imports the detector, the LM scorer or bench metrics.
10. **Continuous learning has guard rails:**
    - **Feedback is local and masked,** and only leaves the machine through opt-in `contribute`.
    - **Contributions are training-only;** the test split is frozen.
    - **A new Lens must not lower AUROC on any attack family nor raise any human group's false-positive rate.**
    - **Releases carry SHA-256 checksums,** and `update` verifies them and rejects unsafe archive paths.

## 4. Findings, bugs and fixes (worth reporting)

| # | What happened | Fix |
|---|---|---|
| 1 | **Real-model Quill run (Qwen2.5-0.5B):** a stray `</paragraph>` tag leaked into the output, and `scoring~\cite{}` became `scoring ~\cite{}` (a double gap in the PDF). | Wrapper-tag stripping plus a hard check; spaces before `~`-led spans are removed on unmasking. |
| 2 | **Label leak from formatting:** small generators emitted `**Abstract**` headers, wrapping quotes and `...`. A detector would learn "formatting = AI". | `clean_generation()` runs on all generated and paraphrased text, and again at assembly. |
| 3 | **Length leak:** in the pilot, the top features were raw TTR and hapax ratio. Both fall mechanically with length, and generated abstracts were about 20% shorter (median 114 vs 139 words). | Dropped TTR and hapax; MATTR is kept. |
| 4 | **RL reward hack:** rewriting the whole paragraph "fixed" every grammar error and scored higher than leaving it alone. | A steep edit-economy penalty, so wholesale rewriting always scores below restraint. |
| 5 | **GRPO zero signal:** truncated completions all fail integrity, so the group variance is 0 and no learning happens (`frac_reward_zero_std = 1` in a CPU smoke run). | Paragraphs that can't finish within `max_completion_length` are skipped. |
| 6 | **CI caught a registry bug:** registering a custom provider before the first load hid all built-in providers. | Built-in loading uses its own flag; regression test added. |
| 7 | **Builder dropped transforms from earlier configs.** It would have lost the laptop's attacks after a separate Kaggle DIPPER run. | Assembly merges every cached `transforms/*.jsonl`. |
| 8 | **Windows cp1252 default encoding** would crash on `⟦⟧`, `≤` and `—`. | All text I/O names UTF-8; a test enforces it. |
| 9 | **Hugging Face downloads behind a proxy:** the Xet protocol and CDN redirects failed. | `HF_HUB_DISABLE_XET=1`, set at package import. Hosts needed: huggingface.co, us.aws.cdn.hf.co, eu.aws.cdn.hf.co, cdn-lfs*.hf.co, download.pytorch.org. |

## 5. CPU pilot result (pipeline check only)

`configs/bench.pilot-cpu.toml` used 120 real pre-2022 arXiv abstracts, AI counterparts from Qwen2.5-0.5B and SmolLM2-360M, 60 LLM-paraphrase attacks and 40 Quill-proofread human abstracts. That's 340 records with no leakage. The full report is `reports/pilot-cpu/report.md`.

**These numbers are not results.** The test split has only 17 texts (7 human), so TPR at 1% FPR is decided by a single human text, and the generators are tiny. The pilot shows that every stage runs end to end on real data:
- **Retrieval** separated paraphrases perfectly, which is expected since the index holds their sources.
- **Fast-DetectGPT, perplexity, the GCN and fusion** all ran.
- **The top fusion features** are now style signals (Fast-DetectGPT, MATTR, top-10 rank fraction, function-word ratio), not length proxies.

## 6. Hardware notes

- **Development container:** 4 CPUs, no GPU.
- **Owner's laptop:** NVIDIA RTX 3050 Laptop GPU (6 GB), Ryzen 5 7235HS, 24 GB RAM, Windows.
  - Generators run through Ollama in 4-bit.
  - Quill-RL uses a 0.5B model, or 1.5B with `--4bit`.
  - DIPPER (11B) doesn't fit, so it runs on a free Kaggle T4 in 8-bit.
- **Runbooks:** `docs/WINDOWS.md` (PowerShell), `docs/LAPTOP.md`, `docs/GPU.md` (24 GB cards).

## 7. Next steps

1. Build the laptop benchmark, run DIPPER on Kaggle, merge (`docs/WINDOWS.md` steps 1–2).
2. Train Lens: the main run plus held-out-attack and no-edit-training ablations. Train Quill-RL (step 3).
3. Publish `bench-v1`, `lens-v1`, `quill-v1`, then trigger `learn.yml` (step 4).
4. Commit the reports to `reports/` and write the paper (`docs/PAPER_PLAN.md`).
5. Later work:
   - **Web UI** for review.
   - **PubMed Central / S2ORC sources,** for domains beyond arXiv.
   - **A non-native-author subset** for the fairness analysis.
   - **A self-hosted GPU runner,** so Quill-RL can retrain automatically.
   - **Sharper DIPPER settings.**
