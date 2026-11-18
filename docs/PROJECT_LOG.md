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

| 10 | **The disclosure statement omitted the Quill-RL adapter**, so it would name only the base model. Found in the end-to-end check. | The statement and model name now include the adapter, e.g. "…with the fine-tuned adapter 'quill-rl-v1'". |
| 11 | **Kaggle rejected the benchmark zip** ("forbidden character in name ('\\')"): `Compress-Archive` in Windows PowerShell 5.1 writes backslash paths. | Runbooks zip with `C:\Windows\System32\tar.exe -a -c -f`. |
| 12 | **Kaggle moved dataset mounts** to `/kaggle/input/datasets/<user>/<slug>/`, so `cp -r /kaggle/input/*/laptop-v1` found nothing. | The notebook cell locates `laptop-v1` with `find`. |
| 13 | **Kaggle notebook usability:** bitsandbytes prints `MatMul8bitLt: inputs will be cast…` on every forward pass, burying progress; the circular-arrow toolbar button is a *factory reset* that deleted a running session's files. | Output piped through `grep -v MatMul8bitLt`; runbooks warn about the button. DIPPER took about 9.5 h for 1,000 texts on a T4 x2 (12 h limit), not 3–4 h. |
| 14 | **Windows torch silently became the CPU build** (`2.15.0+cpu`): `pip install -e ".[all,dev]"` after the `cu126` install replaced it. Quill-RL ran at about 500 s/step on the CPU (about 30 s on the GPU). | Runbooks reinstall the `cu126` build *after* the package and check `torch.__version__`; `learn quill` now warns when torch sees no GPU. |
| 15 | **Isotonic ties understated Lens-fusion's TPR@1% FPR.** Calibrated scores are a step function; when the threshold lands on a plateau, every AI text on it counts as missed (`lens-no-edits` showed clean-human FPR 0.000 instead of 0.007). | Test-split metrics and permutation importance use `FusionDetector.rank_scores` (uncalibrated, same order, no ties); `predict_proba` stays calibrated for users. `scholarlens lens rescore` rewrites old reports from cached features. |
| 16 | **DIPPER artifacts:** the T5 vocabulary drops Greek letters and some symbols (`ΛCDM`→`CDM`, `θ-(ET)2X`→`-(ET)2X`); a few outputs repeat a sentence; 1 of 500 L60_O60 outputs is junk ("Cf. (a) and (b) of the preceding note.", train split). | Kept: these are properties of the attack as published. Worth one sentence in the paper. |
| 17 | **Quill-RL without a quality reward converges to restraint:** v1 changed 0.5% of text; on the sample paper 5 paragraphs were unchanged, 1 barely changed, and 1 dropped `⟦CITE_1⟧` and was rejected by the integrity gate. | Expected from the reward design. With an LLM judge (v2), quality rose but integrity fell and the gate rejected it (§5b). |

## 4b. End-to-end verification (CPU, real models, nothing published)

Every user-facing path was run for real on the pilot data, in addition to the 190 unit tests and 3 slow tests (real GPT-2, a real Qwen Quill edit, a real GRPO step):

| Path | Result |
|---|---|
| `edit` with Qwen2.5-0.5B → `review --accept-all` → `contribute` → `learn validate --strict` | ✅ Final paper written. The contribution file has only masked text: no citation keys, paths or endpoints, and the private `% TODO` comment was masked. |
| `learn lens` round 1 (no current model) | ✅ Accepted, exit 0; 2 contributions became 4 training-only human examples |
| package → install (checksum, safe extract) → `detect` with no path | ✅ Archive excluded the feature cache; release notes and table generated; installed model used automatically |
| `learn lens` round 2 against the installed model | ✅ Hard-example mining (5); candidate slightly worse, so **correctly rejected**, exit 3 (CI won't publish) |
| `lens train --binoculars` | ✅ Binoculars (gpt2/distilgpt2) in every table |
| API `/v1/detect` with a real model | ✅ Same P(AI) as the CLI; empty input handled |
| `learn quill`: GRPO → DPO (2 author pairs) → evaluate base vs adapter → gate | ✅ 1.6 min on CPU; 2 steps don't improve it, so **correctly rejected**. The base 0.5B model keeps all placeholders in only 75% of paragraphs, which is what the integrity reward targets. |
| `edit -m quill:<adapter>` | ✅ Adapter loads; disclosure records it (see finding 10) |

## 5. CPU pilot result (pipeline check only)

`configs/bench.pilot-cpu.toml` used 120 real pre-2022 arXiv abstracts, AI counterparts from Qwen2.5-0.5B and SmolLM2-360M, 60 LLM-paraphrase attacks and 40 Quill-proofread human abstracts. That's 340 records with no leakage. The full report is `reports/pilot-cpu/report.md`.

**These numbers are not results.** The test split has only 17 texts (7 human), so TPR at 1% FPR is decided by a single human text, and the generators are tiny. The pilot shows that every stage runs end to end on real data:
- **Retrieval** separated paraphrases perfectly, which is expected since the index holds their sources.
- **Fast-DetectGPT, perplexity, the GCN and fusion** all ran.
- **The top fusion features** are now style signals (Fast-DetectGPT, MATTR, top-10 rank fraction, function-word ratio), not length proxies.

## 5b. Laptop study (Nov 2026)

Reports are in `reports/` (`lens-v1.md`, `lens-unseen-dipper.md`, `lens-no-edits.md`, `quill-rl-v1.json`, `quill-rl-v2.json`, `bench-manifest.json`). Published: `bench-v1`, `lens-v1`.

**Bench `laptop-v1`:** 1,500 pre-2022 arXiv abstracts, 1,500 matched AI abstracts (Qwen2.5-7B, Phi-3.5, Llama-3.2-3B; 500 each), and 500 each of `llm_paraphrase`, `dipper_L20_O0`, `dipper_L60_O60`, `quill_proofread` and `quill_clarity`: 5,500 records, no leakage. Test split: 145 human, 145 AI, 44 per attack, 46 per edit family.

**Lens (test split, AUROC / TPR@1% FPR).** These TPRs were computed before finding 15 and are conservative for lens-fusion; rerun `lens rescore` before quoting them. AUROC is essentially unaffected.

| detector | ai | dipper_L20_O0 | dipper_L60_O60 | llm_paraphrase |
|---|---|---|---|---|
| lens-fusion | 0.990 / 0.88 | 0.964 / 0.82 | 0.959 / 0.48 | 0.968 / 0.70 |
| lens-gnn | 0.965 / 0.50 | 0.962 / 0.43 | 0.916 / 0.14 | 0.953 / 0.39 |
| fast_detectgpt | 0.891 / 0.36 | 0.910 / 0.27 | 0.918 / 0.27 | 0.686 / 0.05 |
| retrieval (oracle: index holds the test sources) | 1.000 / 1.00 | 0.999 / 0.95 | 0.891 / 0.61 | 1.000 / 1.00 |

- **Fairness:** at the clean-human operating point (1/145 flagged by construction), Quill-edited human text was flagged 1/46 by lens-fusion, the same as fast_detectgpt; 0/46 by lens-gnn; up to 3/46 by neg_log_rank.
- **Unseen attack** (`lens-unseen-dipper`, L60_O60 held out of training): L60_O60 AUROC 0.959 → 0.893 (below fast_detectgpt's 0.918), TPR 0.48 → 0.39 (still above fast_detectgpt's 0.27). Adversarial training helps, but seeing the attack matters. The other columns rose (clean 0.88 → 0.95, llm_paraphrase 0.70 → 0.84); recheck after rescoring before reading this as a trade-off.
- **No edits in training** (`lens-no-edits`): AUROC unchanged (within 0.006) and edited-human FPR 0/46 either way. At this sample size there is no evidence that training on Quill edits lowers false positives; Lens did not flag these edits even without them. Its lower TPRs coincide with the tie problem (clean-human FPR 0.000), so they are not interpretable until rescored.
- **Top fusion features:** `llm_marker_per_1k`, `lm_fast_detectgpt`, `avg_word_len`. The marker-word feature may not transfer to other generators; a leave-one-generator-out run is future work.
- **Sample sizes are small** (44 per attack: a TPR of 0.48 has a margin of about ±0.15). Use the bootstrap `auroc_ci95` in `report.json`.

**Quill-RL** (Qwen2.5-0.5B-Instruct, GRPO, 200 steps, frozen eval set of 60 paragraphs):

| metric | base | v1 (no quality reward) | v2 (`--judge ollama:qwen2.5:3b`) |
|---|---|---|---|
| integrity_rate | 0.633 | **0.717** | 0.667 |
| format_rate | 0.983 | 1.000 | 1.000 |
| mean_similarity | 0.988 | 0.995 | 0.991 |
| mean_quality (judge) | – | 0.233 | **0.550** |
| mean_change | 0.037 | 0.005 | 0.023 |
| gate | – | accepted vs base | **rejected** vs v1 (integrity and reward lower) |

v1: 1.8 h on the RTX 3050. v2: 5.9 h (the judge scores every candidate). The story: integrity-only RL learns restraint; adding a quality judge produces real edits at a small integrity cost, and the gate refuses the regression. `quill-v1` was not published, because it barely edits.

## 6. Hardware notes

- **Development container:** 4 CPUs, no GPU.
- **Owner's laptop:** NVIDIA RTX 3050 Laptop GPU (6 GB), Ryzen 5 7235HS, 24 GB RAM, Windows.
  - Generators run through Ollama in 4-bit.
  - Quill-RL uses a 0.5B model, or 1.5B with `--4bit`.
  - DIPPER (11B) doesn't fit, so it runs on a free Kaggle T4 in 8-bit.
- **Runbooks:** `docs/WINDOWS.md` (PowerShell), `docs/LAPTOP.md`, `docs/GPU.md` (24 GB cards).

## 7. Next steps

1. ~~Build the laptop benchmark, run DIPPER, train Lens and Quill-RL, publish `bench-v1`/`lens-v1`, commit `reports/`.~~ Done (§5b). `learn.yml` was triggered once and now runs weekly.
2. On the laptop: `scholarlens lens rescore data/bench/laptop-v1/bench.jsonl models/lens-v1 models/lens-unseen-dipper models/lens-no-edits`, then copy the three `report.md` files into `reports/` again (finding 15).
3. Write the paper (`docs/PAPER_PLAN.md`) from `reports/`.
4. Quill: a stronger integrity penalty, or LanguageTool as the quality reward, to get edits without the integrity loss seen in v2.
5. Later work:
   - **Web UI** for review.
   - **PubMed Central / S2ORC sources,** for domains beyond arXiv.
   - **A non-native-author subset** for the fairness analysis.
   - **A self-hosted GPU runner,** so Quill-RL can retrain automatically.
   - **Sharper DIPPER settings.**
