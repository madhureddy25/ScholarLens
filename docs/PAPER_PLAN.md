# Paper plan

**Working title:** *Fair and Robust Detection of Machine-Generated Scientific Prose: Paraphrase Attacks, Honest Editing, and a Citation-Safe Editing Assistant*

## Research questions

1. **Robustness.** How much do zero-shot detectors (Fast-DetectGPT, perplexity, Binoculars) and a structure-aware fusion detector (Lens) lose under paraphrase attacks (DIPPER L/O ∈ {20/0, 60/60}, LLM paraphrase)? Does the retrieval defence recover what they lose?
2. **Generalisation.** If an attack family is held out of training (`--exclude ai+dipper_L60_O60`), how much of the robustness survives?
3. **Fairness.** Does legitimate language editing of human text (Quill proofread/clarity, LanguageTool) raise the false-positive rate? Does including edited human text in training (`lens-v1` vs `lens-no-edits`) remove that penalty without lowering detection?
4. **Integrity-preserving editing.** With checkable, detector-free rewards (GRPO + DPO), can a small open model (0.5–1.5B) become a reliable citation-safe editor? Measure placeholder integrity, meaning preservation and grammar errors fixed.

## Data (Bench)

- **Human:** arXiv abstracts from before 2022 (2021 snapshot, CC0), split by paper.
- **AI:** matched abstracts (same title, field and length) from Qwen2.5-7B, Phi-3.5 and Llama-3.2-3B, using two prompt styles.
- **Attacks:** DIPPER (Kaggle, 8-bit) and LLM paraphrase.
- **Edits:** Quill proofread and clarity, plus LanguageTool.
- **Report:** counts from `reports/bench-manifest.json`.

## Method

- **Lens:**
  - **Features:** stylometry (burstiness, MATTR, connectives, marker words), sentence-transition structure (T(D), δᵢⱼ, Var), and LM statistics (PPL, entropy dynamics, ranks, Fast-DetectGPT).
  - **Classifier:** calibrated gradient boosting, plus a sentence-graph GCN.
  - **Length control:** length features are removed (§4 of PROJECT_LOG).
- **Quill:** masking and verification pipeline, similarity gate, retries, disclosure manifest. Quill-RL rewards are described in `src/scholarlens/rl/rewards.py`.

## Tables and figures

| # | Content | Source |
|---|---|---|
| T1 | Bench composition per family × split | `reports/bench-manifest.json` |
| T2 | AUROC / TPR@1%FPR per detector × attack | `reports/lens-v1.md` |
| T3 | Held-out attack generalisation | `reports/lens-unseen-dipper.md` vs `lens-v1.md` |
| T4 | FPR on human families (clean, Quill, LanguageTool) per detector, with and without edited-human training | `reports/lens-v1.md`, `reports/lens-no-edits.md` |
| T5 | Quill-RL: integrity rate, similarity, quality, change ratio (base vs RL adapter) | `reports/quill-rl-v1.json` |
| F1 | System diagram | `docs/ARCHITECTURE.md` |
| F2 | Feature importance (permutation ΔAUROC) | `report.md` "Top fusion features" |

## Threats to validity (report them honestly)

- **Small generators.** Laptop-scale generators are weaker than frontier models, so detection may look easier than in practice.
- **One source corpus.** arXiv only for now; PubMed and S2ORC would extend it to other fields.
- **Reward hacking.** Two artifacts were caught and fixed: generated-text formatting, and text length through TTR/hapax. These show how easily such benchmarks leak labels.
- **Calibration.** Probabilities are calibrated on Bench and may not transfer to other text. Detector scores should never be used alone to judge an author.
