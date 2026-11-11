# Lens evaluation — data/bench/pilot-cpu/bench.jsonl

> CPU pilot: pipeline check only (17 test texts, tiny generators). Not a result. See reports/README.md.

Test split. Threshold per detector set so clean human text has FPR ≤ 1%.

## AI detection under attack (AUROC / TPR@1% FPR)

| detector | ai | ai+llm_paraphrase |
|---|---|---|
| lens-fusion | 0.929 / 0.00 | 0.929 / 0.00 |
| fast_detectgpt | 0.816 / 0.43 | 0.929 / 0.50 |
| neg_log_ppl | 0.837 / 0.57 | 0.929 / 0.50 |
| neg_log_rank | 0.857 / 0.57 | 0.929 / 0.50 |
| neg_entropy | 0.694 / 0.29 | 0.357 / 0.00 |
| retrieval | 1.000 / 1.00 | 1.000 / 1.00 |
| lens-gnn | 0.918 / 0.86 | 0.714 / 0.50 |

## False-positive rate on human text (lower is better)

| detector | human | human+quill_proofread |
|---|---|---|
| lens-fusion | 0.000 (n=7) | 0.000 (n=1) |
| fast_detectgpt | 0.000 (n=7) | 0.000 (n=1) |
| neg_log_ppl | 0.000 (n=7) | 0.000 (n=1) |
| neg_log_rank | 0.000 (n=7) | 0.000 (n=1) |
| neg_entropy | 0.000 (n=7) | 0.000 (n=1) |
| retrieval | 0.000 (n=7) | 0.000 (n=1) |
| lens-gnn | 0.000 (n=7) | 0.000 (n=1) |

## Top fusion features (permutation importance, ΔAUROC)

- `lm_fast_detectgpt`: +0.0555
- `mattr`: +0.0429
- `lm_top10_frac`: +0.0379
- `function_word_ratio`: +0.0311
- `llm_marker_per_1k`: +0.0240
- `tr_local_global_ratio`: +0.0238
- `lm_log_ppl`: +0.0199
- `lm_log_rank_mean`: +0.0195
- `tr_delta_abs_mean`: +0.0188
- `tr_offdiag_sim_mean`: +0.0188
- `lm_top1_frac`: +0.0153
- `tr_delta_adj_std`: +0.0149

Training: {"n_train": 279, "n_val": 44, "excluded_families": [], "families": ["ai", "ai+llm_paraphrase", "human", "human+quill_proofread"], "featurizer": "lm=gpt2|emb=sentence-transformers/all-MiniLM-L6-v2", "estimator": "hgb", "calibration": "isotonic"}
