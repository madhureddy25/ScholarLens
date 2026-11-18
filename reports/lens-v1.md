# Lens evaluation — data\bench\laptop-v1\bench.jsonl

Test split. Threshold per detector set so clean human text has FPR ≤ 1%.

## AI detection under attack (AUROC / TPR@1% FPR)

| detector | ai | ai+dipper_L20_O0 | ai+dipper_L60_O60 | ai+llm_paraphrase |
|---|---|---|---|---|
| lens-fusion | 0.990 / 0.88 | 0.964 / 0.82 | 0.959 / 0.48 | 0.968 / 0.70 |
| fast_detectgpt | 0.891 / 0.36 | 0.910 / 0.27 | 0.918 / 0.27 | 0.686 / 0.05 |
| neg_log_ppl | 0.920 / 0.13 | 0.868 / 0.11 | 0.822 / 0.05 | 0.799 / 0.00 |
| neg_log_rank | 0.916 / 0.24 | 0.860 / 0.25 | 0.818 / 0.14 | 0.766 / 0.02 |
| neg_entropy | 0.798 / 0.01 | 0.672 / 0.00 | 0.524 / 0.00 | 0.726 / 0.00 |
| retrieval | 1.000 / 1.00 | 0.999 / 0.95 | 0.891 / 0.61 | 1.000 / 1.00 |
| lens-gnn | 0.965 / 0.50 | 0.962 / 0.43 | 0.916 / 0.14 | 0.953 / 0.39 |

## False-positive rate on human text (lower is better)

| detector | human | human+quill_clarity | human+quill_proofread |
|---|---|---|---|
| lens-fusion | 0.007 (n=145) | 0.022 (n=46) | 0.022 (n=46) |
| fast_detectgpt | 0.007 (n=145) | 0.022 (n=46) | 0.022 (n=46) |
| neg_log_ppl | 0.007 (n=145) | 0.043 (n=46) | 0.022 (n=46) |
| neg_log_rank | 0.007 (n=145) | 0.065 (n=46) | 0.065 (n=46) |
| neg_entropy | 0.007 (n=145) | 0.022 (n=46) | 0.022 (n=46) |
| retrieval | 0.007 (n=145) | 0.022 (n=46) | 0.022 (n=46) |
| lens-gnn | 0.007 (n=145) | 0.000 (n=46) | 0.000 (n=46) |

## Top fusion features (permutation importance, ΔAUROC)

- `llm_marker_per_1k`: +0.0448
- `lm_fast_detectgpt`: +0.0388
- `avg_word_len`: +0.0264
- `lm_logprob_std`: +0.0060
- `function_word_ratio`: +0.0051
- `sent_len_min`: +0.0048
- `tr_adj_sim_max`: +0.0045
- `lm_top1_frac`: +0.0044
- `tr_delta_adj_mean`: +0.0043
- `lm_entropy_mean`: +0.0039
- `lm_log_ppl`: +0.0038
- `sent_len_mean`: +0.0038

Training: {"n_train": 4311, "n_val": 675, "excluded_families": [], "families": ["ai", "ai+dipper_L20_O0", "ai+dipper_L60_O60", "ai+llm_paraphrase", "human", "human+quill_clarity", "human+quill_proofread"], "featurizer": "lm=gpt2|emb=sentence-transformers/all-MiniLM-L6-v2", "estimator": "hgb", "calibration": "isotonic"}
