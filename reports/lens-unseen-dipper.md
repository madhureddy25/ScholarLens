# Lens evaluation — data\bench\laptop-v1\bench.jsonl

Test split. Threshold per detector set so clean human text has FPR ≤ 1%.

## AI detection under attack (AUROC / TPR@1% FPR)

| detector | ai | ai+dipper_L20_O0 | ai+dipper_L60_O60 | ai+llm_paraphrase |
|---|---|---|---|---|
| lens-fusion | 0.994 / 0.95 | 0.969 / 0.82 | 0.893 / 0.39 | 0.982 / 0.84 |
| fast_detectgpt | 0.891 / 0.36 | 0.910 / 0.27 | 0.918 / 0.27 | 0.686 / 0.05 |
| neg_log_ppl | 0.920 / 0.13 | 0.868 / 0.11 | 0.822 / 0.05 | 0.799 / 0.00 |
| neg_log_rank | 0.916 / 0.24 | 0.860 / 0.25 | 0.818 / 0.14 | 0.766 / 0.02 |
| neg_entropy | 0.798 / 0.01 | 0.672 / 0.00 | 0.524 / 0.00 | 0.726 / 0.00 |
| retrieval | 1.000 / 1.00 | 0.999 / 0.95 | 0.891 / 0.61 | 1.000 / 1.00 |

## False-positive rate on human text (lower is better)

| detector | human | human+quill_clarity | human+quill_proofread |
|---|---|---|---|
| lens-fusion | 0.007 (n=145) | 0.000 (n=46) | 0.000 (n=46) |
| fast_detectgpt | 0.007 (n=145) | 0.022 (n=46) | 0.022 (n=46) |
| neg_log_ppl | 0.007 (n=145) | 0.043 (n=46) | 0.022 (n=46) |
| neg_log_rank | 0.007 (n=145) | 0.065 (n=46) | 0.065 (n=46) |
| neg_entropy | 0.007 (n=145) | 0.022 (n=46) | 0.022 (n=46) |
| retrieval | 0.007 (n=145) | 0.022 (n=46) | 0.022 (n=46) |

## Top fusion features (permutation importance, ΔAUROC)

- `lm_fast_detectgpt`: +0.0550
- `llm_marker_per_1k`: +0.0481
- `avg_word_len`: +0.0281
- `sent_len_min`: +0.0079
- `tr_adj_sim_max`: +0.0052
- `lm_log_ppl`: +0.0044
- `lm_logprob_std`: +0.0042
- `sent_len_mean`: +0.0041
- `lm_log_rank_mean`: +0.0039
- `lm_logprob_mean`: +0.0037
- `tr_delta_adj_mean`: +0.0034
- `sent_len_std`: +0.0033

Training: {"n_train": 3920, "n_val": 610, "excluded_families": ["ai+dipper_L60_O60"], "families": ["ai", "ai+dipper_L20_O0", "ai+llm_paraphrase", "human", "human+quill_clarity", "human+quill_proofread"], "featurizer": "lm=gpt2|emb=sentence-transformers/all-MiniLM-L6-v2", "estimator": "hgb", "calibration": "isotonic"}
