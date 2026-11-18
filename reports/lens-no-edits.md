# Lens evaluation — data\bench\laptop-v1\bench.jsonl

Test split. Threshold per detector set so clean human text has FPR ≤ 1%. lens-fusion is ranked by its uncalibrated score (same order as P(AI), without isotonic ties).

## AI detection under attack (AUROC / TPR@1% FPR)

| detector | ai | ai+dipper_L20_O0 | ai+dipper_L60_O60 | ai+llm_paraphrase |
|---|---|---|---|---|
| lens-fusion | 0.993 / 0.92 | 0.963 / 0.82 | 0.966 / 0.57 | 0.974 / 0.82 |
| fast_detectgpt | 0.891 / 0.36 | 0.910 / 0.27 | 0.918 / 0.27 | 0.686 / 0.05 |
| neg_log_ppl | 0.920 / 0.13 | 0.868 / 0.11 | 0.822 / 0.05 | 0.799 / 0.00 |
| neg_log_rank | 0.916 / 0.24 | 0.860 / 0.25 | 0.818 / 0.14 | 0.766 / 0.02 |
| neg_entropy | 0.798 / 0.01 | 0.672 / 0.00 | 0.524 / 0.00 | 0.726 / 0.00 |
| retrieval | 1.000 / 1.00 | 0.999 / 0.95 | 0.891 / 0.61 | 1.000 / 1.00 |

## False-positive rate on human text (lower is better)

| detector | human | human+quill_clarity | human+quill_proofread |
|---|---|---|---|
| lens-fusion | 0.007 (n=145) | 0.022 (n=46) | 0.022 (n=46) |
| fast_detectgpt | 0.007 (n=145) | 0.022 (n=46) | 0.022 (n=46) |
| neg_log_ppl | 0.007 (n=145) | 0.043 (n=46) | 0.022 (n=46) |
| neg_log_rank | 0.007 (n=145) | 0.065 (n=46) | 0.065 (n=46) |
| neg_entropy | 0.007 (n=145) | 0.022 (n=46) | 0.022 (n=46) |
| retrieval | 0.007 (n=145) | 0.022 (n=46) | 0.022 (n=46) |

## Top fusion features (permutation importance, ΔAUROC)

- `llm_marker_per_1k`: +0.0450
- `lm_fast_detectgpt`: +0.0269
- `avg_word_len`: +0.0248
- `lm_logprob_std`: +0.0033
- `sent_len_min`: +0.0024
- `lm_log_ppl`: +0.0020
- `function_word_ratio`: +0.0016
- `tr_adj_sim_max`: +0.0013
- `tr_delta_adj_mean`: +0.0013
- `lm_logprob_mean`: +0.0012
- `burstiness`: +0.0011
- `sent_len_max`: +0.0010

Training: {"feature_names": ["avg_word_len", "burstiness", "colon_per_1k", "comma_per_sentence", "connective_start_rate", "dash_per_1k", "function_word_ratio", "llm_marker_per_1k", "lm_entropy_mean", "lm_entropy_std", "lm_entropy_window_var", "lm_fast_detectgpt", "lm_log_ppl", "lm_log_rank_mean", "lm_logprob_mean", "lm_logprob_std", "lm_ppl", "lm_top10_frac", "lm_top1_frac", "mattr", "paren_per_1k", "semicolon_per_1k", "sent_len_cv", "sent_len_max", "sent_len_mean", "sent_len_min", "sent_len_std", "tr_adj_sim_max", "tr_adj_sim_mean", "tr_adj_sim_min", "tr_adj_sim_std", "tr_adj_spike_rate", "tr_delta_abs_mean", "tr_delta_adj_mean", "tr_delta_adj_std", "tr_local_global_ratio", "tr_offdiag_sim_mean", "tr_offdiag_sim_var"], "n_train": 3525, "n_val": 553, "excluded_families": ["human+quill_proofread", "human+quill_clarity"], "families": ["ai", "ai+dipper_L20_O0", "ai+dipper_L60_O60", "ai+llm_paraphrase", "human"], "featurizer": "lm=gpt2|emb=sentence-transformers/all-MiniLM-L6-v2", "estimator": "hgb", "calibration": "isotonic"}
