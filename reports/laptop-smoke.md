# Lens evaluation — data\bench\laptop-smoke\bench.jsonl

Test split. Threshold per detector set so clean human text has FPR ≤ 1%.

## AI detection under attack (AUROC / TPR@1% FPR)

| detector | ai | ai+llm_paraphrase |
|---|---|---|
| lens-fusion | 1.000 / 1.00 | 0.500 / 0.00 |
| fast_detectgpt | 1.000 / 1.00 | 1.000 / 1.00 |
| neg_log_ppl | 1.000 / 1.00 | 1.000 / 1.00 |
| neg_log_rank | 1.000 / 1.00 | 1.000 / 1.00 |
| neg_entropy | 1.000 / 1.00 | 1.000 / 1.00 |
| retrieval | 1.000 / 1.00 | 1.000 / 1.00 |

## False-positive rate on human text (lower is better)

| detector | human | human+quill_clarity | human+quill_proofread |
|---|---|---|---|
| lens-fusion | 0.000 (n=1) | 0.000 (n=1) | 0.000 (n=1) |
| fast_detectgpt | 0.000 (n=1) | 1.000 (n=1) | 1.000 (n=1) |
| neg_log_ppl | 0.000 (n=1) | 1.000 (n=1) | 1.000 (n=1) |
| neg_log_rank | 0.000 (n=1) | 1.000 (n=1) | 1.000 (n=1) |
| neg_entropy | 0.000 (n=1) | 0.000 (n=1) | 1.000 (n=1) |
| retrieval | 0.000 (n=1) | 1.000 (n=1) | 0.000 (n=1) |

## Top fusion features (permutation importance, ΔAUROC)

- `llm_marker_per_1k`: +0.1005
- `lm_fast_detectgpt`: +0.0731
- `avg_word_len`: +0.0681
- `tr_adj_sim_max`: +0.0222
- `mattr`: +0.0167
- `paren_per_1k`: +0.0167
- `sent_len_min`: +0.0167
- `function_word_ratio`: +0.0009
- `burstiness`: +0.0000
- `colon_per_1k`: +0.0000
- `comma_per_sentence`: +0.0000
- `connective_start_rate`: +0.0000

Training: {"n_train": 165, "n_val": 30, "excluded_families": [], "families": ["ai", "ai+llm_paraphrase", "human", "human+quill_clarity", "human+quill_proofread"], "featurizer": "lm=gpt2|emb=sentence-transformers/all-MiniLM-L6-v2", "estimator": "hgb", "calibration": "isotonic"}
