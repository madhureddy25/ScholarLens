# Running the study on your GPU

Everything below runs on your own machine. A 24 GB NVIDIA card runs the whole config. Smaller cards need the substitutions noted in `configs/bench.gpu.toml`.

## 1. Set up

```bash
git clone https://github.com/madhureddy25/ScholarLens.git && cd ScholarLens
python -m venv .venv && source .venv/bin/activate        # Windows: use WSL2, or .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match your CUDA (see pytorch.org)
pip install -e ".[all,dev]" datasets accelerate
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pytest -q -m "not slow"                                   # should be all green
```

Optional: `pip install bitsandbytes` (8-bit DIPPER on smaller cards) and `huggingface-cli login` (gated models such as Llama).

## 2. Smoke test (about 10 minutes)

```bash
scholarlens bench build configs/bench.gpu.toml --n-human 40 --out data/bench/gpu-smoke
scholarlens bench stats data/bench/gpu-smoke/bench.jsonl
scholarlens lens train data/bench/gpu-smoke/bench.jsonl --out models/gpu-smoke
```

Check `data/bench/gpu-smoke/ai.jsonl` by eye. The generated abstracts should read like real abstracts, with no "Abstract:" headers or chat wrappers.

## 3. Full benchmark (overnight)

```bash
scholarlens bench build configs/bench.gpu.toml
```

Builds are resumable. If it stops (power, OOM, Ctrl-C), run the same command again and it continues where it left off. Generators load one at a time, and GPU memory is freed between them.

## 4. Experiments

```bash
# A. Main result: robust training on all families
scholarlens lens train data/bench/gpu-v1/bench.jsonl --out models/A-robust --gnn --binoculars

# B. Unseen-attack generalisation: hold one attack family out of training
scholarlens lens train data/bench/gpu-v1/bench.jsonl --out models/B-no-dipper60 --exclude ai+dipper_L60_O60

# C. Fairness without edited-human training data: does editing raise false positives?
scholarlens lens train data/bench/gpu-v1/bench.jsonl --out models/C-no-edits \
    --exclude human+quill_proofread,human+quill_clarity

# D. Stronger LM features
scholarlens lens train data/bench/gpu-v1/bench.jsonl --out models/D-qwen-features --lm Qwen/Qwen2.5-1.5B
```

Each run writes `report.md` (tables) and `report.json` (all numbers, with bootstrap CIs).

## 5. Share results

Commit the reports, but not the data or the models. `data/` and `models/` are git-ignored because they're large, and the arXiv abstracts keep their original licences.

```bash
mkdir -p reports && for d in models/*/; do cp "$d/report.md" "reports/$(basename $d).md"; done
git add reports && git commit -m "GPU study results" && git push
```

## Expected time on one 24 GB GPU

| Step | ≈ time |
|---|---|
| 2000 generations × 3 models (sequential, bf16) | 4–6 h |
| DIPPER, 2 settings × 600 | 2–3 h |
| LLM paraphrase + Quill edits (1800) | 2–3 h |
| Lens training with GNN | < 30 min |

To speed up generation 5–10×, serve a model with vLLM (`vllm serve Qwen/Qwen2.5-7B-Instruct`) and use `model = "vllm:Qwen/Qwen2.5-7B-Instruct"` with `concurrency = 16`.
