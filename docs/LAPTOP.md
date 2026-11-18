# Running ScholarLens on a 6 GB laptop GPU (RTX 3050, Windows)

| Task | Where it runs | Why |
|---|---|---|
| Generators (AI abstracts), paraphrase attacks, Quill edits | Your GPU, through **Ollama** (4-bit) | A 7B model in 4-bit is about 4.7 GB |
| Lens training and evaluation | Your CPU or GPU | GPT-2 features and gradient boosting are light |
| Quill-RL (GRPO + DPO) | Your GPU, 0.5B model (or 1.5B with `--4bit`) | LoRA training fits in 6 GB |
| DIPPER attacks | Free **Kaggle** T4 (16 GB) | An 11B model; needs about 12 GB even in 8-bit |

## 1. Install (once)

The steps below assume WSL2 Ubuntu (recommended). Native Windows works too: use `.venv\Scripts\activate`.

```bash
git clone https://github.com/madhureddy25/ScholarLens.git && cd ScholarLens
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[all,dev]" bitsandbytes
python -c "import torch; print(torch.cuda.get_device_name(0))"     # NVIDIA GeForce RTX 3050 ...
pytest -q -m "not slow"
```

Install **Ollama** for Windows from https://ollama.com. From WSL, set `SCHOLARLENS_OLLAMA_URL=http://<windows-host-ip>:11434/v1` if `localhost` doesn't reach it. Then pull the models:

```bash
ollama pull qwen2.5:7b && ollama pull phi3.5 && ollama pull llama3.2:3b
```

Optional, for the grammar-based reward: Docker Desktop → `docker run -d -p 8010:8010 erikvl87/languagetool`.

## 2. Smoke test (about 15 minutes)

```bash
scholarlens bench build configs/bench.laptop.toml --n-human 30 --out data/bench/laptop-smoke
scholarlens lens train data/bench/laptop-smoke/bench.jsonl --out models/laptop-smoke
```

## 3. Build the benchmark (one or two nights; resumable, so it's fine to stop and restart)

```bash
scholarlens bench build configs/bench.laptop.toml
```

## 4. Train Lens

```bash
scholarlens lens train data/bench/laptop-v1/bench.jsonl --out models/lens-v1 --gnn
```

## 5. DIPPER on Kaggle (free GPU)

1. Zip `data/bench/laptop-v1` and upload it as a private Kaggle dataset. On Windows, zip with `tar.exe` (see `docs/WINDOWS.md`): `Compress-Archive` writes backslash paths that Kaggle rejects.
2. From the dataset page, create a new notebook. In Settings, set Accelerator to **GPU T4 x2** and turn **Internet on**. Then run these cells. Each `!` line runs in its own shell, so `%cd` is used to change directory:
   ```
   !git clone -q https://github.com/madhureddy25/ScholarLens.git /kaggle/working/ScholarLens
   %cd /kaggle/working/ScholarLens
   !pip install -q -e ".[ml]" bitsandbytes sentencepiece datasets
   !SRC=$(find /kaggle/input -maxdepth 5 -type d -name laptop-v1 | head -n 1); echo "found: $SRC"; mkdir -p data/bench && cp -r "$SRC" data/bench/ && ls data/bench/laptop-v1/transforms
   !PYTHONUNBUFFERED=1 scholarlens bench build configs/bench.kaggle-dipper.toml 2>&1 | grep --line-buffered -v MatMul8bitLt
   ```
   Expect about 9.5 hours for both settings (500 texts each) on a T4 x2; Kaggle's limit is 12 hours.
3. From the notebook's Output panel (`/kaggle/working/ScholarLens/data/bench/laptop-v1/transforms`), download `dipper_L20_O0.jsonl` and `dipper_L60_O60.jsonl` into `data/bench/laptop-v1/transforms/` on your laptop, under exactly those names.
4. Rerun `scholarlens bench build configs/bench.laptop.toml` to merge everything.

## 6. Train Quill-RL

```bash
scholarlens learn quill --bench data/bench/laptop-v1/bench.jsonl --base Qwen/Qwen2.5-0.5B-Instruct \
    --mode proofread --steps 200 --grammar-url http://localhost:8010 --out models/quill-rl
# larger model:  --base Qwen/Qwen2.5-1.5B-Instruct --4bit
# clarity mode with a judge:  --mode clarity --judge ollama:qwen2.5:7b
```

On an RTX 3050, expect about 2 hours for 200 steps (about 6 with `--judge`). Check first that `torch.cuda.is_available()` is True: on the CPU it takes about a day. The command compares the new adapter with the current one on the frozen test set and tells you whether it was accepted. Then try it:

```bash
scholarlens edit paper.tex -m quill:models/quill-rl --mode proofread
```

## 7. Share with everyone (see docs/LEARNING.md)

```bash
gh auth login
scholarlens learn publish bench data/bench/laptop-v1     # once: lets CI retrain Lens automatically
scholarlens learn publish lens models/lens-v1 --force    # first Lens release
scholarlens learn publish quill models/quill-rl          # if accepted
```

Publish the benchmark and models only if the licences of the models that generated the text allow it (see the header of `configs/bench.laptop.toml`).
