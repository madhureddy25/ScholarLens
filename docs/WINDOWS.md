# Windows PowerShell runbook (RTX 3050 6 GB laptop)

Run the commands one at a time in **PowerShell**. Background on why each step runs where it does is in `docs/LAPTOP.md`.

## 0. One-time setup

```powershell
winget install --id Git.Git -e
winget install --id Python.Python.3.12 -e
winget install --id GitHub.cli -e
winget install --id Ollama.Ollama -e
```

Close PowerShell and open a new window. Then:

```powershell
nvidia-smi                                   # should list the RTX 3050
cd $HOME
git clone https://github.com/madhureddy25/ScholarLens.git
cd ScholarLens
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[all,dev]"
[Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "User")
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # True NVIDIA GeForce RTX 3050 ...
pytest -q -m "not slow"
ollama pull qwen2.5:7b
ollama pull phi3.5
ollama pull llama3.2:3b
```

**Every new PowerShell window:**

```powershell
cd $HOME\ScholarLens
.\.venv\Scripts\Activate.ps1
```

## 1. Build the benchmark

```powershell
scholarlens bench build configs/bench.laptop.toml --n-human 40 --out data/bench/laptop-smoke    # ~30 min smoke test
scholarlens lens train data/bench/laptop-smoke/bench.jsonl --out models/laptop-smoke
powercfg /change standby-timeout-ac 0                                                           # no sleep while plugged in
scholarlens bench build configs/bench.laptop.toml                                               # overnight; rerun to resume
```

## 2. DIPPER on Kaggle (free GPU)

```powershell
Compress-Archive -Path data\bench\laptop-v1 -DestinationPath laptop-v1.zip -Force
```

On kaggle.com:
1. **Datasets → New Dataset:** upload `laptop-v1.zip` and keep it private.
2. **Code → New Notebook:** in Settings, set Accelerator to **GPU T4** and turn Internet **on** (needs a phone-verified account). Use **Add Input** to attach your dataset.
3. Run these cells:
   ```
   !git clone https://github.com/madhureddy25/ScholarLens.git
   %cd /kaggle/working/ScholarLens
   !pip install -q -e ".[ml]" bitsandbytes sentencepiece datasets
   !ls /kaggle/input/*/
   !mkdir -p data/bench && cp -r /kaggle/input/*/laptop-v1 data/bench/
   !scholarlens bench build configs/bench.kaggle-dipper.toml
   !mkdir -p /kaggle/working/dipper && cp data/bench/laptop-v1/transforms/dipper_*.jsonl /kaggle/working/dipper/
   ```
4. From the **Output** panel, download `dipper_L20_O0.jsonl` and `dipper_L60_O60.jsonl` into `data\bench\laptop-v1\transforms\` on the laptop.

Then merge. Nothing is regenerated:

```powershell
scholarlens bench build configs/bench.laptop.toml
```

## 3. Train Lens and Quill-RL

```powershell
scholarlens lens train data/bench/laptop-v1/bench.jsonl --out models/lens-v1 --gnn
scholarlens lens train data/bench/laptop-v1/bench.jsonl --out models/lens-unseen-dipper --exclude ai+dipper_L60_O60
scholarlens lens train data/bench/laptop-v1/bench.jsonl --out models/lens-no-edits --exclude "human+quill_proofread,human+quill_clarity"
ollama stop qwen2.5:7b                                                                            # free GPU memory
scholarlens learn quill --bench data/bench/laptop-v1/bench.jsonl --base Qwen/Qwen2.5-0.5B-Instruct --mode proofread --steps 200 --out models/quill-rl-v1
scholarlens edit tests/fixtures/sample_paper.tex -m quill:models/quill-rl-v1 --mode proofread     # try it
```

The quality reward is optional. It needs Docker Desktop: `docker run -d -p 8010:8010 erikvl87/languagetool`, then add `--grammar-url http://localhost:8010` to `learn quill`.

## 4. Publish the first releases

```powershell
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
gh auth login
gh auth setup-git
scholarlens learn publish bench data/bench/laptop-v1
scholarlens learn publish lens models/lens-v1 --force
scholarlens learn publish quill models/quill-rl-v1          # only if step 3 printed "accepted"
gh workflow run learn --repo madhureddy25/ScholarLens        # first automatic Lens retraining
```

## 5. Save the results for the paper

```powershell
New-Item -ItemType Directory -Force reports
Get-ChildItem models -Directory | ForEach-Object { if (Test-Path "$($_.FullName)\report.md") { Copy-Item "$($_.FullName)\report.md" "reports\$($_.Name).md" } }
Copy-Item data\bench\laptop-v1\manifest.json reports\bench-manifest.json
Copy-Item models\quill-rl-v1\learn_report.json reports\quill-rl-v1.json
git add reports
git commit -m "Laptop study results"
git push
powercfg /change standby-timeout-ac 30                      # turn sleep back on
```

## If something fails

- **`Activate.ps1 cannot be loaded`:** run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` and answer Y.
- **`torch.cuda.is_available()` is False:** update the NVIDIA driver (GeForce Experience or nvidia.com), then reinstall torch with the `cu126` command.
- **Ollama connection refused:** start the Ollama app from the Start menu, or run `ollama serve` in another window.
- **CUDA out of memory in `learn quill`:** run `ollama stop qwen2.5:7b` first. Also try `--group 2`, or keep the 0.5B base model.
- **A build stopped halfway:** run the same command again. Every step is cached.
