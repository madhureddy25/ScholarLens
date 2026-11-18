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
pip install -e ".[all,dev]"
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126   # GPU torch (plain pip gives the CPU build)
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

Zip with Windows' `tar`, not `Compress-Archive`: PowerShell 5.1 writes backslash paths, which Kaggle rejects ("contains a forbidden character in name").

```powershell
C:\Windows\System32\tar.exe -a -c -f laptop-v1.zip -C data\bench laptop-v1
C:\Windows\System32\tar.exe -tf laptop-v1.zip       # every line should use / (laptop-v1/ai.jsonl ...)
```

On kaggle.com:
1. **Datasets → New Dataset:** upload `laptop-v1.zip` and keep it private.
2. On the dataset page, **Code → New Notebook** (this attaches the dataset). In the **Settings** menu, set **Accelerator → GPU T4 x2** and click **Turn on internet**. Both need a phone-verified account (kaggle.com/settings).
3. Run these cells one at a time:
   ```
   !git clone -q https://github.com/madhureddy25/ScholarLens.git /kaggle/working/ScholarLens
   %cd /kaggle/working/ScholarLens
   !pip install -q -e ".[ml]" bitsandbytes sentencepiece datasets
   !SRC=$(find /kaggle/input -maxdepth 5 -type d -name laptop-v1 | head -n 1); echo "found: $SRC"; mkdir -p data/bench && cp -r "$SRC" data/bench/ && ls data/bench/laptop-v1/transforms
   !PYTHONUNBUFFERED=1 scholarlens bench build configs/bench.kaggle-dipper.toml 2>&1 | grep --line-buffered -v MatMul8bitLt
   ```
   - Kaggle now mounts datasets at `/kaggle/input/datasets/<user>/<slug>/`, so the cell uses `find` rather than a fixed path.
   - The `grep` hides a bitsandbytes notice (`MatMul8bitLt: inputs will be cast…`) that is printed on every forward pass and buries the progress lines.
   - The build should start with `human: 1500`, `ai: 1500`, `transform dipper_L20_O0 on ai: 0 cached, 500 to do`. If it prints `generator echo: … to generate`, stop it: the AI texts were not copied.
   - **Timing on a T4 x2:** model download about 6 min (45 GB), load about 3 min, then about 35 s per text: roughly **9.5 hours** for both settings. Kaggle stops sessions at 12 hours.
   - Keep the tab open and the laptop awake. **Never press the circular-arrow button** next to the power button: it is a factory reset that deletes everything in the session.
4. In the **Output** panel, open `/kaggle/working/ScholarLens/data/bench/laptop-v1/transforms` and download `dipper_L20_O0.jsonl` as soon as `dipper_L60_O60` starts (a backup), then `dipper_L60_O60.jsonl` at the end. Then stop the session with the power button.

Copy the newest download of each file under its exact name (the builder merges every `.jsonl` in `transforms/`, so a stray copy would be counted twice), check, and merge. Nothing is regenerated:

```powershell
Get-ChildItem "$HOME\Downloads" -Filter "dipper_L20_O0*.jsonl" | Sort-Object LastWriteTime | Select-Object -Last 1 | Copy-Item -Destination "data\bench\laptop-v1\transforms\dipper_L20_O0.jsonl"
Get-ChildItem "$HOME\Downloads" -Filter "dipper_L60_O60*.jsonl" | Sort-Object LastWriteTime | Select-Object -Last 1 | Copy-Item -Destination "data\bench\laptop-v1\transforms\dipper_L60_O60.jsonl"
Get-ChildItem data\bench\laptop-v1\transforms | ForEach-Object { "$($_.Name): $((Get-Content $_.FullName).Count)" }   # 5 files, 500 each
scholarlens bench build configs/bench.laptop.toml                                                                       # 5500 records
```

## 3. Train Lens and Quill-RL

```powershell
scholarlens lens train data/bench/laptop-v1/bench.jsonl --out models/lens-v1 --gnn
scholarlens lens train data/bench/laptop-v1/bench.jsonl --out models/lens-unseen-dipper --exclude ai+dipper_L60_O60
scholarlens lens train data/bench/laptop-v1/bench.jsonl --out models/lens-no-edits --exclude "human+quill_proofread,human+quill_clarity"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"                  # must print ...+cu126 True
ollama stop qwen2.5:7b                                                                            # free GPU memory
scholarlens learn quill --bench data/bench/laptop-v1/bench.jsonl --base Qwen/Qwen2.5-0.5B-Instruct --mode proofread --steps 200 --out models/quill-rl-v1
scholarlens edit tests/fixtures/sample_paper.tex -m quill:models/quill-rl-v1 --mode proofread     # try it
```

`learn quill` takes about 2 hours on the RTX 3050 (about 30 s per step). If `step_time` in its log is around 500, it is running on the CPU: stop it and fix torch (see below).

Without a quality reward, training teaches integrity and restraint only, and the editor learns to change very little. To add one (optional, about 6 hours, gated against v1):

```powershell
ollama pull qwen2.5:3b
scholarlens learn quill --bench data/bench/laptop-v1/bench.jsonl --base Qwen/Qwen2.5-0.5B-Instruct --mode proofread --steps 200 --judge ollama:qwen2.5:3b --current models/quill-rl-v1 --out models/quill-rl-v2
```

The training and the 3B judge together fill the 6 GB card, so the laptop feels slow while it runs. LanguageTool is the other quality option: Docker Desktop, `docker run -d -p 8010:8010 erikvl87/languagetool`, then `--grammar-url http://localhost:8010`.

## 4. Publish the first releases

```powershell
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
gh auth login
gh auth setup-git
scholarlens learn publish bench data/bench/laptop-v1        # then add "Built with Llama" to the release notes (Llama 3.2 output)
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
if (Test-Path models\quill-rl-v2\learn_report.json) { Copy-Item models\quill-rl-v2\learn_report.json reports\quill-rl-v2.json }
git add reports
git commit -m "Laptop study results"
git push
powercfg /change standby-timeout-ac 30                      # turn sleep back on
```

## If something fails

- **`Activate.ps1 cannot be loaded`:** run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` and answer Y.
- **`torch.cuda.is_available()` is False, or `learn quill` logs `pin_memory ... no accelerator is found`:** check `python -c "import torch; print(torch.__version__)"`. If it ends in `+cpu`, a later `pip install` replaced the GPU build (on Windows, PyPI only has the CPU build): run `pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126`. If it already ends in `+cu126`, update the NVIDIA driver.
- **Ollama connection refused:** start the Ollama app from the Start menu, or run `ollama serve` in another window.
- **CUDA out of memory in `learn quill`:** run `ollama stop qwen2.5:7b` first. Also try `--group 2`, or keep the 0.5B base model.
- **A build stopped halfway:** run the same command again. Every step is cached.
- **Kaggle: "contains a forbidden character in name ('\\')":** the zip was made with `Compress-Archive`; make it again with `tar.exe` (step 2).
- **Kaggle: GPU T4 x2 is greyed out:** verify your phone number at kaggle.com/settings, then reload the notebook.
- **Lens models trained before the uncalibrated-ranking fix (Nov 2026):** rewrite their reports without retraining: `scholarlens lens rescore data/bench/laptop-v1/bench.jsonl models/lens-v1 models/lens-unseen-dipper models/lens-no-edits`.
