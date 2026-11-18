# =====================================================================================
#  ScholarLens: every command for the laptop study, in order (Windows PowerShell)
#  Laptop: RTX 3050 6 GB · Windows · run each command ONE AT A TIME (copy and paste)
#  More detail and troubleshooting: docs/WINDOWS.md
# =====================================================================================
Write-Host "Don't run this file as a script: copy the commands one at a time." ; return


# -------------------------------------------------------------------------------------
# 0. ONE-TIME SETUP
# -------------------------------------------------------------------------------------

# Install the tools
winget install --id Git.Git -e
winget install --id Python.Python.3.12 -e
winget install --id GitHub.cli -e
winget install --id Ollama.Ollama -e

# >>> Close PowerShell and open a NEW window, then check the GPU (should list "RTX 3050"):
nvidia-smi

# Get the code
cd $HOME
git clone https://github.com/madhureddy25/ScholarLens.git
cd ScholarLens

# Create the Python environment
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[all,dev]"
# Then put the GPU build of torch back: on Windows, plain pip installs the CPU-only build
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
[Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "User")

# Check it works: should print  True NVIDIA GeForce RTX 3050 ...  and all tests should pass
# (if it prints False, run the --force-reinstall torch command above again)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pytest -q -m "not slow"

# Download the AI models (about 9 GB in total)
ollama pull qwen2.5:7b
ollama pull phi3.5
ollama pull llama3.2:3b


# -------------------------------------------------------------------------------------
# EVERY TIME YOU OPEN A NEW POWERSHELL WINDOW: run these two first
# -------------------------------------------------------------------------------------
cd $HOME\ScholarLens
.\.venv\Scripts\Activate.ps1


# -------------------------------------------------------------------------------------
# STEP 1: BUILD THE BENCHMARK
# -------------------------------------------------------------------------------------

# Smoke test (about 30 min): check the counts it prints
scholarlens bench build configs/bench.laptop.toml --n-human 40 --out data/bench/laptop-smoke
scholarlens lens train data/bench/laptop-smoke/bench.jsonl --out models/laptop-smoke

# Stop the laptop from sleeping while plugged in
powercfg /change standby-timeout-ac 0

# Full build, overnight. If it stops for any reason, run the SAME command again to resume.
scholarlens bench build configs/bench.laptop.toml


# -------------------------------------------------------------------------------------
# STEP 2: DIPPER ON KAGGLE (free GPU)
# -------------------------------------------------------------------------------------

# Zip the benchmark with Windows' tar (Compress-Archive in PowerShell 5.1 writes backslash paths,
# which Kaggle rejects: "contains a forbidden character in name")
C:\Windows\System32\tar.exe -a -c -f laptop-v1.zip -C data\bench laptop-v1
# Check: every line should use forward slashes (laptop-v1/ai.jsonl ...)
C:\Windows\System32\tar.exe -tf laptop-v1.zip

# On kaggle.com:
#   1. Datasets -> New Dataset -> upload laptop-v1.zip (keep it private)
#   2. On the dataset page: Code -> New Notebook (the dataset is attached as input)
#      Settings -> Accelerator = GPU T4 x2, Settings -> Turn on internet (both need a phone-verified account)
#   3. Paste and run these notebook cells. One cell at a time; the last one takes about 9-10 hours
#      (45 GB model download ~6 min, then ~35 s per text). Kaggle stops sessions at 12 hours.
#      Keep the tab open and the laptop awake. Never press the circular-arrow button: it is a factory
#      reset that deletes everything in the session.
#        !git clone -q https://github.com/madhureddy25/ScholarLens.git /kaggle/working/ScholarLens
#        %cd /kaggle/working/ScholarLens
#        !pip install -q -e ".[ml]" bitsandbytes sentencepiece datasets
#        !SRC=$(find /kaggle/input -maxdepth 5 -type d -name laptop-v1 | head -n 1); echo "found: $SRC"; mkdir -p data/bench && cp -r "$SRC" data/bench/ && ls data/bench/laptop-v1/transforms
#        !PYTHONUNBUFFERED=1 scholarlens bench build configs/bench.kaggle-dipper.toml 2>&1 | grep --line-buffered -v MatMul8bitLt
#      It should start with  human: 1500 / ai: 1500 / transform dipper_L20_O0 on ai: 0 cached, 500 to do.
#      (If it says "generator echo: ... to generate", stop it: the AI texts were not copied.)
#   4. Output panel -> /kaggle/working/ScholarLens/data/bench/laptop-v1/transforms: download
#      dipper_L20_O0.jsonl (as soon as dipper_L60_O60 starts, as a backup) and dipper_L60_O60.jsonl
#   5. Stop the session with the power button to save GPU hours.

# Copy the newest download of each file into the benchmark, under the exact names
Get-ChildItem "$HOME\Downloads" -Filter "dipper_L20_O0*.jsonl" | Sort-Object LastWriteTime | Select-Object -Last 1 | Copy-Item -Destination "data\bench\laptop-v1\transforms\dipper_L20_O0.jsonl"
Get-ChildItem "$HOME\Downloads" -Filter "dipper_L60_O60*.jsonl" | Sort-Object LastWriteTime | Select-Object -Last 1 | Copy-Item -Destination "data\bench\laptop-v1\transforms\dipper_L60_O60.jsonl"
# Check: exactly 5 files, 500 lines each
Get-ChildItem data\bench\laptop-v1\transforms | ForEach-Object { "$($_.Name): $((Get-Content $_.FullName).Count)" }

# Merge the DIPPER files into the benchmark (nothing gets regenerated)
scholarlens bench build configs/bench.laptop.toml


# -------------------------------------------------------------------------------------
# STEP 3: TRAIN LENS AND QUILL-RL
# -------------------------------------------------------------------------------------

# Lens: the main run, then the two comparison runs for the paper
scholarlens lens train data/bench/laptop-v1/bench.jsonl --out models/lens-v1 --gnn
scholarlens lens train data/bench/laptop-v1/bench.jsonl --out models/lens-unseen-dipper --exclude ai+dipper_L60_O60
scholarlens lens train data/bench/laptop-v1/bench.jsonl --out models/lens-no-edits --exclude "human+quill_proofread,human+quill_clarity"

# Quill-RL: first check torch sees the GPU (must print  ...+cu126 True ; on the CPU this takes a day)
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Free the GPU memory Ollama is holding
ollama stop qwen2.5:7b

# Then train (about 2 hours on the RTX 3050); it prints whether the new version was ACCEPTED
scholarlens learn quill --bench data/bench/laptop-v1/bench.jsonl --base Qwen/Qwen2.5-0.5B-Instruct --mode proofread --steps 200 --out models/quill-rl-v1

# Optional, about 6 hours: v2 with an LLM judge as the quality reward, gated against v1
ollama pull qwen2.5:3b
scholarlens learn quill --bench data/bench/laptop-v1/bench.jsonl --base Qwen/Qwen2.5-0.5B-Instruct --mode proofread --steps 200 --judge ollama:qwen2.5:3b --current models/quill-rl-v1 --out models/quill-rl-v2

# Try the trained editor on the sample paper
scholarlens edit tests/fixtures/sample_paper.tex -m quill:models/quill-rl-v1 --mode proofread


# -------------------------------------------------------------------------------------
# STEP 4: PUBLISH THE FIRST RELEASES
# -------------------------------------------------------------------------------------

# Log in to GitHub and set up git (put your own name and email in the first two)
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
gh auth login
gh auth setup-git

# Publish. The benchmark contains Llama 3.2 output: add "Built with Llama" to the bench-v1 release notes.
scholarlens learn publish bench data/bench/laptop-v1
scholarlens learn publish lens models/lens-v1 --force

# Publish Quill ONLY if step 3 printed "accepted"
scholarlens learn publish quill models/quill-rl-v1

# Start the first automatic Lens retraining on GitHub
gh workflow run learn --repo madhureddy25/ScholarLens


# -------------------------------------------------------------------------------------
# STEP 5: SAVE THE RESULTS FOR THE PAPER
# -------------------------------------------------------------------------------------
New-Item -ItemType Directory -Force reports
Get-ChildItem models -Directory | ForEach-Object { if (Test-Path "$($_.FullName)\report.md") { Copy-Item "$($_.FullName)\report.md" "reports\$($_.Name).md" } }
Copy-Item data\bench\laptop-v1\manifest.json reports\bench-manifest.json
Copy-Item models\quill-rl-v1\learn_report.json reports\quill-rl-v1.json
if (Test-Path models\quill-rl-v2\learn_report.json) { Copy-Item models\quill-rl-v2\learn_report.json reports\quill-rl-v2.json }
git add reports
git commit -m "Laptop study results"
git push

# >>> Then open the repo in Claude Code and say "continue ScholarLens: draft the paper from reports/".

# Turn sleep back on when everything is finished
powercfg /change standby-timeout-ac 30
