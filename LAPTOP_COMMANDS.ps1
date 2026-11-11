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
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[all,dev]"
[Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "User")

# Check it works: should print  True NVIDIA GeForce RTX 3050 ...  and all tests should pass
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

# Zip the benchmark
Compress-Archive -Path data\bench\laptop-v1 -DestinationPath laptop-v1.zip -Force

# On kaggle.com:
#   1. Datasets -> New Dataset -> upload laptop-v1.zip (keep it private)
#   2. Code -> New Notebook -> Settings: Accelerator = GPU T4, Internet = ON (needs a phone-verified account)
#   3. Add Input -> pick your dataset
#   4. Paste and run these notebook cells (about 3-4 hours):
#        !git clone https://github.com/madhureddy25/ScholarLens.git
#        %cd /kaggle/working/ScholarLens
#        !pip install -q -e ".[ml]" bitsandbytes sentencepiece datasets
#        !ls /kaggle/input/*/
#        !mkdir -p data/bench && cp -r /kaggle/input/*/laptop-v1 data/bench/
#        !scholarlens bench build configs/bench.kaggle-dipper.toml
#        !mkdir -p /kaggle/working/dipper && cp data/bench/laptop-v1/transforms/dipper_*.jsonl /kaggle/working/dipper/
#   5. From the Output panel, download  dipper_L20_O0.jsonl  and  dipper_L60_O60.jsonl
#   6. Copy both files into  data\bench\laptop-v1\transforms\

# Merge the DIPPER files into the benchmark (nothing gets regenerated)
scholarlens bench build configs/bench.laptop.toml


# -------------------------------------------------------------------------------------
# STEP 3: TRAIN LENS AND QUILL-RL
# -------------------------------------------------------------------------------------

# Lens: the main run, then the two comparison runs for the paper
scholarlens lens train data/bench/laptop-v1/bench.jsonl --out models/lens-v1 --gnn
scholarlens lens train data/bench/laptop-v1/bench.jsonl --out models/lens-unseen-dipper --exclude ai+dipper_L60_O60
scholarlens lens train data/bench/laptop-v1/bench.jsonl --out models/lens-no-edits --exclude "human+quill_proofread,human+quill_clarity"

# Quill-RL: first free the GPU memory Ollama is holding
ollama stop qwen2.5:7b

# Then train (1-3 hours); it prints whether the new version was ACCEPTED
scholarlens learn quill --bench data/bench/laptop-v1/bench.jsonl --base Qwen/Qwen2.5-0.5B-Instruct --mode proofread --steps 200 --out models/quill-rl-v1

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

# Publish
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
git add reports
git commit -m "Laptop study results"
git push

# >>> Then open the repo in Claude Code and say "continue ScholarLens: draft the paper from reports/".

# Turn sleep back on when everything is finished
powercfg /change standby-timeout-ac 30
