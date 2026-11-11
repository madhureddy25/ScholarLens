"""Build docs/LAPTOP_COMMANDS.pdf (the laptop-study commands, numbered). Usage: python scripts/make_commands_pdf.py docs/LAPTOP_COMMANDS.pdf

Needs reportlab and DejaVu fonts (/usr/share/fonts/truetype/dejavu). Keep in sync with LAPTOP_COMMANDS.ps1."""
import sys
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

F = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("Sans", F + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("Sans-Bold", F + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Mono", F + "DejaVuSansMono.ttf"))
pdfmetrics.registerFont(TTFont("Mono-Bold", F + "DejaVuSansMono-Bold.ttf"))

INK, MUTED, ACCENT = colors.HexColor("#1d1d1b"), colors.HexColor("#5f5e58"), colors.HexColor("#1f4e79")
CODE_BG, NOTE_BG, WARN_BG = colors.HexColor("#f2f1ec"), colors.HexColor("#eaf1f8"), colors.HexColor("#fdf3e1")

title = ParagraphStyle("t", fontName="Sans-Bold", fontSize=20, leading=24, textColor=INK, spaceAfter=2)
subtitle = ParagraphStyle("st", fontName="Sans", fontSize=10, leading=13, textColor=MUTED, spaceAfter=10)
h1 = ParagraphStyle("h1", fontName="Sans-Bold", fontSize=14, leading=18, textColor=ACCENT, spaceBefore=12, spaceAfter=4)
label = ParagraphStyle("l", fontName="Sans-Bold", fontSize=9.5, leading=12, textColor=INK, spaceBefore=6, spaceAfter=3)
body = ParagraphStyle("b", fontName="Sans", fontSize=9.5, leading=13, textColor=INK, alignment=TA_LEFT)
small = ParagraphStyle("s", parent=body, fontSize=8.5, leading=11, textColor=MUTED)
code = ParagraphStyle("c", fontName="Mono", fontSize=7.6, leading=10.2, textColor=INK)
num = ParagraphStyle("n", fontName="Mono-Bold", fontSize=7.6, leading=10.2, textColor=MUTED, alignment=2)

PAGE_W = landscape(A4)[0] - 2 * 14 * mm
counter = [0]


def commands(*cmds):
    rows = []
    for c in cmds:
        counter[0] += 1
        rows.append([Paragraph(str(counter[0]), num), Paragraph(escape(c), code)])
    t = Table(rows, colWidths=[9 * mm, PAGE_W - 9 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("LEFTPADDING", (0, 0), (0, -1), 2), ("RIGHTPADDING", (0, 0), (0, -1), 6),
        ("LEFTPADDING", (1, 0), (1, -1), 4),
    ]))
    return t


def block(text_lines, bg):
    t = Table([[Paragraph(escape(line), code)] for line in text_lines], colWidths=[PAGE_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5), ("LEFTPADDING", (0, 0), (-1, -1), 8)]))
    return t


def note(html, bg=NOTE_BG):
    t = Table([[Paragraph(html, body)]], colWidths=[PAGE_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("TOPPADDING", (0, 0), (-1, -1), 6),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 6), ("LEFTPADDING", (0, 0), (-1, -1), 8),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 8)]))
    return t


def step(label_text, *cmds):
    return KeepTogether([Paragraph(label_text, label), commands(*cmds)])


def numbered(items):
    return ListFlowable([ListItem(Paragraph(i, body), leftIndent=12) for i in items], bulletType="1",
                        bulletFontName="Sans-Bold", bulletFontSize=9.5, leftIndent=14)


s = []
s += [Paragraph("ScholarLens: laptop study commands", title),
      Paragraph("Windows PowerShell · NVIDIA RTX 3050 (6 GB) · github.com/madhureddy25/ScholarLens · "
                "same commands as LAPTOP_COMMANDS.ps1 in the repo", subtitle),
      note("<b>How to use this sheet.</b> Run the grey commands <b>one at a time</b>, in number order: copy one line, "
           "paste it into PowerShell, press Enter, and wait for it to finish before the next. Each numbered line is "
           "one complete command. Copy only the grey text, not the number. If anything errors, copy the full "
           "message, open the repo in Claude Code and say <i>continue ScholarLens</i>. Troubleshooting: "
           "docs/WINDOWS.md.")]

s.append(Paragraph("0 · One-time setup", h1))
s.append(step("Install the tools", "winget install --id Git.Git -e", "winget install --id Python.Python.3.12 -e",
              "winget install --id GitHub.cli -e", "winget install --id Ollama.Ollama -e"))
s.append(step("Close PowerShell and open a NEW window, then check the GPU (it should list “RTX 3050”)", "nvidia-smi"))
s.append(step("Get the code", "cd $HOME", "git clone https://github.com/madhureddy25/ScholarLens.git", "cd ScholarLens"))
s.append(step("Create the Python environment", "py -3.12 -m venv .venv",
              "Set-ExecutionPolicy -Scope CurrentUser RemoteSigned", r".\.venv\Scripts\Activate.ps1",
              "python -m pip install --upgrade pip",
              "pip install torch --index-url https://download.pytorch.org/whl/cu126",
              'pip install -e ".[all,dev]"',
              '[Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "User")'))
s.append(step("Check that it works: the first should print  True NVIDIA GeForce RTX 3050 …  and all tests should pass",
              'python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"',
              'pytest -q -m "not slow"'))
s.append(step("Download the AI models (about 9 GB in total)", "ollama pull qwen2.5:7b", "ollama pull phi3.5",
              "ollama pull llama3.2:3b"))

s.append(Spacer(1, 6))
s.append(KeepTogether([note("<b>Every time you open a new PowerShell window, run these two first:</b>", WARN_BG),
                       commands(r"cd $HOME\ScholarLens", r".\.venv\Scripts\Activate.ps1")]))

s.append(Paragraph("1 · Build the benchmark", h1))
s.append(step("Smoke test (about 30 minutes): check the counts it prints",
              "scholarlens bench build configs/bench.laptop.toml --n-human 40 --out data/bench/laptop-smoke",
              "scholarlens lens train data/bench/laptop-smoke/bench.jsonl --out models/laptop-smoke"))
s.append(step("Stop the laptop from sleeping while plugged in", "powercfg /change standby-timeout-ac 0"))
s.append(step("Full build, overnight. If it stops for any reason, run the SAME command again to resume.",
              "scholarlens bench build configs/bench.laptop.toml"))

s.append(Paragraph("2 · DIPPER on Kaggle (free GPU)", h1))
s.append(step("Zip the benchmark", r"Compress-Archive -Path data\bench\laptop-v1 -DestinationPath laptop-v1.zip -Force"))
kaggle = [Paragraph("On kaggle.com", label), numbered([
    "<b>Datasets → New Dataset</b>: upload <font name='Mono'>laptop-v1.zip</font> and keep it private.",
    "<b>Code → New Notebook</b>. In Settings, set Accelerator to <b>GPU T4</b> and turn Internet <b>on</b> "
    "(needs a phone-verified account).",
    "<b>Add Input</b>: pick your dataset.",
    "Paste and run these notebook cells (about 3–4 hours). <i>These are Kaggle cells, not PowerShell.</i>"]),
    Spacer(1, 3),
    block(["!git clone https://github.com/madhureddy25/ScholarLens.git",
           "%cd /kaggle/working/ScholarLens",
           '!pip install -q -e ".[ml]" bitsandbytes sentencepiece datasets',
           "!ls /kaggle/input/*/",
           "!mkdir -p data/bench && cp -r /kaggle/input/*/laptop-v1 data/bench/",
           "!scholarlens bench build configs/bench.kaggle-dipper.toml",
           "!mkdir -p /kaggle/working/dipper && cp data/bench/laptop-v1/transforms/dipper_*.jsonl /kaggle/working/dipper/"],
          colors.HexColor("#eef5ec"))]
s.append(KeepTogether(kaggle))
s.append(Spacer(1, 3))
s.append(ListFlowable([ListItem(Paragraph(t, body), leftIndent=12, value=v) for v, t in (
    (5, "From the notebook's <b>Output</b> panel, download <font name='Mono'>dipper_L20_O0.jsonl</font> and "
        "<font name='Mono'>dipper_L60_O60.jsonl</font>."),
    (6, "Copy both files into <font name='Mono'>data\\bench\\laptop-v1\\transforms\\</font> on the laptop."))],
    bulletType="1", bulletFontName="Sans-Bold", bulletFontSize=9.5, leftIndent=14))
s.append(step("Merge the DIPPER files into the benchmark (nothing gets regenerated)",
              "scholarlens bench build configs/bench.laptop.toml"))

s.append(Paragraph("3 · Train Lens and Quill-RL", h1))
s.append(step("Lens: the main run, then the two comparison runs for the paper",
              "scholarlens lens train data/bench/laptop-v1/bench.jsonl --out models/lens-v1 --gnn",
              "scholarlens lens train data/bench/laptop-v1/bench.jsonl --out models/lens-unseen-dipper --exclude ai+dipper_L60_O60",
              'scholarlens lens train data/bench/laptop-v1/bench.jsonl --out models/lens-no-edits --exclude "human+quill_proofread,human+quill_clarity"'))
s.append(step("Quill-RL: first free the GPU memory Ollama is holding", "ollama stop qwen2.5:7b"))
s.append(step("Then train (1–3 hours). It prints whether the new version was ACCEPTED: note the answer",
              "scholarlens learn quill --bench data/bench/laptop-v1/bench.jsonl --base Qwen/Qwen2.5-0.5B-Instruct --mode proofread --steps 200 --out models/quill-rl-v1"))
s.append(step("Try the trained editor on the sample paper",
              "scholarlens edit tests/fixtures/sample_paper.tex -m quill:models/quill-rl-v1 --mode proofread"))

s.append(Paragraph("4 · Publish the first releases", h1))
s.append(step("Log in to GitHub and set up git (put your own name and email in the first two)",
              'git config --global user.name "Your Name"', 'git config --global user.email "you@example.com"',
              "gh auth login", "gh auth setup-git"))
s.append(step("Publish", "scholarlens learn publish bench data/bench/laptop-v1",
              "scholarlens learn publish lens models/lens-v1 --force"))
s.append(step("Publish Quill ONLY if step 3 printed “accepted”", "scholarlens learn publish quill models/quill-rl-v1"))
s.append(step("Start the first automatic Lens retraining on GitHub",
              "gh workflow run learn --repo madhureddy25/ScholarLens"))

s.append(Paragraph("5 · Save the results for the paper", h1))
s.append(step("Copy the reports into the repo and push them",
              "New-Item -ItemType Directory -Force reports",
              'Get-ChildItem models -Directory | ForEach-Object { if (Test-Path "$($_.FullName)\\report.md") { Copy-Item "$($_.FullName)\\report.md" "reports\\$($_.Name).md" } }',
              r"Copy-Item data\bench\laptop-v1\manifest.json reports\bench-manifest.json",
              r"Copy-Item models\quill-rl-v1\learn_report.json reports\quill-rl-v1.json",
              "git add reports", 'git commit -m "Laptop study results"', "git push"))
s.append(Spacer(1, 4))
s.append(note("<b>Then:</b> open the repo in Claude Code and say <i>continue ScholarLens: draft the paper from "
              "reports/</i>."))
s.append(step("Turn sleep back on when everything is finished", "powercfg /change standby-timeout-ac 30"))

s.append(Paragraph("If something fails", h1))
s.append(Table([[Paragraph(f"<b>{a}</b>", body), Paragraph(b, body)] for a, b in (
    ("Activate.ps1 cannot be loaded", "Run command 10 (Set-ExecutionPolicy) and answer Y."),
    ("torch.cuda.is_available() is False", "Update the NVIDIA driver (GeForce Experience or nvidia.com), then run command 13 again."),
    ("Ollama connection refused", "Start the Ollama app from the Start menu, or run <font name='Mono'>ollama serve</font> in another window."),
    ("CUDA out of memory in learn quill", "Run <font name='Mono'>ollama stop qwen2.5:7b</font> first; also try adding <font name='Mono'>--group 2</font>."),
    ("A build stopped halfway", "Run the same command again: every step is cached and resumes."),
    ("Anything else", "Copy the full error message and ask Claude in Claude Code with the repo open."))],
    colWidths=[70 * mm, PAGE_W - 70 * mm],
    style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#dddad2")),
                      ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)])))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Sans", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(14 * mm, 8 * mm, "ScholarLens · laptop study commands · github.com/madhureddy25/ScholarLens")
    canvas.drawRightString(landscape(A4)[0] - 14 * mm, 8 * mm, f"page {doc.page}")
    canvas.restoreState()


out = sys.argv[1]
doc = SimpleDocTemplate(out, pagesize=landscape(A4), leftMargin=14 * mm, rightMargin=14 * mm, topMargin=12 * mm,
                        bottomMargin=14 * mm, title="ScholarLens: laptop study commands", author="ScholarLens")
doc.build(s, onFirstPage=footer, onLaterPages=footer)
print("commands:", counter[0])
