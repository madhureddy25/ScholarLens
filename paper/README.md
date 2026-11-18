# Paper draft

`main.tex` + `refs.bib` → `main.pdf`. Every number comes from `../reports/` (rescored Lens reports, Quill-RL JSON, bench manifest). The 95% intervals are Clopper–Pearson intervals from the counts behind the reported rates (for example, 0.50 of 44 = 22/44).

## Build

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

This needs TeX Live with `natbib`, `booktabs`, `caption`, `enumitem`, `microtype` and `lmodern`. On Windows, MiKTeX installs missing packages automatically. Alternatively, upload `main.tex` and `refs.bib` to Overleaf.

## Before submitting

- [ ] **Author block:** affiliation, and check the name spelling.
- [ ] **"Use of AI assistants" section:** say how AI was used, following the venue's policy. This draft was written with Claude from `reports/`.
- [ ] **Verify every reference** in `refs.bib` against the published version: authors, venue, year.
- [ ] **Choose a venue** and switch to its template; the content is template-independent.
- [ ] **Optional, to strengthen claims:**
  - AUROC confidence intervals (`auroc_ci95` in `models/*/report.json` on the laptop)
  - multiple seeds
  - a larger test split
  - Binoculars
  - LanguageTool edits
  - a non-native-author subset
  - a leave-one-generator-out run
- [ ] **Figures:** a system diagram (from `docs/ARCHITECTURE.md`) and a feature-importance bar chart (from `reports/lens-v1.md`) are planned in `docs/PAPER_PLAN.md`; they are not in the draft yet.
