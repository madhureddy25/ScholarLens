# How ScholarLens keeps improving

Every user can make the shared models better, and everyone gets the improvements through `scholarlens update`. Guard rails keep this safe: no private text leaves anyone's machine without consent, and no contribution can quietly make a model worse.

```
  author's machine                          GitHub (public)                       everyone
 ┌──────────────────────┐  opt-in PR   ┌───────────────────────────┐  release  ┌─────────────────┐
 │ scholarlens edit     │ ───────────▶ │ contributions/*.jsonl     │ ────────▶ │ scholarlens     │
 │ scholarlens review   │  (masked     │  CI: validate --strict    │  lens-vN  │   update        │
 │  accept / reject     │   text only) │  learn.yml: retrain Lens  │  quill-vN │ -m quill:latest │
 │ scholarlens          │              │  gate vs frozen test set  │           │ detect (latest) │
 │   contribute         │              │  publish only if better   │           └─────────────────┘
 └──────────────────────┘              └───────────────────────────┘
                                                    ▲
                     maintainer GPU: learn quill ───┘ (GRPO + DPO, gated, then publish)
```

## What each user does

```bash
scholarlens update                          # newest community Lens + Quill
scholarlens edit paper.tex -m quill:latest  # or any model you prefer
scholarlens review paper_quill/paper.session.json
scholarlens contribute --repo-dir ~/ScholarLens   # optional, public, masked text only → open a PR
```

Your accept/reject decisions are stored only on your machine (`~/.scholarlens/feedback.jsonl`) until you run `contribute`. Before anything is shared, `contribute` does three things:
- **Masks it:** citations, numbers, equations, references, URLs and domain terms are already placeholders.
- **Strips identifying metadata:** endpoints, file names, titles and exact times are removed.
- **Screens for personal data:** entries with emails, phone numbers, ORCIDs and similar are held back.

## What happens automatically

| Trigger | Action | Published only if |
|---|---|---|
| Pull request touching `contributions/` | `scholarlens learn validate --strict` | Every entry passes the schema, placeholder and personal-data checks |
| Merge to `main` with new contributions, or the weekly schedule | `learn.yml` retrains Lens on CPU: base benchmark, plus contributions as training-only human examples, plus hard examples the current Lens misclassifies | On the frozen test split, no attack family's AUROC drops, mean AUROC holds, and no human group's false-positive rate rises |
| Maintainer with a GPU | `scholarlens learn quill` (GRPO on public paragraphs, plus DPO on contributed decisions) | Integrity rate doesn't drop, mean reward improves, and meaning is preserved at least as well |

## Design rules

1. **Quill learns to be a better copy editor, never to fool a detector.** Its rewards are placeholder integrity, meaning preservation, grammar errors actually fixed, restraint, and clean output. `tests/test_rl.py` fails if the RL code imports the detector.
2. **The yardstick is frozen.** Contributions only ever go into training. The test split comes from the published base benchmark, so contributed data can't move the bar it's measured against.
3. **Releases are verifiable.** Each release lists its SHA-256 in the notes, and `update` refuses a mismatch or an archive with unsafe paths. Lens models are joblib files, which can run code when loaded, so install only from the official repository.
4. **Nothing is uploaded without consent.** Feedback stays local until you run `contribute`, and contributions are reviewed in a pull request before they're merged.

## One-time setup for maintainers

1. Build a base benchmark (`docs/LAPTOP.md` or `docs/GPU.md`) and publish it: `scholarlens learn publish bench data/bench/<name>`. This enables `learn.yml`.
2. Publish a first Lens: `scholarlens learn publish lens models/<dir> --force`.
3. Optionally publish a first Quill adapter after `scholarlens learn quill`.
4. In the repository settings, require the CI check on pull requests, and review contribution PRs before merging.
