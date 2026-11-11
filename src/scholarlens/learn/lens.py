"""Continuous learning for Lens.

Each round:
  1. base benchmark (train/val) + community contributions → training data
     (contributions are author-approved Quill edits of the author's own text: human examples)
  2. hard-example mining: texts the current Lens misclassifies get extra weight
  3. train a candidate, evaluate candidate and current Lens on the *frozen* test split
  4. accept only if, for every attack family, AUROC does not drop (beyond a small
     tolerance) and mean AUROC improves or holds, and no human family's false-positive
     rate rises beyond tolerance.

Contributions never enter the test split, so the yardstick cannot be moved by
contributed data. Rejected candidates are reported, not published.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scholarlens.bench.metrics import evaluate
from scholarlens.bench.schema import LABEL_AI, LABEL_HUMAN, Record, read_jsonl
from scholarlens.detect.featurize import Featurizer
from scholarlens.detect.fusion import FusionDetector
from scholarlens.learn.contribute import load_contributions
from scholarlens.rl.rewards import mask_from_masked


def contribution_records(entries: list[dict]) -> list[Record]:
    out = []
    for e in entries:
        if e["decision"] != "accept":
            continue
        pid = f"contrib:{e['id']}"
        for transform, masked in (("contrib_original", e["masked_input"]), ("quill_contrib", e["masked_output"])):
            out.append(Record(id=f"{pid}:{transform}", paper_id=pid, text=mask_from_masked(masked).render(),
                              label=LABEL_HUMAN, source="contribution", transform=transform, split="train"))
    return out


def lens_gate(new: dict, cur: dict, *, auroc_tol: float = 0.005, fpr_tol: float = 0.005) -> tuple[bool, list[str]]:
    reasons = []
    shared = set(new["attacks"]) & set(cur["attacks"])
    for fam in sorted(shared):
        a_new, a_cur = new["attacks"][fam]["auroc"], cur["attacks"][fam]["auroc"]
        if a_new < a_cur - auroc_tol:
            reasons.append(f"{fam}: AUROC {a_new:.4f} < current {a_cur:.4f}")
    if shared:
        m_new = np.mean([new["attacks"][f]["auroc"] for f in shared])
        m_cur = np.mean([cur["attacks"][f]["auroc"] for f in shared])
        if m_new < m_cur:
            reasons.append(f"mean AUROC {m_new:.4f} < current {m_cur:.4f}")
    for fam in sorted(set(new["human_fpr"]) & set(cur["human_fpr"])):
        f_new, f_cur = new["human_fpr"][fam]["fpr"], cur["human_fpr"][fam]["fpr"]
        if f_new > f_cur + fpr_tol:
            reasons.append(f"{fam}: false-positive rate {f_new:.4f} > current {f_cur:.4f}")
    return not reasons, reasons


def improve_lens(*, bench: str | Path, repo_dir: str | Path, out_dir: str | Path, featurizer: Featurizer,
                 current_dir: str | Path | None = None, hard_weight: float = 2.0, fpr: float = 0.01, log=print) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = list(read_jsonl(bench))
    train = [r for r in records if r.split == "train"]
    val = [r for r in records if r.split == "val"]
    test = [r for r in records if r.split == "test"]
    contrib, notes = load_contributions(repo_dir)
    for n in notes:
        log(f"  contribution skipped: {n}")
    extra = contribution_records(contrib)
    train += extra
    log(f"train {len(train)} (+{len(extra)} from {len(contrib)} contributions) · val {len(val)} · frozen test {len(test)}")

    current = None
    weights: dict[str, float] = {}
    if current_dir and (Path(current_dir) / "fusion" / "meta.json").is_file():
        current = FusionDetector.load(Path(current_dir) / "fusion", featurizer)
        p = current.predict_proba(train)
        y = np.array([r.label == LABEL_AI for r in train])
        wrong = (p >= 0.5) != y
        weights = {r.id: hard_weight for r, w in zip(train, wrong, strict=True) if w}
        log(f"hard examples (misclassified by current Lens): {int(wrong.sum())}")

    candidate = FusionDetector(featurizer).fit(train, val, sample_weight=weights or None)
    ids = [r.id for r in test]
    rep_new = evaluate(test, dict(zip(ids, candidate.predict_proba(test).tolist(), strict=True)), fpr=fpr)
    if current is not None:
        rep_cur = evaluate(test, dict(zip(ids, current.predict_proba(test).tolist(), strict=True)), fpr=fpr)
        accepted, reasons = lens_gate(rep_new, rep_cur)
    else:
        rep_cur = None
        bad = [f for f, row in rep_new["attacks"].items() if not row["auroc"] > 0.6]
        accepted, reasons = not bad, [f"{f}: AUROC not above 0.6" for f in bad]
    candidate.train_info.update({"n_contributions": len(contrib), "n_hard_examples": len(weights)})
    candidate.save(out / "fusion")
    report = {"accepted": accepted, "reasons": reasons, "candidate": rep_new, "current": rep_cur,
              "n_contributions": len(contrib), "n_hard_examples": len(weights), "bench": str(bench)}
    (out / "learn_report.json").write_text(json.dumps(report, indent=2, default=lambda o: list(o) if isinstance(o, tuple) else str(o)))
    return report
