"""Train and evaluate Lens detectors on a Bench file; write a report.

    report.json   full metrics per detector (robustness + fairness)
    report.md     human-readable tables
    fusion/       saved fusion model (+ gnn/ if trained)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scholarlens.bench.metrics import evaluate
from scholarlens.bench.schema import LABEL_AI, Record, read_jsonl
from scholarlens.detect.featurize import Featurizer
from scholarlens.detect.fusion import FusionDetector
from scholarlens.detect.retrieval import RetrievalDetector
from scholarlens.detect.zero_shot import baseline_scores


def train_and_evaluate(bench: str | Path, out: str | Path, *, lm=None, embedder=None, gnn: bool = False,
                       binoculars=None, retrieval: bool = True, exclude_families: tuple[str, ...] = (),
                       estimator: str = "hgb", fpr: float = 0.01, log=print) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    records = list(read_jsonl(bench))
    split = {s: [r for r in records if r.split == s] for s in ("train", "val", "test")}
    log("records: " + ", ".join(f"{k}={len(v)}" for k, v in split.items()))
    if not split["train"] or not split["test"]:
        raise ValueError("bench needs train and test records")

    featurizer = Featurizer(lm=lm, embedder=embedder, cache=out / "features.jsonl")
    progress = lambda i, n: log(f"  features {i}/{n}")  # noqa: E731
    fusion = FusionDetector(featurizer, estimator=estimator).fit(
        split["train"], split["val"], exclude_families=exclude_families, on_progress=progress)
    fusion.save(out / "fusion")
    test = split["test"]
    scores: dict[str, dict[str, float]] = {}
    scores["lens-fusion"] = dict(zip([r.id for r in test], fusion.predict_proba(test).tolist(), strict=True))

    X, names = featurizer.featurize([(r.id, r.text) for r in test])
    for name, s in baseline_scores(X, names).items():
        scores[name] = dict(zip([r.id for r in test], s.tolist(), strict=True))

    if retrieval and embedder is not None:
        index = [r for r in records if r.label == LABEL_AI and r.transform == "none"]
        det = RetrievalDetector(embedder)
        det.add([r.text for r in index], [r.id for r in index])
        sims, _ = det.search([r.text for r in test])
        scores["retrieval"] = dict(zip([r.id for r in test], sims.tolist(), strict=True))

    if binoculars is not None:
        scores["binoculars"] = dict(zip([r.id for r in test], binoculars.score([r.text for r in test]).tolist(),
                                        strict=True))

    if gnn and embedder is not None:
        from scholarlens.detect.graph import GNNDetector

        g = GNNDetector(embedder).fit([r for r in split["train"] if r.family not in exclude_families],
                                      split["val"], log=log)
        g.save(out / "gnn")
        scores["lens-gnn"] = dict(zip([r.id for r in test], g.score_records(test).tolist(), strict=True))

    reports = {name: evaluate(test, s, fpr=fpr) for name, s in scores.items()}
    importance = fusion.importance(split["val"] or test)
    result = {"bench": str(bench), "fpr": fpr, "train_info": fusion.train_info, "detectors": reports,
              "fusion_feature_importance": importance}
    (out / "report.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")
    (out / "report.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def render_markdown(result: dict) -> str:
    fpr = result["fpr"]
    key = f"tpr@{fpr:g}fpr"
    dets = result["detectors"]
    attack_names = sorted({a for r in dets.values() for a in r["attacks"]})
    human_names = sorted({h for r in dets.values() for h in r["human_fpr"]})
    lines = [f"# Lens evaluation — {result['bench']}", "",
             f"Test split. Threshold per detector set so clean human text has FPR ≤ {fpr:.0%}.", "",
             "## AI detection under attack (AUROC / TPR@" + f"{fpr:.0%} FPR)", "",
             "| detector | " + " | ".join(attack_names) + " |", "|---" * (len(attack_names) + 1) + "|"]
    for name, r in dets.items():
        cells = []
        for a in attack_names:
            row = r["attacks"].get(a)
            cells.append(f"{row['auroc']:.3f} / {row[key]:.2f}" if row else "—")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## False-positive rate on human text (lower is better)", "",
              "| detector | " + " | ".join(human_names) + " |", "|---" * (len(human_names) + 1) + "|"]
    for name, r in dets.items():
        lines.append(f"| {name} | " + " | ".join(
            f"{r['human_fpr'][h]['fpr']:.3f} (n={r['human_fpr'][h]['n']})" if h in r["human_fpr"] else "—"
            for h in human_names) + " |")
    imp = list(result["fusion_feature_importance"].items())[:12]
    lines += ["", "## Top fusion features (permutation importance, ΔAUROC)", ""]
    lines += [f"- `{k}`: {v:+.4f}" for k, v in imp]
    lines += ["", f"Training: {json.dumps(result['train_info'])}", ""]
    return "\n".join(lines)


def records_from_texts(texts: list[str]) -> list[Record]:
    return [Record(id=f"q{i}", paper_id=f"q{i}", text=t, label="human", source="query") for i, t in enumerate(texts)]
