import json
import random

import numpy as np
import pytest

from scholarlens.bench.metrics import auroc
from scholarlens.bench.schema import Record, write_jsonl
from scholarlens.bench.splits import assign_splits
from scholarlens.detect import Featurizer, FusionDetector, RetrievalDetector, baseline_scores
from scholarlens.features.transitions import HashingEmbedder

pytest.importorskip("sklearn")

TOPICS = ["protein folding", "graph neural networks", "dark matter halos", "soil microbiomes", "option pricing",
          "speech recognition", "coral bleaching", "quantum error correction", "urban mobility", "sleep apnea"]
VERBS = ["measured", "estimated", "found", "observed", "tested", "compared", "derived", "modelled"]


def human_text(rng: random.Random, topic: str) -> str:
    sents = []
    for _ in range(rng.randint(4, 8)):
        n = rng.choice([5, 9, 14, 22, 31])
        words = [rng.choice(["we", "the", "data", topic.split()[0], "results", "samples", "noise", "trend",
                             "model", "field", "error", "rate", "site", "year"]) for _ in range(n)]
        sents.append(" ".join(words).capitalize() + rng.choice([".", ".", ";", ":"]).replace(";", ".").replace(":", "."))
    return " ".join(sents)


def ai_text(rng: random.Random, topic: str) -> str:
    opener = ["Furthermore", "Moreover", "Additionally", "Notably", "Overall"]
    sents = [f"This study delves into {topic} and its pivotal role in the broader landscape."]
    for i in range(rng.randint(4, 6)):
        sents.append(f"{opener[i % 5]}, we {rng.choice(VERBS)} key aspects of {topic} with a comprehensive framework.")
    return " ".join(sents)


def synthetic_bench(n=120, seed=0) -> list[Record]:
    rng = random.Random(seed)
    recs = []
    for i in range(n):
        topic = TOPICS[i % len(TOPICS)]
        pid = f"p{i}"
        h = Record(id=f"h{i}", paper_id=pid, text=human_text(rng, topic), label="human", source="syn",
                   domain="cs" if i % 2 else "bio")
        a = Record(id=f"a{i}", paper_id=pid, text=ai_text(rng, topic), label="ai", source="syn", generator="g",
                   parent_id=h.id)
        para = a.derive(a.text.replace("Furthermore", "Also").replace("delves into", "explores"),
                        transform="llm_paraphrase")
        recs += [h, a, para]
    return assign_splits(recs)


def test_fusion_learns_and_calibrates(tmp_path):
    recs = synthetic_bench()
    tr = [r for r in recs if r.split == "train"]
    va = [r for r in recs if r.split == "val"]
    te = [r for r in recs if r.split in ("test", "val")]
    det = FusionDetector(Featurizer(cache=tmp_path / "f.jsonl")).fit(tr, va)
    p = det.predict_proba(te)
    y = np.array([r.label == "ai" for r in te], dtype=int)
    assert auroc(y, p) > 0.95
    assert ((0 <= p) & (p <= 1)).all()
    assert "n_words" not in det.feature_names  # length shortcuts removed
    det.save(tmp_path / "m")
    again = FusionDetector.load(tmp_path / "m", Featurizer(cache=tmp_path / "f.jsonl"))
    assert np.allclose(again.predict_proba(te), p)
    assert (tmp_path / "f.jsonl").stat().st_size > 0  # features cached


def test_exclude_families_holds_out_attack():
    recs = synthetic_bench(60)
    tr = [r for r in recs if r.split == "train"]
    det = FusionDetector(Featurizer(), calibration=None).fit(tr, exclude_families=("ai+llm_paraphrase",))
    assert "ai+llm_paraphrase" not in det.train_info["families"]


def test_retrieval_finds_paraphrase_source():
    rng = random.Random(1)
    originals = [ai_text(rng, t) for t in TOPICS]
    det = RetrievalDetector(HashingEmbedder())
    det.add(originals, [f"g{i}" for i in range(len(originals))])
    para = originals[3].replace("Moreover", "In addition").replace("comprehensive", "thorough")
    sims, ids = det.search([para, human_text(rng, "coral bleaching")])
    assert ids[0] == "g3" and sims[0] > 0.7 and sims[0] > sims[1]
    s, ids2 = det.search([originals[3]], exclude=[{"g3"}])
    assert ids2[0] != "g3"


def test_baseline_scores_sign_convention():
    X = np.array([[2.0, 3.0], [1.0, 5.0]])
    out = baseline_scores(X, ["lm_fast_detectgpt", "lm_log_ppl"])
    assert np.allclose(out["fast_detectgpt"], [2.0, 1.0]) and np.allclose(out["neg_log_ppl"], [-3.0, -5.0])


def test_gnn_trains_on_sentence_graphs():
    pytest.importorskip("torch")
    from scholarlens.detect.graph import GNNDetector, build_graph

    X, A = build_graph(["a b c.", "d e f.", "a b g."], np.eye(3, 8), knn=1)
    assert X.shape == (3, 10) and A[0, 1] == 1 and np.allclose(A, A.T)
    recs = synthetic_bench(60)
    tr = [r for r in recs if r.split == "train"]
    va = [r for r in recs if r.split != "train"]
    g = GNNDetector(HashingEmbedder(256), epochs=15, hidden=32).fit(tr, va)
    s = g.score_records(va)
    y = np.array([r.label == "ai" for r in va], dtype=int)
    assert s.shape == (len(va),) and auroc(y, s) > 0.8


def test_runner_writes_report(tmp_path):
    from scholarlens.detect.runner import train_and_evaluate

    write_jsonl(synthetic_bench(), tmp_path / "bench.jsonl")
    res = train_and_evaluate(tmp_path / "bench.jsonl", tmp_path / "out", embedder=HashingEmbedder(), log=lambda *_: None)
    assert {"lens-fusion", "retrieval"} <= set(res["detectors"])
    fusion = res["detectors"]["lens-fusion"]
    assert {"ai", "ai+llm_paraphrase"} <= set(fusion["attacks"]) and "human" in fusion["human_fpr"]
    assert (tmp_path / "out" / "report.md").read_text().startswith("# Lens evaluation")
    json.loads((tmp_path / "out" / "report.json").read_text())
