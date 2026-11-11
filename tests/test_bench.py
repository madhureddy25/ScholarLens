import json

import numpy as np
import pytest

from scholarlens.bench.builder import build, summarize
from scholarlens.bench.generate import generate_counterpart
from scholarlens.bench.metrics import auroc, ece, evaluate, rate_above, threshold_at_fpr, tpr_at_fpr
from scholarlens.bench.schema import Record, read_jsonl, stable_id, write_jsonl
from scholarlens.bench.sources import Filters, JsonlSource, arxiv_domain, arxiv_year
from scholarlens.bench.splits import assign_splits, leakage, split_of
from scholarlens.bench.transforms import LLMParaphraser, QuillTransform
from scholarlens.edit.prompts import extract_paragraph
from scholarlens.llm.testing import ScriptedModel

ABSTRACT = ("We study the problem of detecting machine generated text in scientific abstracts. "
            "Our approach combines statistical signals with sentence transition structure. "
            "Experiments on 12,000 abstracts show an AUROC of 0.91 across four domains. ") * 3


def human(i=0, **kw):
    pid = f"arxiv:{i}"
    return Record(id=stable_id(pid, "human"), paper_id=pid, text=ABSTRACT, label="human", source="arxiv",
                  domain=kw.get("domain", "cs"), year=2019, title=f"Paper {i}")


# ------------------------------------------------------------------ metrics
def test_auroc_matches_sklearn_with_ties():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.normal(size=300) + y, 1)  # rounding creates ties
    assert auroc(y, s) == pytest.approx(sk.roc_auc_score(y, s))


def test_threshold_controls_fpr():
    neg = np.arange(1000) / 1000
    t = threshold_at_fpr(neg, 0.01)
    assert rate_above(neg, t) <= 0.01 and rate_above(neg, t - 1e-3) > 0.01
    y = np.r_[np.zeros(1000), np.ones(100)]
    s = np.r_[neg, np.full(100, 2.0)]
    assert tpr_at_fpr(y, s, 0.01) == 1.0


def test_ece_perfect_and_bad():
    y = np.array([0, 0, 1, 1])
    assert ece(y, y.astype(float)) == 0
    assert ece(y, 1 - y.astype(float)) == pytest.approx(1.0)


def test_evaluate_report_structure():
    recs, scores = [], {}
    for i in range(200):
        h = human(i, domain="cs" if i % 2 else "bio")
        a = h.derive("ai text", transform="none")
        a.label, a.transform, a.id = "ai", "none", f"ai{i}"
        p = a.derive("para", transform="llm_paraphrase")
        q = h.derive("edited", transform="quill_proofread")
        recs += [h, a, p, q]
        scores.update({h.id: i / 400, a.id: 0.9 + i / 4000, p.id: 0.3 + i / 1000, q.id: i / 380})
    rep = evaluate(recs, scores, ci=False)
    assert rep["attacks"]["ai"]["auroc"] == 1.0
    assert rep["attacks"]["ai+llm_paraphrase"]["auroc_drop"] > 0
    assert set(rep["human_fpr"]) == {"human", "human+quill_proofread"}
    assert rep["human_fpr"]["human"]["fpr"] <= 0.01
    assert rep["human_fpr"]["human+quill_proofread"]["fpr"] > rep["human_fpr"]["human"]["fpr"]
    assert set(rep["human_fpr_by_domain"]) == {"cs", "bio"}


# ------------------------------------------------------------- splits/schema
def test_paper_level_splits_deterministic_and_leak_free():
    recs = assign_splits([human(i) for i in range(500)])
    assert {r.split for r in recs} == {"train", "val", "test"}
    assert split_of("arxiv:7") == split_of("arxiv:7")
    counts = {s: sum(r.split == s for r in recs) for s in ("train", "val", "test")}
    assert 350 < counts["train"] < 450
    derived = [r.derive("x", transform="t") for r in recs]
    assert not leakage(recs + derived)
    bad = recs[0].derive("y", transform="u")
    bad.split = "test" if recs[0].split != "test" else "train"
    assert recs[0].paper_id in leakage(recs + [bad])


def test_jsonl_roundtrip(tmp_path):
    recs = [human(i) for i in range(3)]
    write_jsonl(recs, tmp_path / "r.jsonl")
    assert list(read_jsonl(tmp_path / "r.jsonl")) == recs


# ------------------------------------------------------------------ sources
@pytest.mark.parametrize("aid,year", [("0704.0001", 2007), ("2112.01234", 2021), ("hep-th/9901001", 1999),
                                      ("math/0211159", 2002), ("garbage", None)])
def test_arxiv_year(aid, year):
    assert arxiv_year(aid) == year


@pytest.mark.parametrize("cats,dom", [("['hep-ph']", "physics"), ("['cs.CL cs.LG']", "cs"),
                                      ("['q-bio.NC']", "q-bio"), ("['math.CO cs.CG']", "math")])
def test_arxiv_domain(cats, dom):
    assert arxiv_domain(cats) == dom


def test_jsonl_source_filters_cutoff_and_length(tmp_path):
    rows = [{"id": 1, "text": ABSTRACT, "year": 2019, "title": "A"},
            {"id": 2, "text": ABSTRACT, "year": 2023, "title": "B"},   # after cutoff
            {"id": 3, "text": "too short", "year": 2018, "title": "C"},
            {"id": 4, "text": ABSTRACT, "title": "D"}]                  # unknown year → excluded
    p = tmp_path / "pmc.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    got = list(JsonlSource(p).records(Filters(before_year=2022, min_words=20)))
    assert [r.paper_id for r in got] == ["pmc:1"]


# ------------------------------------------------------- generation/transforms
def test_generate_counterpart_matches_topic():
    model = ScriptedModel(["  A generated   abstract about detection.  "])
    rec = generate_counterpart(human(1), model, style="title")
    assert rec.label == "ai" and rec.paper_id == "arxiv:1" and rec.text == "A generated abstract about detection."
    prompt = model.calls[0]["messages"][0]["content"]
    assert "Paper 1" in prompt and "cs paper" in prompt
    assert generate_counterpart(human(1), ScriptedModel(["<<REFUSE>>"])) is None


def test_first_sentence_style_prepends_human_opening():
    rec = generate_counterpart(human(2), ScriptedModel(["The rest."]), style="title_first_sentence")
    assert rec.text.startswith("We study the problem") and rec.text.endswith("The rest.")


def test_llm_paraphraser_and_quill_transform():
    h = human(3)
    p = LLMParaphraser(ScriptedModel(["Reworded text."])).apply(h)
    assert p.transform == "llm_paraphrase" and p.parent_id == h.id and p.paper_id == h.paper_id
    q = QuillTransform(ScriptedModel(lambda u: extract_paragraph(u).replace("We study", "We examine"))).apply(h)
    assert q.transform == "quill_proofread" and "We examine" in q.text and q.meta["quill_status"] == "accepted"
    assert "12,000" in q.text and "0.91" in q.text


# ------------------------------------------------------------------ builder
def test_build_end_to_end_and_resume(tmp_path):
    src = tmp_path / "src.jsonl"
    src.write_text("\n".join(json.dumps({"id": i, "text": ABSTRACT, "year": 2015, "title": f"T{i}"})
                             for i in range(30)))
    calls = {"n": 0}

    def fake_load(spec, **kw):
        def respond(u):
            calls["n"] += 1
            return (extract_paragraph(u).replace("We study", "We examine") if "<paragraph>" in u
                    else f"Generated abstract number {calls['n']}.")
        return ScriptedModel(respond, model=spec)

    cfg = {"bench": {"out": str(tmp_path / "out"), "n_human": 20, "min_words": 20},
           "sources": [{"type": "jsonl", "path": str(src)}],
           "generators": [{"model": "fake-a"}, {"model": "fake-b"}],
           "transforms": [{"type": "llm_paraphrase", "model": "fake-p", "apply_to": "ai"},
                          {"type": "quill", "model": "fake-q", "mode": "proofread", "apply_to": "human", "limit": 5}]}
    m = build(cfg, log=lambda *_: None, load_model=fake_load)
    fams = m["by_family_and_split"]
    assert sum(fams["human"].values()) == 20 and sum(fams["ai"].values()) == 20
    assert sum(fams["ai+llm_paraphrase"].values()) == 20
    assert sum(fams["human+quill_proofread"].values()) == 5
    assert not m["leakage"] and set(m["generators"]) == {"scripted:fake-a", "scripted:fake-b"}

    before = calls["n"]
    build(cfg, log=lambda *_: None, load_model=fake_load)  # everything cached → no new model calls
    assert calls["n"] == before
    all_recs = list(read_jsonl(tmp_path / "out" / "bench.jsonl"))
    assert summarize(all_recs)["n_records"] == 65
