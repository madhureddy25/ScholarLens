import importlib.util
import math

import numpy as np
import pytest

from scholarlens.analysis import analyze, prose_units, segments
from scholarlens.core.text import split_sentences, word_windows, words
from scholarlens.features import burstiness, extract_features, stylometric_features
from scholarlens.features.transitions import (
    HashingEmbedder,
    sentence_graph,
    transition_deviation,
    transition_features,
    transition_matrix,
)
from scholarlens.ingest import parse_markdown


# ------------------------------------------------------------ sentence split
@pytest.mark.parametrize("text,expected", [
    ("We used GPT-2. It worked.", 2),
    ("As Smith et al. showed, it works. Next sentence.", 2),
    ("See Fig. 3 for details. Also Eq. 4 holds.", 2),
    ("Values were 3.14 and 2.71. Done.", 2),
    ("Methods (e.g. PCA) were used. Results follow.", 2),
    ("Is it robust? Yes! It is.", 3),
    ("He said “it works.” Then left.", 2),
    ("Masked ⟦CITE_1⟧. ⟦TERM_1⟧ binds.", 2),
])
def test_sentence_split(text, expected):
    assert len(split_sentences(text)) == expected


def test_words_treat_sentinels_as_single_tokens():
    assert words("⟦CITE_1⟧ shows state-of-the-art BRCA1 results") == [
        "⟦CITE_1⟧", "shows", "state-of-the-art", "BRCA1", "results"]


def test_word_windows_cover_everything_and_overlap():
    sents = [" ".join(["w"] * 50)] * 12  # 600 words
    wins = word_windows(sents, size=250)
    assert wins[0][0] == 0 and wins[-1][1] == 12
    assert all(b > a for a, b in wins)
    assert any(w2[0] < w1[1] for w1, w2 in zip(wins, wins[1:]))  # overlapping


# ---------------------------------------------------------------- burstiness
def test_burstiness_formula():
    lengths = [5, 25, 10, 40]
    mu, sigma = np.mean(lengths), np.std(lengths)
    assert burstiness(lengths) == pytest.approx((sigma - mu) / (sigma + mu))
    assert burstiness([20, 20, 20]) == pytest.approx(-1.0)  # perfectly uniform
    assert burstiness([]) == 0.0


def test_stylometry_detects_connectives_and_markers():
    templated = ("Furthermore, the model is robust. Moreover, it is efficient. "
                 "Additionally, results delve into the intricate landscape.")
    plain = "The model is robust. It runs fast. We report results in the table."
    t, p = stylometric_features(templated), stylometric_features(plain)
    assert t["connective_start_rate"] == pytest.approx(1.0) and p["connective_start_rate"] == 0
    assert t["llm_marker_per_1k"] > 0 and p["llm_marker_per_1k"] == 0


# --------------------------------------------------------------- transitions
def test_transition_deviation_matches_definition():
    rng = np.random.default_rng(0)
    T = transition_matrix(rng.normal(size=(6, 16)))
    D = transition_deviation(T)
    M = T.shape[0]
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            expected = T[i, j] - sum(T[i, k] for k in range(M) if k != i) / (M - 1)
            assert D[i, j] == pytest.approx(expected)


def test_transition_matrix_is_cosine():
    T = transition_matrix(np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]))
    assert T[0, 1] == pytest.approx(0.0) and T[0, 2] == pytest.approx(1 / math.sqrt(2))
    assert np.allclose(np.diag(T), 1.0)


def test_transition_features_and_graph():
    sents = ["Cells were cultured.", "Cells were then imaged.", "Imaging used confocal microscopy.",
             "Statistics used t-tests.", "Cells were counted."]
    emb = HashingEmbedder()
    f = transition_features(sents, emb)
    assert set(f) >= {"adj_sim_mean", "offdiag_sim_var", "delta_abs_mean"}
    assert transition_features(sents[:2], emb)["adj_sim_mean"] == 0.0  # too short
    g = sentence_graph(sents, emb, knn=2)
    assert g.number_of_nodes() == 5 and g.has_edge(0, 1)


def test_hashing_embedder_deterministic():
    e = HashingEmbedder(256)
    assert np.array_equal(e.encode(["a b c"]), e.encode(["a b c"]))


# ------------------------------------------------------------------ pipeline
def test_analyze_document_end_to_end():
    md = "# Intro\n\n" + "\n\n".join(
        f"Paragraph {i} studies BRCA1 [{i}]. It reports p < 0.05 in cohort {i}. Results were consistent."
        for i in range(1, 40))
    doc = parse_markdown(md)
    report = analyze(doc, segment_size=120)
    assert report["n_prose_units"] == 39
    assert report["protected_spans"]["CITE"] == 39
    assert len(report["segments"]) > 2
    segs = segments(prose_units(doc), 120)
    assert all(s.n_words >= 100 for s in segs[:-1])
    assert "burstiness" in report["document_features"]


@pytest.mark.slow
@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="transformers not installed")
def test_lm_scorer_gpt2_sliding_window():
    from scholarlens.features.lm import CausalLMScorer

    try:
        lm = CausalLMScorer("gpt2", max_length=64, stride=32)
    except OSError:
        pytest.skip("gpt2 weights unavailable")
    text = " ".join(["The cell cycle is regulated by cyclins and kinases."] * 20)
    n_ids = len(lm.tok(text).input_ids)
    stats = lm.score(text)
    assert stats.n_tokens == n_ids  # every token scored exactly once (BOS prepended)
    assert stats.ppl > 1 and stats.entropy_mean > 0
    # repeated text is highly predictable → strongly positive Fast-DetectGPT score
    assert stats.fast_detectgpt > 2
    feats = extract_features(text, lm=lm)
    assert "lm_fast_detectgpt" in feats
