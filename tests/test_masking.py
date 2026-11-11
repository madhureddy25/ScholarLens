from pathlib import Path

import pytest

from scholarlens.analysis import prose_units
from scholarlens.ingest import load
from scholarlens.masking import IntegrityError, Kind, MaskConfig, Masker

FIX = Path(__file__).parent / "fixtures"
M = Masker()


def symbols(text, fmt="text", masker=M):
    r = masker.mask(text, fmt)
    return r, {v: k.split("_")[0].strip("⟦") for k, v in r.symbols.items()}


def test_blueprint_example():
    text = ("According to recent findings (Smith et al., 2024), the activation of CRISPR-Cas9 "
            "endonuclease complexes induces double-strand breaks at target genomic loci with high specificity.")
    r, sym = symbols(text)
    assert sym == {"(Smith et al., 2024)": "CITE", "CRISPR-Cas9": "TERM"}
    assert r.masked.startswith("According to recent findings ⟦CITE_1⟧, the activation of ⟦TERM_1⟧")
    assert r.unmask() == text


@pytest.mark.parametrize("cite", [
    "[3]", "[1, 2]", "[5-7]", "[1–3, 8]", "(Smith, 2020)", "(Smith & Jones, 2020; Lee, 2019a)",
    "(e.g., Smith et al., 2021)", "(see van der Berg et al., 2018, p. 4)", "(Lee, in press)",
])
def test_parenthetical_and_numeric_citations(cite):
    _, sym = symbols(f"This was shown before {cite} in detail.")
    assert sym.get(cite) == "CITE"


@pytest.mark.parametrize("cite", ["Smith et al. (2024)", "Smith and Jones (2020)", "Vaswani et al. (2017, p. 3)"])
def test_narrative_citations(cite):
    _, sym = symbols(f"As {cite} argued, models scale.")
    assert sym.get(cite) == "CITE"


@pytest.mark.parametrize("ref", ["Figure 3", "Fig. 2a", "Figs. 2 and 3", "Table S2", "Eq. (4)",
                                 "Section 3.2", "Appendix A", "Supplementary Table 1"])
def test_cross_references(ref):
    _, sym = symbols(f"Results are in {ref} below.")
    assert sym.get(ref) == "REF"


def test_paren_not_swallowed_by_reference():
    r, sym = symbols("Results improve (see Figure 3) overall.")
    assert sym["Figure 3"] == "REF" and r.masked == "Results improve (see ⟦REF_1⟧) overall."


@pytest.mark.parametrize("q", ["p < 0.05", "n = 1,200", "3.2 ± 0.4", "95%", "12,000", "-0.73", "1.5e-3"])
def test_quantities(q):
    _, sym = symbols(f"We observed {q} in the cohort.")
    assert sym.get(q) == "NUM"


@pytest.mark.parametrize("term", ["BRCA1", "IL-6", "SARS-CoV-2", "GPT-2", "COVID-19", "mRNA", "SciBERT",
                                  "NaCl", "H2O", "TNF-α", "E. coli", "DNA"])
def test_hard_terms(term):
    _, sym = symbols(f"Expression of {term} increased.")
    assert sym.get(term) == "TERM"


def test_ordinary_words_not_masked():
    r, _ = symbols("However, the results were robust across all settings and seeds.")
    assert r.spans == []


def test_glossary_terms():
    masker = Masker(MaskConfig(glossary=("double-strand breaks", "genomic loci")))
    r, sym = symbols("It induces double-strand breaks at genomic loci.", masker=masker)
    assert sym == {"double-strand breaks": "TERM", "genomic loci": "TERM"}


def test_latex_constructs():
    text = (r"As in Fig.~\ref{fig:a} and \citet{lee2020}, loss fell to $0.3$ from 70.3\% "
            r"with \textit{E. coli} (Section~\ref{sec:x}). Schr\"odinger. % todo")
    r, sym = symbols(text, fmt="latex")
    assert sym["Fig.~\\ref{fig:a}"] == "REF"
    assert sym["\\citet{lee2020}"] == "CITE"
    assert sym["$0.3$"] == "MATH"
    assert sym["70.3\\%"] == "NUM"
    assert sym["\\textit{E. coli}"] == "CMD"
    assert sym["Section~\\ref{sec:x}"] == "REF"
    assert sym['Schr\\"odinger'] == "CMD"
    assert sym["% todo"] == "CMD"
    assert r.unmask() == text


def test_literal_sentinel_chars_in_source_are_protected():
    text = "A weird ⟦bracket⟧ appears."
    r = M.mask(text)
    assert r.unmask() == text and r.verify(r.masked).ok


def test_mask_unmask_identity_on_whole_paper():
    doc = load(FIX / "sample_paper.tex")
    for u in prose_units(doc):
        assert u.mask.unmask() == doc.get(u.node_id).text


# ---------------------------------------------------------- integrity checks
@pytest.fixture
def cited():
    return M.mask("Prior work (Smith, 2020) and [4] used BRCA1 with p < 0.05.")


def test_verify_ok_when_reworded(cited):
    edited = "Using ⟦TERM_1⟧, earlier studies ⟦CITE_1⟧ ⟦CITE_2⟧ reported ⟦NUM_1⟧."
    rep = cited.verify(edited)
    assert rep.ok and not rep.citation_order_changed
    assert cited.unmask(edited) == "Using BRCA1, earlier studies (Smith, 2020) [4] reported p < 0.05."


def test_verify_detects_missing_duplicate_unknown(cited):
    edited = "Work ⟦CITE_1⟧ ⟦CITE_1⟧ used ⟦TERM_1⟧ with ⟦NUM_1⟧ and ⟦CITE_9⟧."
    rep = cited.verify(edited)
    assert not rep.ok
    assert rep.missing == ["⟦CITE_2⟧"] and rep.duplicated == ["⟦CITE_1⟧"] and rep.unknown == ["⟦CITE_9⟧"]
    with pytest.raises(IntegrityError):
        cited.unmask(edited)


def test_citation_reorder_is_warning_not_error(cited):
    rep = cited.verify("⟦CITE_2⟧ and ⟦CITE_1⟧ used ⟦TERM_1⟧ with ⟦NUM_1⟧.")
    assert rep.ok and rep.citation_order_changed and rep.warnings


def test_repair_damaged_sentinels(cited):
    damaged = "Work [CITE_1] and <CITE_2> used TERM_1 with ⟦NUM_1⟧."
    assert not cited.verify(damaged).ok
    fixed, repairs = cited.repair(damaged)
    assert len(repairs) == 3
    assert cited.verify(fixed).ok
    assert cited.unmask(fixed) == "Work (Smith, 2020) and [4] used BRCA1 with p < 0.05."


def test_latex_brace_balance_checked():
    r = M.mask(r"Text with {\bf bold} words.", fmt="latex")
    assert r.verify(r.masked).ok
    assert r.verify(r.masked + " {").brace_imbalance == 1


def test_soft_terms_verified_not_masked():
    masker = Masker(soft_terms=lambda text: ["double-strand breaks"])
    r = masker.mask("Cas9 induces double-strand breaks efficiently.")
    assert "double-strand breaks" in r.masked
    assert r.verify(r.masked).soft_terms_missing == []
    rep = r.verify(r.masked.replace("double-strand breaks", "DNA damage"))
    assert rep.ok and rep.soft_terms_missing == ["double-strand breaks"]


def test_render_for_scoring_strips_markup():
    r = M.mask(r"As in Fig.~\ref{a}, \textit{E. coli} grew~\cite{x} with $x^2$ (p < 0.05).", fmt="latex")
    assert r.render() == "As in Figure 1, E. coli grew [1] with x (p < 0.05)."


def test_mask_config_disables_kinds():
    r = Masker(MaskConfig(quantities=False, terms=False)).mask("BRCA1 rose by 12% [3].")
    assert {s.kind for s in r.spans} == {Kind.CITE}
