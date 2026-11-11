from pathlib import Path

import pytest

from scholarlens.core.document import NodeType, ReconstructionError
from scholarlens.ingest import load, parse_latex, parse_markdown, parse_text

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def paper():
    return load(FIX / "sample_paper.tex")


# ------------------------------------------------------------------ LaTeX
def test_latex_roundtrip_is_byte_exact(paper):
    assert paper.reconstruct({}) == paper.source


def test_latex_title_and_preamble(paper):
    assert paper.meta["title"].startswith("Structure-Aware Detection")
    assert paper.nodes[0].type == NodeType.METADATA
    assert "\\usepackage{amsmath}" in paper.nodes[0].text


def test_latex_block_types(paper):
    types = [n.type for n in paper.nodes]
    assert types.count(NodeType.EQUATION) == 2  # equation env + \[ \]
    assert NodeType.FIGURE in types and NodeType.TABLE in types
    assert NodeType.CODE in types
    assert types.count(NodeType.BIBLIOGRAPHY) == 2


def test_latex_abstract_is_prose_with_section(paper):
    abstract = [n for n in paper.prose() if n.section == "Abstract"]
    assert len(abstract) == 1
    assert abstract[0].text.startswith("Large language models")
    assert "\\end{abstract}" not in abstract[0].text


def test_latex_paragraph_split_on_blank_line(paper):
    intro = [n for n in paper.prose() if n.section == "Introduction" and n.type == NodeType.PARAGRAPH]
    assert len(intro) == 2
    assert intro[1].text.startswith("We evaluate on 12,000")


def test_latex_lists_including_nested(paper):
    items = [n.text for n in paper.nodes if n.type == NodeType.LIST_ITEM]
    assert items == [
        "A benchmark of human, AI and AI-paraphrased abstracts (Section~\\ref{sec:data}).",
        "A detector that is robust to paraphrase attacks:",
        "statistical features, and",
        "sentence-transition graph features.",
    ]


def test_latex_captions_are_prose_inside_floats(paper):
    caps = [n for n in paper.nodes if n.type == NodeType.CAPTION]
    assert [c.text for c in caps] == [
        "Overview of the pipeline. Prose is masked before scoring~\\cite{bao2024}.",
        "Detection results (AUROC).",
    ]
    fig = paper.get(caps[0].meta["parent"])
    assert fig.type == NodeType.FIGURE and fig.start < caps[0].start < caps[0].end < fig.end


def test_latex_verbatim_with_stray_braces_is_code(paper):
    code = [n for n in paper.nodes if n.type == NodeType.CODE]
    assert len(code) == 1 and "stray } braces {" in code[0].text


def test_latex_unknown_environment_passed_through(paper):
    raw = [n for n in paper.nodes if n.type == NodeType.RAW and n.meta.get("env") == "theorem"]
    assert len(raw) == 1


def test_latex_headings_carry_own_section(paper):
    heads = [n for n in paper.nodes if n.type == NodeType.HEADING]
    assert [h.section for h in heads] == ["Introduction", "Contributions", "Method", "Conclusion"]
    assert all(h.section == h.meta["title"] for h in heads)


def test_latex_replace_only_touches_target_span(paper):
    target = next(n for n in paper.prose() if n.section == "Conclusion")
    out = paper.reconstruct({target.id: "Robust evaluation is essential."})
    assert "Robust evaluation is essential." in out
    assert "Detectors must be evaluated" not in out
    before, after = paper.source[: target.start], paper.source[target.end :]
    assert out == before + "Robust evaluation is essential." + after


def test_reconstruct_refuses_non_prose(paper):
    eq = next(n for n in paper.nodes if n.type == NodeType.EQUATION)
    with pytest.raises(ReconstructionError):
        paper.reconstruct({eq.id: "x"})


def test_latex_fragment_without_document_env():
    doc = parse_latex("First para with $x$.\n\nSecond para~\\cite{a}.\n")
    assert [n.text for n in doc.prose()] == ["First para with $x$.", "Second para~\\cite{a}."]
    assert doc.reconstruct({}) == doc.source


def test_latex_comment_then_blank_line_breaks_paragraph():
    doc = parse_latex("Alpha beta. % note\n\nGamma delta.\n")
    assert [n.text for n in doc.prose()] == ["Alpha beta. % note", "Gamma delta."]


# --------------------------------------------------------------- Markdown
MD = """---
title: Test
---
# Introduction

LLMs are widely used [1]. They write fluent text.
Second line of same paragraph.

```python
x = {1: 2}
```

$$
E = mc^2
$$

- first item
- second item
  continued

| a | b |
|---|---|

## References

[1] Smith, J. (2020). A paper. Journal.
"""


def test_markdown_structure():
    doc = parse_markdown(MD)
    assert doc.reconstruct({}) == MD
    types = [n.type for n in doc.nodes]
    assert types[0] == NodeType.METADATA
    assert NodeType.CODE in types and NodeType.EQUATION in types and NodeType.TABLE in types
    prose = [n.text for n in doc.prose()]
    assert prose[0].endswith("Second line of same paragraph.")
    assert prose[1:] == ["first item", "second item\n  continued"]
    assert doc.nodes[-1].type == NodeType.BIBLIOGRAPHY


def test_plain_text_heuristic_headings():
    txt = "1. Introduction\n\nWe propose a method.\n\nREFERENCES\n\nSmith J. 2020. Title.\n"
    doc = parse_text(txt)
    assert [n.type for n in doc.nodes] == [NodeType.HEADING, NodeType.PARAGRAPH, NodeType.HEADING,
                                          NodeType.BIBLIOGRAPHY]
    assert doc.nodes[1].section == "Introduction"


# ------------------------------------------------------------------- DOCX
def test_docx_protected_runs(tmp_path):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_heading("Methods", level=1)
    p = d.add_paragraph("Cultures of ")
    p.add_run("E. coli").italic = True
    p.add_run(" were grown at 37 °C; CO")
    sub = p.add_run("2")
    sub.font.subscript = True
    p.add_run(" was 5%.")
    t = d.add_table(rows=1, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "a", "b"
    d.add_heading("References", level=1)
    d.add_paragraph("Smith J (2020) A paper.")
    path = tmp_path / "t.docx"
    d.save(path)

    doc = load(path)
    types = [n.type for n in doc.nodes]
    assert types == [NodeType.HEADING, NodeType.PARAGRAPH, NodeType.TABLE, NodeType.HEADING,
                     NodeType.BIBLIOGRAPHY]
    para = doc.nodes[1]
    assert para.text == "Cultures of E. coli were grown at 37 °C; CO2 was 5%."
    spans = {(para.text[s["start"]:s["end"]], s["reason"]) for s in para.meta["protected"]}
    assert spans == {("E. coli", "italic"), ("2", "subscript")}
    assert para.section == "Methods"


# -------------------------------------------------------------------- TEI
TEI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
 <teiHeader><fileDesc><titleStmt><title level="a" type="main">A Study</title></titleStmt></fileDesc>
  <profileDesc><abstract><div><p>We study things.</p></div></abstract></profileDesc></teiHeader>
 <text><body>
  <div><head n="1">Introduction</head>
   <p>Prior work <ref type="bibr" target="#b0">(Smith et al., 2020)</ref> shows gains (<ref type="figure" target="#fig_0">Fig. 1</ref>).</p>
   <formula xml:id="formula_0">E = mc^2</formula>
   <figure xml:id="fig_0"><head>Figure 1</head><figDesc>Pipeline overview.</figDesc></figure>
  </div>
 </body>
 <back><div type="references"><listBibl>
  <biblStruct xml:id="b0"><analytic><title>Old paper</title></analytic></biblStruct>
 </listBibl></div></back></text>
</TEI>"""


def test_tei_parsing_with_protected_refs():
    from scholarlens.ingest.tei import parse_tei

    doc = parse_tei(TEI_XML)
    types = [n.type for n in doc.nodes]
    assert types == [NodeType.TITLE, NodeType.PARAGRAPH, NodeType.HEADING, NodeType.PARAGRAPH,
                     NodeType.EQUATION, NodeType.FIGURE, NodeType.CAPTION, NodeType.BIBLIOGRAPHY]
    p = doc.nodes[3]
    got = [(p.text[s["start"]:s["end"]], s["reason"]) for s in p.meta["protected"]]
    assert got == [("(Smith et al., 2020)", "citation"), ("Fig. 1", "figure-ref")]
    assert "Pipeline overview." in doc.render_markdown()


def test_all_text_io_is_explicitly_utf8():
    """Windows defaults to cp1252: every text read/write must name UTF-8 (placeholders, ≤, —)."""
    import re

    src = Path(__file__).parents[1] / "src" / "scholarlens"
    bad = []
    for p in src.rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        for m in re.finditer(r"\.(read_text|write_text)\(", text):
            call = text[m.end():m.end() + 400]
            depth, i = 1, 0
            while depth and i < len(call):
                depth += {"(": 1, ")": -1}.get(call[i], 0)
                i += 1
            if "encoding=" not in call[:i]:
                bad.append(f"{p.name}: {text[m.start():m.end() + i][:80]}")
        bad += [f"{p.name}: open() without encoding" for line in text.splitlines()
                if re.search(r"\.open\(\"[aw]\"\)", line)]
    assert not bad, bad
