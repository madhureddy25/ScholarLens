import json
from pathlib import Path

import pytest

from scholarlens.edit import EditConfig, Quill, Status, write_outputs
from scholarlens.edit.diff import diff, render_html, render_latex_tracked
from scholarlens.edit.prompts import extract_paragraph, system_prompt
from scholarlens.ingest import load, parse_latex
from scholarlens.llm.base import ModelConfigError, ModelError
from scholarlens.llm.testing import EchoModel, ScriptedModel
from scholarlens.masking import Masker

FIX = Path(__file__).parent / "fixtures"
PARA = "Prior work (Smith et al., 2020) showed that BRCA1 expression rises by 12% in tumours [4]."


def masked(text=PARA):
    return Masker().mask(text).masked


def quill(responses, **cfg):
    return Quill(ScriptedModel(responses), config=EditConfig(**cfg))


def test_accepts_valid_edit_and_restores_protected_text():
    edit = masked().replace("showed that", "demonstrated that")
    e = quill([edit]).edit_text(PARA)
    assert e.status == Status.ACCEPTED
    assert e.edited == PARA.replace("showed that", "demonstrated that")
    assert e.stats["words_removed"] == 1 and e.stats["words_added"] == 1


def test_retry_with_feedback_when_placeholder_dropped():
    bad = masked().replace("⟦CITE_1⟧ ", "")
    good = masked().replace("showed", "found")
    model = ScriptedModel([bad, good])
    e = Quill(model).edit_text(PARA)
    assert e.status == Status.ACCEPTED and len(e.attempts) == 2
    feedback = model.calls[1]["messages"][-1]["content"]
    assert "⟦CITE_1⟧" in feedback and "missing" in feedback
    assert model.calls[1]["messages"][1]["role"] == "assistant"


def test_damaged_placeholders_are_repaired():
    damaged = masked().replace("⟦CITE_1⟧", "[CITE_1]").replace("showed", "found")
    e = quill([damaged]).edit_text(PARA)
    assert e.status == Status.ACCEPTED and e.attempts[0].repairs
    assert "(Smith et al., 2020)" in e.edited


def test_rejected_after_max_attempts_keeps_original():
    bad = masked().replace("⟦NUM_1⟧", "15%")
    e = quill([bad, bad, bad], max_attempts=3).edit_text(PARA)
    assert e.status == Status.REJECTED and len(e.attempts) == 3
    assert e.final_text == PARA and "⟦NUM_1⟧" in e.reason


def test_similarity_gate():
    edit = masked().replace("showed that", "argued that")
    low = Quill(ScriptedModel([edit, edit]), similarity=lambda a, b: 0.5,
                config=EditConfig(max_attempts=2, min_similarity=0.94)).edit_text(PARA)
    assert low.status == Status.REJECTED and "similarity" in low.reason
    high = Quill(ScriptedModel([edit]), similarity=lambda a, b: 0.99).edit_text(PARA)
    assert high.status == Status.ACCEPTED and high.similarity == 0.99


def test_unchanged_refused_error_and_short():
    assert Quill(EchoModel()).edit_text(PARA).status == Status.UNCHANGED
    assert quill(["<<REFUSE>>"]).edit_text(PARA).status == Status.REFUSED
    assert quill([]).edit_text("Too short [1].").status == Status.SKIPPED

    class Failing(ScriptedModel):
        def __init__(self, exc):
            super().__init__([])
            self.exc = exc

        def generate(self, *a, **k):
            raise self.exc

    assert Quill(Failing(ModelError("timeout"))).edit_text(PARA).status == Status.ERROR
    with pytest.raises(ModelConfigError):
        Quill(Failing(ModelConfigError("no key"))).edit_text(PARA)


def test_wrapper_tags_never_leak():
    leaked = masked().replace("showed", "found") + "\n</paragraph>"
    e = quill([leaked]).edit_text(PARA)
    assert e.status == Status.ACCEPTED and "paragraph>" not in e.edited


@pytest.mark.parametrize("raw,expected", [
    ("<paragraph>\nText.\n</paragraph>", "Text."),
    ("```\nText.\n```", "Text."),
    ("Here is the revised paragraph:\n\nText.", "Text."),
    ("\"Text.\"", "Text."),
    ("Text.\n</paragraph>", "Text."),
])
def test_extract_paragraph(raw, expected):
    assert extract_paragraph(raw) == expected


def test_prompt_modes_and_format():
    assert "LaTeX" in system_prompt("proofread", "latex")
    with pytest.raises(ValueError):
        system_prompt("humanize", "text")


def test_latex_tilde_spacing_preserved():
    src = r"Prose is masked before the scoring step~\cite{bao2024} in our pipeline."
    m = Masker().mask(src, fmt="latex")
    e = quill([m.masked.replace("step⟦", "step ⟦").replace("Prose is", "Text is")]).edit_text(src, "latex")
    assert e.edited == r"Text is masked before the scoring step~\cite{bao2024} in our pipeline."


# ------------------------------------------------------------------- documents
def _rewrite(user_msg):
    p = extract_paragraph(user_msg)
    return p.replace("matters for", "is important for").replace("must be evaluated", "should be evaluated")


def test_document_edit_only_touches_accepted_spans():
    doc = load(FIX / "sample_paper.tex")
    session = Quill(ScriptedModel(_rewrite)).edit_document(doc)
    accepted = [e for e in session.edits if e.status == Status.ACCEPTED]
    assert len(accepted) == 2
    out = session.output_text()
    # everything outside accepted node spans is byte-identical
    spans = sorted((doc.get(e.node_id).start, doc.get(e.node_id).end) for e in accepted)
    cursor_src, pieces = 0, []
    for a, b in spans:
        pieces.append(doc.source[cursor_src:a])
        cursor_src = b
    pieces.append(doc.source[cursor_src:])
    pos = 0
    for piece in pieces:
        idx = out.find(piece, pos)
        assert idx >= 0
        pos = idx + len(piece)
    assert "\\begin{equation}" in out and "stray } braces {" in out


def test_section_and_node_filters():
    doc = load(FIX / "sample_paper.tex")
    s = Quill(ScriptedModel(_rewrite), config=EditConfig(sections=("Conclusion",))).edit_document(doc)
    touched = [e.node_id for e in s.edits if e.status != Status.SKIPPED]
    assert touched == [n.id for n in doc.prose() if n.section == "Conclusion"]


def test_locked_nodes_are_skipped():
    doc = parse_latex("A paragraph with plenty of editable words in it here.\n")
    next(doc.prose()).meta["locked"] = "inline-equation"
    s = Quill(ScriptedModel([])).edit_document(doc)
    assert s.edits[0].status == Status.SKIPPED and "locked" in s.edits[0].reason


# ----------------------------------------------------------------- outputs
def test_tracked_latex_and_html_rendering():
    m = Masker().mask(r"The model clearly works well~\cite{a} on data.", fmt="latex")
    new = m.masked.replace("clearly works well", "performs well")
    ops = diff(m.masked, new)
    tex = render_latex_tracked(ops, m)
    assert "\\DIFdel{clearly works}" in tex and "\\DIFadd{performs}" in tex and "~\\cite{a}" in tex
    page = render_html(ops, m)
    assert "<del>" in page and "<ins>performs</ins>" in page


def test_write_outputs(tmp_path):
    doc = load(FIX / "sample_paper.tex")
    session = Quill(ScriptedModel(_rewrite)).edit_document(doc)
    files = write_outputs(session, tmp_path, input_path=FIX / "sample_paper.tex")
    assert set(files) == {"edited", "tracked", "disclosure", "review", "session"}
    manifest = json.loads(files["disclosure"].read_text())
    assert manifest["summary"]["accepted"] == 2 and manifest["model"]["provider"] == "scripted"
    assert manifest["input"]["sha256"] and manifest["output"]["sha256"]
    assert "ScholarLens Quill" in manifest["statement_draft"]
    assert "\\DIFadd{" in files["tracked"].read_text()
    assert "<ins>" in files["review"].read_text()


def test_markdown_document_roundtrip(tmp_path):
    md = tmp_path / "notes.md"
    md.write_text("# Intro\n\nThe results matters for many downstream applications [1].\n\n$$x=1$$\n")
    doc = load(md)
    s = Quill(ScriptedModel(lambda u: extract_paragraph(u).replace("results matters", "results matter"))
              ).edit_document(doc)
    assert s.output_text() == md.read_text().replace("results matters", "results matter")


@pytest.mark.slow
def test_real_local_model_end_to_end():
    pytest.importorskip("transformers")
    from scholarlens.llm import load_model

    try:
        model = load_model("hf:Qwen/Qwen2.5-0.5B-Instruct", max_tokens=200)
    except OSError:
        pytest.skip("model weights unavailable")
    text = "Our method builds on Fast-DetectGPT [3] and reach an AUROC of 0.91 on 12,000 abstracts."
    e = Quill(model, config=EditConfig(mode="proofread")).edit_text(text)
    assert e.status in (Status.ACCEPTED, Status.UNCHANGED, Status.REJECTED)
    if e.status == Status.ACCEPTED:  # whatever the model did, protected items survived verbatim
        for item in ("Fast-DetectGPT", "[3]", "AUROC", "0.91", "12,000"):
            assert item in e.edited


# ------------------------------------------------------------------- DOCX
def _make_docx(path):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_heading("Methods", level=1)
    p = d.add_paragraph("Cultures of ")
    p.add_run("E. coli").italic = True
    p.add_run(" was grown overnight at 37 °C in ")
    p.add_run("rich medium and the cells was then ")  # second plain run: edits cross a run boundary
    p.add_run("harvested for analysis [3].")
    d.add_paragraph("A second paragraph that the editor leaves completely untouched here.")
    d.save(path)


def _docx_editor(user_msg):
    text = extract_paragraph(user_msg)
    return (text.replace("was grown overnight", "were grown overnight")
                .replace("the cells was then", "the cells were then"))


def test_docx_tracked_and_accepted_outputs(tmp_path):
    from lxml import etree

    from scholarlens.ingest.docx import W

    src = tmp_path / "paper.docx"
    _make_docx(src)
    doc = load(src)
    session = Quill(ScriptedModel(_docx_editor)).edit_document(doc)
    target = next(e for e in session.edits if e.status == Status.ACCEPTED)
    files = write_outputs(session, tmp_path / "out", input_path=src)
    assert files["edited"].suffix == ".docx" and files["tracked"].suffix == ".docx"

    accepted = load(files["edited"])
    tracked = load(files["tracked"])  # parser skips w:del and keeps w:ins → the accepted view
    for d in (accepted, tracked):
        para = d.get(target.node_id)
        assert para.text == target.edited
        assert {(para.text[s["start"]:s["end"]], s["reason"]) for s in para.meta["protected"]} == {("E. coli", "italic")}
        assert d.nodes[-1].text == doc.nodes[-1].text  # untouched paragraph

    import zipfile
    xml = etree.fromstring(zipfile.ZipFile(files["tracked"]).read("word/document.xml"))
    ins, dels = xml.findall(f".//{W}ins"), xml.findall(f".//{W}del")
    assert len(ins) == 2 and len(dels) == 2
    assert all(el.get(f"{W}author") == "ScholarLens Quill" for el in ins + dels)
    deleted = "".join(t.text for t in xml.iter(f"{W}delText"))
    assert "was" in deleted
    manifest = json.loads(files["disclosure"].read_text())
    assert manifest["docx"]["written"] == [target.node_id] and manifest["docx"]["kept_original"] == []


def test_docx_edit_that_moves_protected_item_keeps_original(tmp_path):
    src = tmp_path / "paper.docx"
    _make_docx(src)
    doc = load(src)

    def mover(user_msg):  # moves the italic species name: not representable as safe revisions
        text = extract_paragraph(user_msg)
        return text.replace("Cultures of ⟦PROT_1⟧", "Cultures").replace("the cells was then",
                                                                         "the ⟦PROT_1⟧ cells were then")

    session = Quill(ScriptedModel(mover)).edit_document(doc)
    files = write_outputs(session, tmp_path / "out", input_path=src)
    manifest = json.loads(files["disclosure"].read_text())
    assert manifest["docx"]["written"] == [] and "moves a protected item" in manifest["docx"]["kept_original"][0]
    assert load(files["edited"]).nodes[1].text == doc.nodes[1].text


def test_disclosure_names_fine_tuned_adapter():
    from scholarlens.edit.disclosure import disclosure_statement

    doc = load(FIX / "sample_paper.tex")
    session = Quill(ScriptedModel(_rewrite)).edit_document(doc)
    session.model = {"provider": "hf", "model": "Qwen/Qwen2.5-0.5B-Instruct", "adapter": "/x/models/quill-rl-v1"}
    assert "fine-tuned adapter 'quill-rl-v1'" in disclosure_statement(session)
