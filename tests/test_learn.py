import json
from pathlib import Path

import pytest

from scholarlens.edit import EditConfig, Quill, write_outputs
from scholarlens.edit.prompts import extract_paragraph
from scholarlens.ingest import load
from scholarlens.learn import feedback
from scholarlens.learn.contribute import load_contributions, prepare, problems, sanitize, write_contribution
from scholarlens.learn.review import review_session
from scholarlens.llm.testing import ScriptedModel

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("SCHOLARLENS_HOME", str(tmp_path / "home"))


def _editor(u):
    return extract_paragraph(u).replace("matters for", "is important for").replace("must be evaluated", "should be evaluated")


def _session(tmp_path):
    doc = load(FIX / "sample_paper.tex")
    s = Quill(ScriptedModel(_editor), config=EditConfig(mode="clarity")).edit_document(doc)
    files = write_outputs(s, tmp_path / "out", input_path=FIX / "sample_paper.tex")
    return files["session"]


def test_review_applies_only_accepted_edits_and_records_feedback(tmp_path):
    session_file = _session(tmp_path)
    data = json.loads(session_file.read_text())
    first, second = (e["node_id"] for e in data["edits"])
    res = review_session(session_file, lambda e: "accept" if e.node_id == first else "reject")
    final = res.files["final"].read_text()
    src = (FIX / "sample_paper.tex").read_text()
    assert "is important for" in final and "must be evaluated" in final  # second edit rejected
    assert final == src.replace("matters for", "is important for")
    manifest = json.loads(res.files["disclosure"].read_text())
    assert manifest["summary"]["accepted"] == 1 and manifest["author_review"]["rejected"] == [second]
    stored = feedback.load_all()
    assert [e["decision"] for e in stored] == ["accept", "reject"]
    assert all("⟦CITE_1⟧" in e["masked_input"] for e in stored[:1])  # masked, not raw citations
    assert "\\citep" not in json.dumps(stored)


def test_review_refuses_if_source_changed(tmp_path):
    src = tmp_path / "p.tex"
    src.write_text((FIX / "sample_paper.tex").read_text())
    s = Quill(ScriptedModel(_editor)).edit_document(load(src))
    files = write_outputs(s, tmp_path / "out", input_path=src)
    src.write_text(src.read_text().replace("Detecting machine-generated", "Finding machine-generated"))
    with pytest.raises(ValueError, match="changed since"):
        review_session(files["session"], lambda e: "accept")


def test_review_without_feedback(tmp_path):
    review_session(_session(tmp_path), lambda e: "accept", record_feedback=False)
    assert feedback.load_all() == []


ENTRY = {"id": "x", "created_at": "2026-11-11T10:00:00+00:00", "tool_version": "0.1.0", "mode": "clarity",
         "format": "latex", "model": {"provider": "openai-compatible", "model": "m", "base_url": "http://internal:8000"},
         "section": "Methods", "masked_input": "We measured ⟦NUM_1⟧ samples as in ⟦CITE_1⟧ with care.",
         "masked_output": "We measured ⟦NUM_1⟧ samples, as in ⟦CITE_1⟧, carefully.", "decision": "accept"}


def test_sanitize_drops_endpoints_and_fine_timestamps():
    e = sanitize(ENTRY)
    assert "base_url" not in json.dumps(e) and e["date"] == "2026-11-11" and e["section"] == "methods"
    assert sanitize({**ENTRY, "section": "Our secret project X"})["section"] is None


@pytest.mark.parametrize("text,reason", [
    ("Contact me at jane.doe@uni.edu about ⟦NUM_1⟧ results today.", "email"),
    ("See https://internal.example/report for ⟦NUM_1⟧ details please.", "url"),
    ("Call +44 20 7946 0958 regarding the ⟦NUM_1⟧ samples now.", "phone"),
    ("Author ORCID 0000-0002-1825-0097 measured ⟦NUM_1⟧ samples.", "orcid"),
])
def test_pii_screen(text, reason):
    e = sanitize({**ENTRY, "masked_input": text, "masked_output": text})
    assert any(reason in p for p in problems(e))


def test_accepted_entries_must_preserve_placeholders():
    e = sanitize({**ENTRY, "masked_output": "We measured many samples carefully as before."})
    assert "accepted edit does not preserve placeholders" in problems(e)
    assert not problems(sanitize({**ENTRY, "decision": "reject", "masked_output": "We measured many samples."}))


def test_contribution_roundtrip_dedupes(tmp_path):
    ok, bad = prepare([ENTRY, dict(ENTRY, id="y"), {**ENTRY, "id": "z", "mode": "humanize"}])
    assert len(ok) == 1 and len(bad) == 2  # duplicate content + invalid mode
    path = write_contribution(ok, tmp_path)
    assert path.parent.name == "contributions"
    loaded, notes = load_contributions(tmp_path)
    assert len(loaded) == 1 and not notes
    (tmp_path / "contributions" / "evil.jsonl").write_text(json.dumps({**ENTRY, "masked_input": "x"}) + "\n")
    loaded, notes = load_contributions(tmp_path)
    assert len(loaded) == 1 and any("length" in n for n in notes)
