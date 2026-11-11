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


# ------------------------------------------------------------ Lens learning
import io  # noqa: E402
import random  # noqa: E402
import tarfile  # noqa: E402

from scholarlens.bench.schema import Record, write_jsonl  # noqa: E402
from scholarlens.bench.splits import assign_splits  # noqa: E402
from scholarlens.learn.lens import contribution_records, improve_lens, lens_gate  # noqa: E402
from scholarlens.learn.release import install, latest_release, next_tag, package, publish, sha256  # noqa: E402


def _bench(tmp_path, n=90):
    rng = random.Random(0)
    words = ["we", "the", "data", "results", "samples", "noise", "trend", "model", "field", "error", "rate", "site"]
    recs = []
    for i in range(n):
        h = " ".join(" ".join(rng.choice(words) for _ in range(rng.choice([6, 11, 19, 28]))).capitalize() + "."
                     for _ in range(rng.randint(4, 7)))
        a = " ".join(f"{o}, this work delves into a comprehensive and pivotal framework for topic {i % 7}."
                     for o in ("Furthermore", "Moreover", "Additionally", "Notably", "Overall"))
        recs += [Record(id=f"h{i}", paper_id=f"p{i}", text=h, label="human", source="syn"),
                 Record(id=f"a{i}", paper_id=f"p{i}", text=a, label="ai", source="syn", generator="g")]
    path = tmp_path / "bench.jsonl"
    write_jsonl(assign_splits(recs), path)
    return path


def test_lens_gate():
    cur = {"attacks": {"ai": {"auroc": 0.95}, "ai+p": {"auroc": 0.80}}, "human_fpr": {"human": {"fpr": 0.01}}}
    better = {"attacks": {"ai": {"auroc": 0.95}, "ai+p": {"auroc": 0.86}}, "human_fpr": {"human": {"fpr": 0.01}}}
    assert lens_gate(better, cur)[0]
    unfair = {**better, "human_fpr": {"human": {"fpr": 0.05}}}
    ok, why = lens_gate(unfair, cur)
    assert not ok and "false-positive" in why[0]
    trade = {"attacks": {"ai": {"auroc": 0.90}, "ai+p": {"auroc": 0.99}}, "human_fpr": {"human": {"fpr": 0.0}}}
    assert not lens_gate(trade, cur)[0]  # may not sacrifice one attack family for another


def test_contribution_records_are_train_only_human():
    recs = contribution_records([{**ENTRY, "id": "abc"}, {**ENTRY, "id": "def", "decision": "reject"}])
    assert len(recs) == 2 and {r.split for r in recs} == {"train"} and {r.label for r in recs} == {"human"}
    assert "⟦" not in recs[0].text and "[1]" in recs[0].text


def test_improve_lens_rounds(tmp_path):
    from scholarlens.detect.featurize import Featurizer

    bench = _bench(tmp_path)
    write_contribution(prepare([ENTRY])[0], tmp_path / "repo")
    r1 = improve_lens(bench=bench, repo_dir=tmp_path / "repo", out_dir=tmp_path / "v1", featurizer=Featurizer(),
                      log=lambda *_: None)
    assert r1["accepted"] and r1["current"] is None and r1["n_contributions"] == 1
    r2 = improve_lens(bench=bench, repo_dir=tmp_path / "repo", out_dir=tmp_path / "v2", featurizer=Featurizer(),
                      current_dir=tmp_path / "v1", log=lambda *_: None)
    assert r2["current"] is not None and isinstance(r2["accepted"], bool)
    assert (tmp_path / "v2" / "fusion" / "model.joblib").is_file()


# ---------------------------------------------------------------- releases
def test_next_tag():
    assert next_tag("lens", []) == "lens-v1"
    assert next_tag("lens", ["lens-v1", "quill-v4", "lens-v3", "lens-vx"]) == "lens-v4"


def test_package_and_publish_refuses_unaccepted(tmp_path):
    d = tmp_path / "cand"
    (d / "fusion").mkdir(parents=True)
    (d / "fusion" / "model.joblib").write_bytes(b"m")
    (d / "fusion" / "meta.json").write_text("{}")
    (d / "features.jsonl").write_text("big cache")
    (d / "learn_report.json").write_text(json.dumps({"accepted": False}))
    archive = package("lens", d, tmp_path, "lens-v1")
    assert sorted(tarfile.open(archive).getnames()) == ["fusion/meta.json", "fusion/model.joblib", "learn_report.json"]
    calls = []

    def runner(cmd, **kw):
        calls.append(cmd)
        return type("R", (), {"stdout": "lens-v1\nlens-v2\n"})()

    with pytest.raises(ValueError, match="not mark this candidate as accepted"):
        publish("lens", d, runner=runner)
    (d / "learn_report.json").write_text(json.dumps({"accepted": True}))
    assert publish("lens", d, runner=runner, repo="o/r") == "lens-v3"
    create = calls[-1]
    assert create[:4] == ["gh", "release", "create", "lens-v3"] and "sha256:" in create[create.index("--notes") + 1]


def _release(tmp_path, files: dict[str, bytes], tag="lens-v2", corrupt=False):
    archive = tmp_path / f"{tag}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    digest = "0" * 64 if corrupt else sha256(archive)
    rel = {"tag_name": tag, "body": f"notes\n\nsha256: {digest}\n",
           "assets": [{"name": archive.name, "browser_download_url": "https://example/x"}]}
    return rel, lambda url, dest: dest.write_bytes(archive.read_bytes())


def test_install_verifies_checksum_and_paths(tmp_path):
    from scholarlens.learn.release import installed

    rel, dl = _release(tmp_path, {"fusion/meta.json": b"{}", "fusion/model.joblib": b"x"})
    dest = install("lens", rel, download=dl)
    assert (dest / "fusion" / "meta.json").is_file() and installed("lens") == dest
    rel, dl = _release(tmp_path, {"fusion/meta.json": b"{}"}, tag="lens-v3", corrupt=True)
    with pytest.raises(ValueError, match="checksum"):
        install("lens", rel, download=dl)
    rel, dl = _release(tmp_path, {"../../evil.py": b"print('x')"}, tag="lens-v4")
    with pytest.raises(ValueError, match="unsafe"):
        install("lens", rel, download=dl)


def test_latest_release_picks_newest_matching_tag():
    rels = [{"tag_name": "quill-v2"}, {"tag_name": "lens-v5", "draft": True}, {"tag_name": "lens-v4"},
            {"tag_name": "lens-v3"}]
    assert latest_release("lens", fetch=lambda url: rels)["tag_name"] == "lens-v4"
    assert latest_release("bench", fetch=lambda url: rels) is None
