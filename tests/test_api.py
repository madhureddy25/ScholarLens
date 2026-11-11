import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from scholarlens.api.app import Settings, create_app  # noqa: E402
from scholarlens.edit.prompts import extract_paragraph  # noqa: E402
from scholarlens.llm import register_provider  # noqa: E402
from scholarlens.llm.testing import ScriptedModel  # noqa: E402

LATEX = r"""\section{Intro}
Detecting machine-generated text matters for peer review \citep{liang2024}, as shown in Fig.~\ref{fig:a}.

\begin{equation} x = 1 \end{equation}
"""


@pytest.fixture
def client():
    register_provider("apitest", lambda model, **kw: ScriptedModel(
        lambda u: extract_paragraph(u).replace("matters for", "is important for"), model=model or "m"))
    app = create_app(Settings(allowed_models=["apitest:m"], similarity_model=None, max_chars=5000))
    return TestClient(app)


def test_health_and_models(client):
    assert client.get("/health").json()["detect_enabled"] is False
    assert client.get("/v1/models").json()["allowed"] == ["apitest:m"]


def test_parse_and_mask(client):
    nodes = client.post("/v1/parse", json={"text": LATEX, "format": "latex"}).json()["nodes"]
    assert [n["type"] for n in nodes] == ["heading", "paragraph", "equation"]
    units = client.post("/v1/mask", json={"text": LATEX, "format": "latex"}).json()["units"]
    assert "⟦CITE_1⟧" in units[0]["masked"] and units[0]["symbols"]["⟦CITE_1⟧"] == "\\citep{liang2024}"


def test_edit_document(client):
    r = client.post("/v1/edit", json={"text": LATEX, "format": "latex", "model": "apitest:m", "min_similarity": None})
    body = r.json()
    assert r.status_code == 200 and body["summary"]["accepted"] == 1
    assert body["text"] == LATEX.replace("matters for", "is important for")
    assert "<ins>" in body["nodes"][0]["diff_html"] and body["disclosure"]["statement_draft"]


def test_model_allowlist_and_limits(client):
    r = client.post("/v1/edit", json={"text": "x", "model": "openai-compatible:evil", "format": "text"})
    assert r.status_code == 403
    assert client.post("/v1/parse", json={"text": "a" * 6000}).status_code == 413
    assert client.post("/v1/parse", json={"text": "a", "format": "docx"}).status_code == 422
    assert client.post("/v1/detect", json={"text": "hello"}).status_code == 501
