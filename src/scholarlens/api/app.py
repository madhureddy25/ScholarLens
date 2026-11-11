"""HTTP API (FastAPI).

    uvicorn scholarlens.api.app:app --port 8000

Configuration (environment):
    SCHOLARLENS_ALLOWED_MODELS  comma-separated model specs or profile names clients may use
                                (default: profiles in scholarlens.toml). Clients cannot pass base URLs
                                or credentials: model endpoints and keys are server-side only.
    SCHOLARLENS_LENS_MODEL      directory from `scholarlens lens train` (enables /v1/detect)
    SCHOLARLENS_SIMILARITY      sentence-transformers model for Quill's similarity gate ("" disables)
    SCHOLARLENS_MAX_CHARS       request size limit (default 200000)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from scholarlens import __version__
from scholarlens.analysis import prose_units, segments
from scholarlens.edit import EditConfig, Quill, Status
from scholarlens.edit.diff import diff, render_html
from scholarlens.edit.disclosure import build_manifest
from scholarlens.ingest import parse_string
from scholarlens.llm import ModelConfigError, available_providers, load_model, load_profiles
from scholarlens.masking import Masker


@dataclass
class Settings:
    allowed_models: list[str] = field(default_factory=list)
    lens_model: str | None = None
    similarity_model: str | None = "sentence-transformers/all-MiniLM-L6-v2"
    max_chars: int = 200_000

    @classmethod
    def from_env(cls) -> Settings:
        allowed = [m.strip() for m in os.environ.get("SCHOLARLENS_ALLOWED_MODELS", "").split(",") if m.strip()]
        if not allowed:
            allowed = list(load_profiles()[0])
        sim = os.environ.get("SCHOLARLENS_SIMILARITY", "sentence-transformers/all-MiniLM-L6-v2")
        return cls(allowed_models=allowed, lens_model=os.environ.get("SCHOLARLENS_LENS_MODEL") or None,
                   similarity_model=sim or None, max_chars=int(os.environ.get("SCHOLARLENS_MAX_CHARS", 200_000)))


class TextIn(BaseModel):
    text: str
    format: str = Field("text", pattern="^(text|markdown|latex)$")


class EditIn(TextIn):
    model: str
    mode: str = Field("clarity", pattern="^(proofread|clarity|concise)$")
    min_similarity: float | None = Field(0.94, ge=0, le=1)
    max_attempts: int = Field(3, ge=1, le=5)
    sections: list[str] | None = None
    glossary: list[str] = Field(default_factory=list, max_length=500)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="ScholarLens", version=__version__,
                  description="Citation-safe editing (Quill) and robust AI-text detection (Lens) for academic text.")

    def get_settings() -> Settings:
        return settings

    def check_size(body: TextIn, s: Settings = Depends(get_settings)) -> None:
        if len(body.text) > s.max_chars:
            raise HTTPException(413, f"text longer than {s.max_chars} characters")

    @lru_cache(maxsize=8)
    def model_for(spec: str):
        return load_model(spec)

    @lru_cache(maxsize=1)
    def similarity():
        if not settings.similarity_model:
            return None
        from scholarlens.edit import sentence_transformer_similarity

        return sentence_transformer_similarity(settings.similarity_model)

    @lru_cache(maxsize=1)
    def lens():
        import json
        from pathlib import Path

        from scholarlens.detect import load_featurizer_from_signature
        from scholarlens.detect.fusion import FusionDetector

        path = Path(settings.lens_model) / "fusion"
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        return FusionDetector.load(path, load_featurizer_from_signature(meta["featurizer"]))

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__, "detect_enabled": bool(settings.lens_model)}

    @app.get("/v1/models")
    def models() -> dict:
        return {"allowed": settings.allowed_models, "providers": available_providers()}

    @app.post("/v1/parse", dependencies=[Depends(check_size)])
    def parse(body: TextIn) -> dict:
        doc = parse_string(body.text, body.format)
        return {"format": doc.source_format, "stats": doc.stats(),
                "nodes": [{"id": n.id, "type": n.type.value, "section": n.section, "span": [n.start, n.end],
                           "prose": n.is_prose, "text": n.text} for n in doc.nodes]}

    @app.post("/v1/mask", dependencies=[Depends(check_size)])
    def mask(body: TextIn) -> dict:
        doc = parse_string(body.text, body.format)
        return {"units": [{"node_id": u.node_id, "section": u.section, "masked": u.mask.masked,
                           "symbols": u.mask.symbols, "counts": u.mask.counts()} for u in prose_units(doc)]}

    @app.post("/v1/edit", dependencies=[Depends(check_size)])
    def edit(body: EditIn) -> dict:
        if body.model not in settings.allowed_models:
            raise HTTPException(403, f"model not allowed; choose one of {settings.allowed_models}")
        try:
            chat = model_for(body.model)
        except ModelConfigError as e:
            raise HTTPException(503, f"model unavailable: {e}") from e
        doc = parse_string(body.text, body.format)
        cfg = EditConfig(mode=body.mode, min_similarity=body.min_similarity or None, max_attempts=body.max_attempts,
                         sections=tuple(body.sections) if body.sections else None)
        from scholarlens.masking import MaskConfig

        quill = Quill(chat, masker=Masker(MaskConfig(glossary=tuple(body.glossary))),
                      similarity=similarity() if cfg.min_similarity else None, config=cfg)
        try:
            session = quill.edit_document(doc)
        except ModelConfigError as e:
            raise HTTPException(503, f"model unavailable: {e}") from e
        nodes = []
        for e in session.edits:
            item = {"node_id": e.node_id, "section": e.section, "status": e.status.value, "reason": e.reason,
                    "similarity": e.similarity, "attempts": len(e.attempts), "warnings": e.warnings}
            if e.status == Status.ACCEPTED and e.mask is not None and e.edited_masked is not None:
                item["edited"] = e.edited
                item["diff_html"] = render_html(diff(e.mask.masked, e.edited_masked), e.mask)
            nodes.append(item)
        manifest = build_manifest(session)
        return {"text": session.output_text(), "summary": manifest["summary"], "nodes": nodes,
                "disclosure": manifest}

    @app.post("/v1/detect", dependencies=[Depends(check_size)])
    def detect(body: TextIn) -> dict:
        if not settings.lens_model:
            raise HTTPException(501, "detection is not configured on this server (SCHOLARLENS_LENS_MODEL)")
        segs = segments(prose_units(parse_string(body.text, body.format)))
        if not segs:
            return {"segments": [], "document": None}
        probs = lens().score_texts([s.text for s in segs], ids=[f"api:{hash(s.text)}" for s in segs])
        return {
            "segments": [{"index": s.index, "node_ids": s.node_ids, "words": s.n_words, "p_ai": round(float(p), 4)}
                         for s, p in zip(segs, probs, strict=True)],
            "document": {"mean_p_ai": round(float(probs.mean()), 4), "max_p_ai": round(float(probs.max()), 4)},
            "notice": "Calibrated on the Bench data the model was trained on. Evidence, not a verdict: never use it "
                      "alone to make decisions about an author.",
        }

    return app


def __getattr__(name: str):  # lazily build the default app for `uvicorn scholarlens.api.app:app`
    if name == "app":
        return create_app()
    raise AttributeError(name)
