"""Transformations applied to benchmark texts.

Two purposes, kept separate by ``apply_to``:

* **Attacks on AI text** (the detector's threat model): DIPPER paraphrasing at
  several lexical/order diversities, and generic LLM paraphrasing. These are the
  attacks published in the detection literature (Krishna et al., 2023) and are
  used only to *measure* and *improve* detector robustness.
* **Legitimate edits of human text** (the false-positive study): Quill in each
  mode and a rule-based grammar checker (LanguageTool). A fair detector should
  not flag a human author for using these.
"""

from __future__ import annotations

import json
from typing import Protocol
from urllib import parse, request

from scholarlens.bench.schema import Record
from scholarlens.core.text import split_sentences
from scholarlens.llm.base import ChatModel, ModelError


class Transform(Protocol):
    name: str
    model_name: str | None
    apply_to: str  # "human" | "ai" | "both"

    def apply(self, rec: Record) -> Record | None: ...


class DipperParaphraser:
    """DIPPER (Krishna et al., 2023): T5-XXL paraphraser with lexical (L) and order (O)
    diversity controls in {0, 20, …, 100}. Needs a GPU with ~45 GB (fp32) or ~23 GB (bf16)."""

    def __init__(self, lex: int = 60, order: int = 60, *, model: str = "kalpeshk2011/dipper-paraphraser-xxl",
                 tokenizer: str = "google/t5-v1_1-xxl", device: str = "cuda", sent_interval: int = 3,
                 top_p: float = 0.75, apply_to: str = "ai") -> None:
        if lex not in range(0, 101, 20) or order not in range(0, 101, 20):
            raise ValueError("lex and order must be multiples of 20 in [0, 100]")
        self.lex, self.order, self.sent_interval, self.top_p = lex, order, sent_interval, top_p
        self.name = f"dipper_L{lex}_O{order}"
        self.model_name, self.apply_to = model, apply_to
        self._model_id, self._tok_id, self.device = model, tokenizer, device
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import T5ForConditionalGeneration, T5Tokenizer


            self.torch = torch
            self.tok = T5Tokenizer.from_pretrained(self._tok_id)
            dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
            self._model = T5ForConditionalGeneration.from_pretrained(self._model_id, dtype=dtype).to(self.device).eval()

    def paraphrase(self, text: str) -> str:
        self._load()
        # DIPPER's control codes are inverted: code = 100 - diversity
        codes = f"lexical = {100 - self.lex}, order = {100 - self.order}"
        sentences = split_sentences(" ".join(text.split()))
        prefix, output = "", []
        for i in range(0, len(sentences), self.sent_interval):
            window = " ".join(sentences[i:i + self.sent_interval])
            inp = codes + (f" {prefix}" if prefix else "") + f" <sent> {window} </sent>"
            enc = {k: v.to(self.device) for k, v in self.tok([inp], return_tensors="pt").items()}
            with self.torch.inference_mode():
                out = self._model.generate(**enc, do_sample=True, top_p=self.top_p, top_k=None, max_length=512)
            piece = self.tok.batch_decode(out, skip_special_tokens=True)[0].strip()
            prefix = f"{prefix} {piece}".strip()
            output.append(piece)
        return " ".join(output)

    def apply(self, rec: Record) -> Record | None:
        text = self.paraphrase(rec.text)
        return rec.derive(text, transform=self.name, transform_model=self.model_name) if text else None


class LLMParaphraser:
    """Generic paraphrase with any chat model (plain meaning-preserving paraphrase)."""

    SYSTEM = "You rewrite passages in different words while keeping their meaning."
    PROMPT = ("Paraphrase the following passage. Keep its meaning, technical terms and numbers. "
              "Return only the paraphrase.\n\n{text}")

    def __init__(self, model: ChatModel, *, apply_to: str = "ai", name: str = "llm_paraphrase") -> None:
        self.model, self.apply_to, self.name = model, apply_to, name
        self.model_name = model.name

    def apply(self, rec: Record) -> Record | None:
        try:
            gen = self.model.generate(self.SYSTEM, [{"role": "user", "content": self.PROMPT.format(text=rec.text)}])
        except ModelError:
            return None
        from scholarlens.bench.generate import clean_generation

        text = clean_generation(gen.text)
        if gen.refused or not text:
            return None
        return rec.derive(text, transform=self.name, transform_model=f"{gen.provider}:{gen.model}")


class QuillTransform:
    """A human abstract after legitimate language editing with Quill."""

    def __init__(self, model: ChatModel, *, mode: str = "proofread", similarity=None, apply_to: str = "human") -> None:
        from scholarlens.edit import EditConfig, Quill

        self.quill = Quill(model, similarity=similarity, config=EditConfig(mode=mode))
        self.name, self.apply_to, self.model_name = f"quill_{mode}", apply_to, model.name

    def apply(self, rec: Record) -> Record | None:
        e = self.quill.edit_text(rec.text, "text", node_id=rec.id)
        if e.status.value in ("error", "refused"):
            return None
        return rec.derive(e.final_text, transform=self.name, transform_model=self.model_name,
                          quill_status=e.status.value, similarity=e.similarity, change=e.stats or None)


class LanguageToolEditor:
    """Rule-based grammar/spelling correction via a LanguageTool server
    (``docker run -p 8010:8010 erikvl87/languagetool``). Applies the first
    suggestion of every match, like an author clicking "accept all"."""

    def __init__(self, url: str = "http://localhost:8010", *, language: str = "en-US", apply_to: str = "human",
                 timeout: float = 30.0) -> None:
        self.url, self.language, self.apply_to, self.timeout = url.rstrip("/"), language, apply_to, timeout
        self.name, self.model_name = "languagetool", f"languagetool@{self.url}"

    def correct(self, text: str) -> str:
        data = parse.urlencode({"text": text, "language": self.language}).encode()
        with request.urlopen(request.Request(f"{self.url}/v2/check", data=data), timeout=self.timeout) as r:  # noqa: S310
            matches = json.loads(r.read())["matches"]
        out, last = text, len(text) + 1
        for m in sorted(matches, key=lambda m: m["offset"], reverse=True):
            if not m.get("replacements") or m["offset"] + m["length"] > last:
                continue
            out = out[: m["offset"]] + m["replacements"][0]["value"] + out[m["offset"] + m["length"]:]
            last = m["offset"]
        return out

    def apply(self, rec: Record) -> Record | None:
        try:
            text = self.correct(rec.text)
        except OSError:
            return None
        return rec.derive(text, transform=self.name, transform_model=self.model_name)


def make_transform(spec: dict, load_model) -> Transform:
    spec = dict(spec)
    kind = spec.pop("type")
    spec.pop("limit", None)  # builder-level setting
    if kind == "dipper":
        return DipperParaphraser(**spec)
    if kind == "llm_paraphrase":
        return LLMParaphraser(load_model(spec.pop("model")), **spec)
    if kind == "quill":
        similarity = None
        sim_model = spec.pop("similarity_model", None)
        if sim_model:
            from scholarlens.edit import sentence_transformer_similarity

            similarity = sentence_transformer_similarity(sim_model)
        return QuillTransform(load_model(spec.pop("model")), similarity=similarity, **spec)
    if kind == "languagetool":
        return LanguageToolEditor(**spec)
    raise ValueError(f"unknown transform type {kind!r}")
