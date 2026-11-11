"""In-process Hugging Face model (any chat/instruct checkpoint with a chat template).

Runs fully offline once weights are cached — no API key, no data leaves the
machine. Small instruct models (0.5–3B) run on CPU; use a GPU or a local server
(vLLM/Ollama via ``openai_compat``) for 7B+.
"""

from __future__ import annotations

from scholarlens.llm.base import ChatModel, Generation, ModelConfigError


class HFLocalModel(ChatModel):
    provider = "hf"

    def __init__(self, model: str, *, device: str | None = None, temperature: float | None = None,
                 top_p: float | None = None, max_tokens: int = 1024, dtype: str = "auto") -> None:
        super().__init__(model)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            import scholarlens._hub  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ModelConfigError("pip install 'scholarlens[ml]'") from e
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature, self.top_p, self.max_tokens = temperature, top_p, max_tokens
        self.tok = AutoTokenizer.from_pretrained(model)
        self.lm = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(self.device).eval()
        if self.tok.chat_template is None:
            raise ModelConfigError(f"{model} has no chat template; use an instruct/chat checkpoint")

    def describe(self) -> dict:
        return {"provider": self.provider, "model": self.model, "device": self.device,
                "temperature": self.temperature, "top_p": self.top_p}

    def generate(self, system: str, messages: list[dict[str, str]], *,
                 max_tokens: int | None = None) -> Generation:
        chat = [{"role": "system", "content": system}, *messages]
        ids = self.tok.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True)
        ids = {k: v.to(self.device) for k, v in ids.items()}
        sample = self.temperature is not None and self.temperature > 0
        gen_kwargs = {"max_new_tokens": max_tokens or self.max_tokens, "do_sample": sample,
                      "pad_token_id": self.tok.pad_token_id or self.tok.eos_token_id}
        if sample:
            gen_kwargs["temperature"] = self.temperature
            if self.top_p is not None:
                gen_kwargs["top_p"] = self.top_p
        with self.torch.no_grad():
            out = self.lm.generate(**ids, **gen_kwargs)
        new = out[0, ids["input_ids"].shape[1]:]
        text = self.tok.decode(new, skip_special_tokens=True)
        hit_limit = len(new) >= gen_kwargs["max_new_tokens"]
        return Generation(text=text.strip(), provider=self.provider, model=self.model,
                          stop_reason="max_tokens" if hit_limit else "end_turn",
                          usage={"input_tokens": int(ids["input_ids"].shape[1]), "output_tokens": len(new)})
