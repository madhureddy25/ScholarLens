"""Any OpenAI-compatible chat-completions endpoint.

Covers OpenAI itself and the many servers/providers that speak the same
protocol: Ollama, vLLM, LM Studio, llama.cpp server, OpenRouter, Groq,
Together, DeepSeek, Mistral, Gemini's OpenAI-compatible endpoint, Azure-hosted
gateways, and in-house deployments. Use a preset name as the provider, or
``openai-compatible`` with an explicit ``base_url``.
"""

from __future__ import annotations

import os

from scholarlens.llm.base import ChatModel, Generation, ModelConfigError, ModelError

# provider name -> (base_url, api-key environment variable or None for "no key needed")
PRESETS: dict[str, tuple[str | None, str | None]] = {
    "openai": (None, "OPENAI_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
    "vllm": ("http://localhost:8000/v1", None),
    "lmstudio": ("http://localhost:1234/v1", None),
    "llamacpp": ("http://localhost:8080/v1", None),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "openai-compatible": (None, "OPENAI_API_KEY"),
}


class OpenAICompatibleModel(ChatModel):
    def __init__(self, model: str, *, provider: str = "openai-compatible", base_url: str | None = None,
                 api_key: str | None = None, api_key_env: str | None = None,
                 temperature: float | None = 0.2, top_p: float | None = None,
                 max_tokens: int = 4096, timeout: float = 120.0, max_retries: int = 4,
                 client=None) -> None:
        super().__init__(model)
        self.provider = provider
        preset_url, preset_env = PRESETS.get(provider, (None, "OPENAI_API_KEY"))
        self.base_url = base_url or os.environ.get(f"SCHOLARLENS_{provider.upper().replace('-', '_')}_URL") \
            or preset_url
        self.temperature, self.top_p, self.max_tokens = temperature, top_p, max_tokens
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as e:  # pragma: no cover
                raise ModelConfigError("pip install 'scholarlens[openai]'") from e
            env = api_key_env or preset_env
            key = api_key or (os.environ.get(env) if env else None) or ("not-needed" if env is None else None)
            if key is None:
                raise ModelConfigError(f"set {env} to use provider {provider!r}")
            client = OpenAI(base_url=self.base_url, api_key=key, timeout=timeout, max_retries=max_retries)
        self.client = client

    def describe(self) -> dict:
        return {"provider": self.provider, "model": self.model, "base_url": self.base_url,
                "temperature": self.temperature, "top_p": self.top_p}

    def generate(self, system: str, messages: list[dict[str, str]], *,
                 max_tokens: int | None = None) -> Generation:
        import openai

        kwargs: dict = {"model": self.model,
                        "messages": [{"role": "system", "content": system}, *messages]}
        limit = max_tokens or self.max_tokens
        # OpenAI's own newer models take max_completion_tokens; compatible servers take max_tokens
        kwargs["max_completion_tokens" if self.provider == "openai" else "max_tokens"] = limit
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except openai.AuthenticationError as e:
            raise ModelConfigError(f"{self.provider}: credentials rejected: {e}") from e
        except openai.NotFoundError as e:
            raise ModelConfigError(f"{self.provider}: unknown model {self.model!r}: {e}") from e
        except openai.APIStatusError as e:
            raise ModelError(f"{self.provider}: API error {e.status_code}: {e}") from e
        except openai.APIConnectionError as e:
            raise ModelError(f"{self.provider}: cannot reach {self.base_url or 'api.openai.com'}: {e}") from e
        choice = resp.choices[0]
        refusal = getattr(choice.message, "refusal", None)
        usage = {}
        if getattr(resp, "usage", None) is not None:
            usage = {"input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens}
        return Generation(text=choice.message.content or "", provider=self.provider,
                          model=getattr(resp, "model", self.model) or self.model,
                          stop_reason=choice.finish_reason, refused=bool(refusal), usage=usage,
                          request_id=getattr(resp, "id", None))
