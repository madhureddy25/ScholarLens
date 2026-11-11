"""Claude via the official Anthropic Python SDK.

Credentials resolve the SDK's usual way (``ANTHROPIC_API_KEY``,
``ANTHROPIC_AUTH_TOKEN``, or an ``ant auth login`` profile).

Current Claude models choose their own sampling: there is no temperature/top_p,
and thinking depth is set with ``effort``. Server-side refusal fallbacks are
enabled by default on models that support them, so a classifier false positive
on a benign paragraph is retried on Anthropic's recommended fallback model
instead of failing the edit.
"""

from __future__ import annotations

from scholarlens.llm.base import ChatModel, Generation, ModelConfigError, ModelError

DEFAULT_MODEL = "claude-opus-5-5"
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Model families with server-side refusal fallback ("default" routing).
_FALLBACK_FAMILIES = ("claude-fable-5", "claude-mythos-5", "claude-opus-5", "claude-sonnet-5-5")


class AnthropicModel(ChatModel):
    provider = "anthropic"

    def __init__(self, model: str = DEFAULT_MODEL, *, effort: str | None = "medium",
                 fallbacks: bool | None = None, max_tokens: int = 16000, max_retries: int = 4,
                 timeout: float = 120.0, client=None) -> None:
        super().__init__(model or DEFAULT_MODEL)
        self.effort = effort
        self.max_tokens = max_tokens
        self.fallbacks = (any(self.model.startswith(f) for f in _FALLBACK_FAMILIES)
                          if fallbacks is None else fallbacks)
        if client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover
                raise ModelConfigError("pip install 'scholarlens[anthropic]'") from e
            client = anthropic.Anthropic(max_retries=max_retries, timeout=timeout)
        self.client = client

    def describe(self) -> dict:
        return {"provider": self.provider, "model": self.model, "effort": self.effort,
                "refusal_fallbacks": self.fallbacks}

    def generate(self, system: str, messages: list[dict[str, str]], *,
                 max_tokens: int | None = None) -> Generation:
        import anthropic

        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            # cache the (stable) instructions; the per-paragraph user turn varies
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": messages,
        }
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        if self.fallbacks:
            kwargs["betas"] = [_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        try:
            resp = self.client.beta.messages.create(**kwargs)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
            raise ModelConfigError(f"Anthropic credentials rejected: {e}") from e
        except anthropic.NotFoundError as e:
            raise ModelConfigError(f"unknown Claude model {self.model!r}: {e}") from e
        except anthropic.BadRequestError as e:
            raise ModelError(f"bad request: {e}") from e
        except anthropic.RateLimitError as e:
            raise ModelError(f"rate limited after retries: {e}") from e
        except anthropic.APIStatusError as e:
            raise ModelError(f"API error {e.status_code}: {e}") from e
        except anthropic.APIConnectionError as e:
            raise ModelError(f"connection error: {e}") from e

        usage = {}
        if getattr(resp, "usage", None) is not None:
            usage = {k: getattr(resp.usage, k, None) for k in
                     ("input_tokens", "output_tokens", "cache_read_input_tokens",
                      "cache_creation_input_tokens")}
        refused = resp.stop_reason == "refusal"
        text = "" if refused else "".join(b.text for b in resp.content if b.type == "text")
        return Generation(text=text, provider=self.provider, model=getattr(resp, "model", self.model),
                          stop_reason=resp.stop_reason, refused=refused, usage=usage,
                          request_id=getattr(resp, "_request_id", None))
