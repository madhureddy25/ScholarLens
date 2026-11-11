"""Provider-neutral chat-model interface.

Every backend — hosted API, local server, or in-process model — implements
``ChatModel.generate``. Quill and Bench only ever talk to this interface, so a
model is swapped by changing one spec string (``anthropic:claude-opus-5-5``,
``ollama:llama3.1``, ``hf:Qwen/Qwen2.5-1.5B-Instruct``) or a profile name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class ModelError(RuntimeError):
    """A request failed after the backend's own retries (network, rate limit, server)."""


class ModelConfigError(ModelError):
    """Misconfiguration: missing credentials, unknown provider, missing optional package."""


@dataclass
class Generation:
    text: str
    provider: str
    model: str
    """Model that actually served the request (may differ after a server-side fallback)."""
    stop_reason: str | None = None
    refused: bool = False
    usage: dict = field(default_factory=dict)
    request_id: str | None = None


class ChatModel(ABC):
    provider: str = "base"

    def __init__(self, model: str) -> None:
        self.model = model

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    @abstractmethod
    def generate(self, system: str, messages: list[dict[str, str]], *,
                 max_tokens: int | None = None) -> Generation:
        """``messages`` alternate ``{"role": "user"|"assistant", "content": str}``."""

    def describe(self) -> dict:
        """Settings recorded in the disclosure manifest. Never include secrets."""
        return {"provider": self.provider, "model": self.model}
