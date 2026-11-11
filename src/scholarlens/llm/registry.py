"""Model registry: spec strings, named profiles, and third-party plugins.

Three ways to pick a model, all resolving to a ``ChatModel``:

1. **Spec string** ``provider:model``::

       anthropic:claude-opus-5-5        ollama:llama3.1:8b
       openai:<model-id>                hf:Qwen/Qwen2.5-1.5B-Instruct
       openrouter:<org>/<model>         vllm:<served-model-name>

2. **Named profile** from a TOML file (``scholarlens.toml`` in the working
   directory, ``~/.config/scholarlens/models.toml``, or ``$SCHOLARLENS_MODELS``)::

       default = "claude"
       [models.claude]
       provider = "anthropic"
       model = "claude-opus-5-5"
       effort = "medium"

       [models.lab-server]
       provider = "openai-compatible"
       base_url = "http://gpu-box:8000/v1"
       model = "my-finetuned-editor"

3. **Plugins**: any installed package can add a provider through the
   ``scholarlens.providers`` entry-point group (a callable ``(model, **opts) ->
   ChatModel``), or call ``register_provider`` at runtime.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Callable
from importlib.metadata import entry_points
from pathlib import Path

from scholarlens.llm.base import ChatModel, ModelConfigError

Factory = Callable[..., ChatModel]
_PROVIDERS: dict[str, Factory] = {}
_BUILTINS_LOADED = False


def register_provider(name: str, factory: Factory) -> None:
    """Add or replace a provider. Safe to call before or after built-ins are loaded."""
    _PROVIDERS[name.lower()] = factory


def _builtin() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    user_registered = dict(_PROVIDERS)  # providers registered earlier keep precedence

    def anthropic(model, **kw):
        from scholarlens.llm.anthropic_backend import AnthropicModel

        return AnthropicModel(model or None, **kw)

    def hf(model, **kw):
        from scholarlens.llm.hf_local import HFLocalModel

        return HFLocalModel(model, **kw)

    def scripted(model, **kw):
        from scholarlens.llm.testing import ScriptedModel

        return ScriptedModel(model=model or "scripted", **kw)

    def echo(model, **kw):
        from scholarlens.llm.testing import EchoModel

        return EchoModel(model or "echo")

    def compat(provider):
        def make(model, **kw):
            from scholarlens.llm.openai_compat import OpenAICompatibleModel

            if not model:
                raise ModelConfigError(f"{provider}: specify a model, e.g. {provider}:<model-name>")
            return OpenAICompatibleModel(model, provider=provider, **kw)

        return make

    for name, fac in (("anthropic", anthropic), ("claude", anthropic), ("hf", hf), ("local", hf),
                      ("scripted", scripted), ("echo", echo)):
        register_provider(name, fac)
    from scholarlens.llm.openai_compat import PRESETS

    for name in PRESETS:
        register_provider(name, compat(name))
    for ep in entry_points(group="scholarlens.providers"):
        try:
            register_provider(ep.name, ep.load())
        except Exception:  # a broken plugin must not break built-in providers
            continue
    _PROVIDERS.update(user_registered)


def available_providers() -> list[str]:
    _builtin()
    return sorted(_PROVIDERS)


def _profile_paths(path: str | Path | None) -> list[Path]:
    candidates = [path, os.environ.get("SCHOLARLENS_MODELS"), "scholarlens.toml",
                  Path.home() / ".config" / "scholarlens" / "models.toml"]
    return [Path(p) for p in candidates if p]


def load_profiles(path: str | Path | None = None) -> tuple[dict[str, dict], str | None]:
    """Return (profiles, default profile name) from the first config file found."""
    for p in _profile_paths(path):
        if p.is_file():
            data = tomllib.loads(p.read_text())
            return {k: dict(v) for k, v in data.get("models", {}).items()}, data.get("default")
    return {}, None


def parse_options(pairs: list[str] | None) -> dict:
    """``["effort=low", "temperature=0.3", "fallbacks=false"]`` → typed kwargs."""
    out: dict = {}
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        val: object = raw
        if raw.lower() in ("true", "false"):
            val = raw.lower() == "true"
        elif raw.lower() in ("none", "null"):
            val = None
        else:
            for cast in (int, float):
                try:
                    val = cast(raw)
                    break
                except ValueError:
                    pass
        out[key.strip().replace("-", "_")] = val
    return out


def load_model(spec: str | None = None, *, config: str | Path | None = None, **overrides) -> ChatModel:
    _builtin()
    profiles, default = load_profiles(config)
    spec = spec or os.environ.get("SCHOLARLENS_MODEL") or default
    if not spec:
        raise ModelConfigError(
            "no model selected: pass --model provider:model (e.g. anthropic:claude-opus-5-5, "
            "ollama:llama3.1, hf:Qwen/Qwen2.5-1.5B-Instruct), set SCHOLARLENS_MODEL, or add a "
            "default profile to scholarlens.toml")
    if spec in profiles:
        opts = dict(profiles[spec])
        provider = str(opts.pop("provider", "")).lower()
        model = str(opts.pop("model", ""))
    else:
        provider, _, model = spec.partition(":")
        provider, opts = provider.lower(), {}
    if provider not in _PROVIDERS:
        raise ModelConfigError(f"unknown provider {provider!r}; available: {', '.join(available_providers())}")
    return _PROVIDERS[provider](model, **{**opts, **overrides})
