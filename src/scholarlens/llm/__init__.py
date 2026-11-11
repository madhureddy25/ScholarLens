"""Switchable model layer: one interface, any provider."""

from scholarlens.llm.base import ChatModel, Generation, ModelConfigError, ModelError
from scholarlens.llm.registry import available_providers, load_model, load_profiles, register_provider

__all__ = ["ChatModel", "Generation", "ModelConfigError", "ModelError", "available_providers",
           "load_model", "load_profiles", "register_provider"]
