"""Deterministic models for tests, dry runs and CI (no network, no weights)."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from scholarlens.llm.base import ChatModel, Generation


class ScriptedModel(ChatModel):
    """Returns queued responses in order, or computes them with a function of the
    last user message. Records every call for assertions."""

    provider = "scripted"

    def __init__(self, responses: Iterable[str] | Callable[[str], str] = (), model: str = "scripted") -> None:
        super().__init__(model)
        self._fn = responses if callable(responses) else None
        self._queue = [] if callable(responses) else list(responses)
        self.calls: list[dict] = []

    def generate(self, system, messages, *, max_tokens=None) -> Generation:
        self.calls.append({"system": system, "messages": [dict(m) for m in messages]})
        last = messages[-1]["content"]
        if self._fn is not None:
            text = self._fn(last)
        elif self._queue:
            text = self._queue.pop(0)
        else:
            raise AssertionError("ScriptedModel ran out of responses")
        refused = text == "<<REFUSE>>"
        return Generation(text="" if refused else text, provider=self.provider, model=self.model,
                          stop_reason="refusal" if refused else "end_turn", refused=refused)


class EchoModel(ScriptedModel):
    """Returns the paragraph it was given unchanged (pipeline smoke tests)."""

    provider = "echo"

    def __init__(self, model: str = "echo") -> None:
        from scholarlens.edit.prompts import extract_paragraph

        super().__init__(extract_paragraph, model=model)
