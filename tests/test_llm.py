from types import SimpleNamespace

import pytest

from scholarlens.llm import ModelConfigError, available_providers, load_model, register_provider
from scholarlens.llm.registry import parse_options
from scholarlens.llm.testing import EchoModel, ScriptedModel


def test_parse_options_types():
    assert parse_options(["effort=low", "temperature=0.3", "max_tokens=512", "fallbacks=false", "x=none"]) == {
        "effort": "low", "temperature": 0.3, "max_tokens": 512, "fallbacks": False, "x": None}


def test_builtin_providers_listed():
    provs = available_providers()
    for p in ("anthropic", "openai", "ollama", "vllm", "openai-compatible", "hf", "openrouter", "echo"):
        assert p in provs


def test_spec_string_and_overrides():
    m = load_model("scripted:my-model", responses=["hi"])
    assert m.name == "scripted:my-model"
    assert m.generate("sys", [{"role": "user", "content": "x"}]).text == "hi"
    assert isinstance(load_model("echo"), EchoModel)


def test_profiles_from_toml(tmp_path, monkeypatch):
    cfg = tmp_path / "scholarlens.toml"
    cfg.write_text('default = "fake"\n[models.fake]\nprovider = "scripted"\nmodel = "from-profile"\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SCHOLARLENS_MODEL", raising=False)
    m = load_model(responses=["ok"])  # falls back to the default profile
    assert m.name == "scripted:from-profile"
    assert load_model("fake", responses=["ok"]).model == "from-profile"


def test_env_var_selects_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SCHOLARLENS_MODEL", "echo")
    assert isinstance(load_model(), EchoModel)


def test_errors_are_helpful(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SCHOLARLENS_MODEL", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(ModelConfigError, match="no model selected"):
        load_model()
    with pytest.raises(ModelConfigError, match="unknown provider"):
        load_model("nope:model")
    with pytest.raises(ModelConfigError, match="specify a model"):
        load_model("ollama")


def test_register_custom_provider():
    register_provider("mylab", lambda model, **kw: ScriptedModel(["custom"], model=model or "x"))
    assert load_model("mylab:edit-v2").generate("s", [{"role": "user", "content": "q"}]).text == "custom"


def test_scripted_refusal():
    g = ScriptedModel(["<<REFUSE>>"]).generate("s", [{"role": "user", "content": "q"}])
    assert g.refused and g.text == ""


# ------------------------------------------------------------ Anthropic backend
class _FakeAnthropic:
    def __init__(self, response):
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self._response = response

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def _resp(text, stop="end_turn", model="claude-opus-5-5"):
    return SimpleNamespace(
        stop_reason=stop, model=model, _request_id="req_1",
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0,
                              cache_creation_input_tokens=0))


def test_anthropic_request_shape():
    pytest.importorskip("anthropic")
    from scholarlens.llm.anthropic_backend import AnthropicModel

    fake = _FakeAnthropic(_resp("Edited paragraph."))
    m = AnthropicModel(client=fake)
    g = m.generate("SYSTEM", [{"role": "user", "content": "p"}])
    kw = fake.calls[0]
    assert kw["model"] == "claude-opus-5-5"
    assert kw["output_config"] == {"effort": "medium"}
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "temperature" not in kw and "top_p" not in kw and "thinking" not in kw
    assert g.text == "Edited paragraph." and not g.refused and g.usage["input_tokens"] == 10
    assert m.describe()["refusal_fallbacks"] is True


def test_anthropic_refusal_and_haiku_without_fallbacks():
    pytest.importorskip("anthropic")
    from scholarlens.llm.anthropic_backend import AnthropicModel

    fake = _FakeAnthropic(_resp("", stop="refusal"))
    g = AnthropicModel(client=fake).generate("s", [{"role": "user", "content": "p"}])
    assert g.refused and g.text == ""
    fake2 = _FakeAnthropic(_resp("x", model="claude-haiku-5-5"))
    AnthropicModel("claude-haiku-5-5", client=fake2, effort="low").generate("s", [{"role": "user", "content": "p"}])
    assert "fallbacks" not in fake2.calls[0] and fake2.calls[0]["output_config"] == {"effort": "low"}


# ------------------------------------------------------ OpenAI-compatible backend
class _FakeOpenAI:
    def __init__(self, text):
        self.calls = []
        self.text = text
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        msg = SimpleNamespace(content=self.text, refusal=None)
        return SimpleNamespace(id="cmpl", model=kwargs["model"], usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
                               choices=[SimpleNamespace(message=msg, finish_reason="stop")])


def test_openai_compatible_request_shape():
    pytest.importorskip("openai")
    from scholarlens.llm.openai_compat import OpenAICompatibleModel

    fake = _FakeOpenAI("done")
    m = OpenAICompatibleModel("llama3.1", provider="ollama", client=fake, temperature=0.1)
    g = m.generate("SYS", [{"role": "user", "content": "p"}], max_tokens=99)
    kw = fake.calls[0]
    assert kw["messages"][0] == {"role": "system", "content": "SYS"}
    assert kw["max_tokens"] == 99 and kw["temperature"] == 0.1
    assert g.text == "done" and g.provider == "ollama"
    assert m.base_url == "http://localhost:11434/v1"

    fake2 = _FakeOpenAI("x")
    OpenAICompatibleModel("some-model", provider="openai", client=fake2).generate("s", [{"role": "user", "content": "p"}])
    assert "max_completion_tokens" in fake2.calls[0] and "max_tokens" not in fake2.calls[0]


def test_openai_requires_key_but_local_servers_do_not(monkeypatch):
    pytest.importorskip("openai")
    from scholarlens.llm.openai_compat import OpenAICompatibleModel

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ModelConfigError, match="OPENAI_API_KEY"):
        OpenAICompatibleModel("m", provider="openai")
    assert OpenAICompatibleModel("llama3.1", provider="ollama").client is not None


def test_registering_before_first_load_keeps_builtins(monkeypatch):
    """Regression: a provider registered before any load must not hide the built-ins."""
    from scholarlens.llm import registry

    monkeypatch.setattr(registry, "_PROVIDERS", {})
    monkeypatch.setattr(registry, "_BUILTINS_LOADED", False)
    register_provider("early", lambda model, **kw: ScriptedModel(["x"], model=model))
    provs = available_providers()
    assert "early" in provs and "anthropic" in provs and "ollama" in provs
    register_provider("echo", lambda model, **kw: ScriptedModel(["override"], model="o"))
    assert load_model("echo").generate("s", [{"role": "user", "content": "q"}]).text == "override"
