import pytest

from gta_felirat.backends import ClaudeBackend
from gta_felirat.translator import (CachedTranslator, TranslationError, TranslatorChoice, create_backend,
                                    resolve_backend_name, resolve_key, split_speaker)


class FakeBackend:
    def __init__(self, responses=None, error=None):
        self.calls = []
        self._responses = responses or {}
        self._error = error

    def translate(self, text):
        self.calls.append(text)
        if self._error:
            raise self._error
        return self._responses.get(text, f"HU({text})")


@pytest.mark.parametrize("text, expected", [
    ("Jason: We go now.", ("Jason", "We go now.")),
    ("Lucia: Wait!", ("Lucia", "Wait!")),
    ("We go now.", ("", "We go now.")),
    ("Listen to me very carefully: go", ("", "Listen to me very carefully: go")),
    ("time 10:30 now", ("", "time 10:30 now")),
])
def test_split_speaker(text, expected):
    assert split_speaker(text) == expected


def test_keeps_speaker_name_and_translates_only_the_line():
    backend = FakeBackend({"We go now.": "Most megyünk."})
    assert CachedTranslator(backend).translate("Jason: We go now.") == "Jason: Most megyünk."
    assert backend.calls == ["We go now."]


def test_repeated_line_is_served_from_cache():
    backend = FakeBackend()
    translator = CachedTranslator(backend)
    translator.translate("Hello.")
    translator.translate("hello.")
    assert backend.calls == ["Hello."]


def test_cache_evicts_oldest_entry_when_full():
    backend = FakeBackend()
    translator = CachedTranslator(backend, cache_size=2)
    for text in ("a", "b", "c", "a"):
        translator.translate(text)
    assert backend.calls == ["a", "b", "c", "a"]


def test_backend_failure_is_wrapped_in_translation_error():
    translator = CachedTranslator(FakeBackend(error=ConnectionError("offline")))
    with pytest.raises(TranslationError, match="offline"):
        translator.translate("Hello")


def test_empty_backend_answer_is_an_error():
    translator = CachedTranslator(FakeBackend({"Hello": "  "}))
    with pytest.raises(TranslationError):
        translator.translate("Hello")


def choice(name, lang="hu"):
    return TranslatorChoice(name, lang, "claude-haiku-5-5", "gpt-4.1-mini")


@pytest.mark.parametrize("name, env_var", [
    ("deepl", "DEEPL_API_KEY"), ("claude", "ANTHROPIC_API_KEY"), ("openai", "OPENAI_API_KEY")])
def test_missing_key_is_rejected_with_variable_name(monkeypatch, name, env_var):
    monkeypatch.delenv(env_var, raising=False)
    with pytest.raises(TranslationError, match=env_var):
        create_backend(choice(name))


def test_auto_prefers_deepl_then_claude_then_openai(monkeypatch):
    for var in ("DEEPL_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert resolve_backend_name("auto", {}) == "google"
    assert resolve_backend_name("auto", {"openai": "k"}) == "openai"
    assert resolve_backend_name("auto", {"openai": "k", "claude": "k"}) == "claude"
    monkeypatch.setenv("DEEPL_API_KEY", "x:fx")
    assert resolve_backend_name("auto", {"claude": "k"}) == "deepl"


def test_session_key_wins_over_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    assert resolve_key("claude", {"claude": "typed"}) == "typed"
    assert resolve_key("claude", {}) == "from-env"


def test_backend_for_explicit_choice_is_created(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert isinstance(create_backend(choice("claude", "de")), ClaudeBackend)
