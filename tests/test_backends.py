import pytest
import requests

from gta_felirat import backends
from gta_felirat.backends import (BackendError, ClaudeBackend, DeepLBackend, OpenAIBackend,
                                  build_prompt)
from gta_felirat.languages import find_language


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


@pytest.fixture
def fake_post(monkeypatch):
    calls = []
    responses = []

    def post(url, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json})
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(backends._session, "post", post)
    monkeypatch.setattr(backends.time, "sleep", lambda _s: None)
    return calls, responses


def test_deepl_free_key_uses_free_endpoint_and_language_code(fake_post):
    calls, responses = fake_post
    responses.append(FakeResponse(200, {"translations": [{"text": "Los geht's."}]}))
    assert DeepLBackend("abc:fx", find_language("de")).translate("Let's go.") == "Los geht's."
    assert calls[0]["url"] == backends.DEEPL_FREE_URL
    assert calls[0]["json"]["target_lang"] == "DE"


def test_deepl_rejects_unsupported_language():
    with pytest.raises(BackendError, match="DeepL"):
        DeepLBackend("abc:fx", find_language("hr"))


def test_claude_request_and_answer(fake_post):
    calls, responses = fake_post
    responses.append(FakeResponse(200, {"content": [{"type": "text", "text": "Gyerünk."}]}))
    backend = ClaudeBackend("key", find_language("hu"), "claude-haiku-5-5")
    assert backend.translate("Let's go.") == "Gyerünk."
    request = calls[0]
    assert request["url"] == backends.CLAUDE_URL
    assert request["headers"]["x-api-key"] == "key"
    assert request["json"]["model"] == "claude-haiku-5-5"
    assert "Hungarian" in request["json"]["system"]


def test_llm_sends_previous_lines_as_context(fake_post):
    calls, responses = fake_post
    responses.extend([FakeResponse(200, {"choices": [{"message": {"content": "A"}}]}),
                      FakeResponse(200, {"choices": [{"message": {"content": "B"}}]})])
    backend = OpenAIBackend("key", find_language("es"), "gpt-4.1-mini")
    backend.translate("First line.")
    backend.translate("Second line.")
    second_user_message = calls[1]["json"]["messages"][1]["content"]
    assert "First line." in second_user_message and "Second line." in second_user_message
    assert calls[1]["headers"]["Authorization"] == "Bearer key"


def test_retries_once_on_overload_then_succeeds(fake_post):
    calls, responses = fake_post
    responses.extend([FakeResponse(529), FakeResponse(200, {"content": [{"type": "text", "text": "OK"}]})])
    assert ClaudeBackend("key", find_language("hu"), "m").translate("Hi") == "OK"
    assert len(calls) == 2


def test_bad_key_gives_clear_error_without_leaking_key(fake_post):
    _, responses = fake_post
    responses.append(FakeResponse(401))
    with pytest.raises(BackendError) as err:
        ClaudeBackend("secret-key", find_language("hu"), "m").translate("Hi")
    assert "secret-key" not in str(err.value)
    assert "API-kulcs" in str(err.value)


def test_network_failure_after_retry_is_reported(fake_post):
    _, responses = fake_post
    responses.extend([requests.ConnectionError("x"), requests.ConnectionError("x")])
    with pytest.raises(BackendError, match="Hálózati hiba"):
        OpenAIBackend("k", find_language("hu"), "m").translate("Hi")


def test_prompt_without_context_is_just_the_line():
    system, user = build_prompt(find_language("pl"), [], "Hello")
    assert "Polish" in system
    assert user == "Hello"


def test_unknown_language_code_still_works():
    language = find_language("lt")
    assert language.code == "lt" and language.english_name == "lt"
