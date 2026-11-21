import logging
import socket
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

from gta_felirat import backends
from gta_felirat.backends import (BackendError, ClaudeBackend, DeepLBackend, GoogleBackend, OpenAIBackend,
                                  build_prompt)
from gta_felirat.languages import find_language

HELLO = {"translations": [{"text": "Szia."}]}
# Válasz (vagy hiba), amely ennyi másodperccel a kérés elküldése után érkezik meg (a hamis órán)
After = namedtuple("After", ["seconds", "result"])


class FakeResponse:
    def __init__(self, status_code, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self):
        return self._payload


class FakeClock:
    """A backends modul órája: csak akkor megy előre, ha a teszt vagy egy sleep lépteti."""

    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Biztosíték: ha egy hamisítvány kimaradna, a teszt hibázzon, ne valódi szolgáltatást hívjon."""
    def refuse(*_args, **_kwargs):
        raise AssertionError("a test tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(backends, "time", fake)
    return fake


@pytest.fixture
def fake_post(monkeypatch, clock):
    calls = []
    responses = []

    def post(url, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        result = responses.pop(0)
        if isinstance(result, After):
            clock.now += result.seconds
            result = result.result
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(backends._session, "post", post)
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


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504, 529])
def test_overload_is_retried_after_a_short_pause(fake_post, clock, status):
    calls, responses = fake_post
    responses.extend([FakeResponse(status), FakeResponse(200, HELLO)])

    assert DeepLBackend("abc:fx", find_language("hu")).translate("Hi.") == "Szia."

    assert clock.sleeps == [0.3]
    assert len(calls) == 2


def test_dropped_connection_is_retried_at_once_without_a_pause(fake_post, clock):
    calls, responses = fake_post
    responses.extend([requests.ConnectionError("connection reset"), FakeResponse(200, HELLO)])

    assert DeepLBackend("abc:fx", find_language("hu")).translate("Hi.") == "Szia."

    assert clock.sleeps == []
    assert len(calls) == 2


@pytest.mark.parametrize("retry_after, pause", [
    ("0.1", 0.1), ("1", 1.0), ("30", 1.0), ("0", 0.0), ("-5", 0.0),
    ("nan", 0.3), ("inf", 0.3), ("Wed, 21 Oct 2026 07:28:00 GMT", 0.3)])
def test_pause_follows_retry_after_but_never_exceeds_a_second(fake_post, clock, retry_after, pause):
    _, responses = fake_post
    responses.extend([FakeResponse(429, headers={"Retry-After": retry_after}), FakeResponse(200, HELLO)])

    DeepLBackend("abc:fx", find_language("hu")).translate("Hi.")

    assert clock.sleeps == [pause]


def test_lasting_overload_is_reported_after_a_single_retry(fake_post, clock):
    calls, responses = fake_post
    responses.extend([FakeResponse(503), FakeResponse(503)])

    with pytest.raises(BackendError, match="overloaded"):
        DeepLBackend("abc:fx", find_language("hu")).translate("Hi.")

    assert len(calls) == 2
    assert clock.sleeps == [0.3]


@pytest.mark.parametrize("slow_answer, message", [
    (After(8.5, requests.ConnectionError("slow network")), "Network error"),
    (After(7.8, FakeResponse(429)), "overloaded")])  # 2,2 mp maradt: a szünettel együtt már nem fér bele
def test_slow_failure_is_not_retried_when_too_little_time_is_left(fake_post, clock, slow_answer, message):
    calls, responses = fake_post
    responses.append(slow_answer)

    with pytest.raises(BackendError, match=message):
        DeepLBackend("abc:fx", find_language("hu")).translate("Hi.")

    assert len(calls) == 1
    assert clock.sleeps == []


def test_retry_may_only_use_the_time_left_of_the_budget(fake_post, clock):
    calls, responses = fake_post
    responses.extend([After(4.0, requests.ConnectionError("connection reset")), FakeResponse(200, HELLO)])

    DeepLBackend("abc:fx", find_language("hu")).translate("Hi.")

    assert calls[0]["timeout"] == backends.REQUEST_TIMEOUT_SECONDS
    assert calls[1]["timeout"] == pytest.approx(backends.TRANSLATION_BUDGET_SECONDS - 4.0)


@pytest.mark.parametrize("status", [400, 404, 413, 456])
def test_request_the_service_rejects_is_not_sent_again(fake_post, clock, status):
    calls, responses = fake_post
    responses.append(FakeResponse(status))

    with pytest.raises(BackendError, match=f"HTTP {status}"):
        DeepLBackend("abc:fx", find_language("hu")).translate("Hi.")

    assert len(calls) == 1
    assert clock.sleeps == []


def test_bad_key_gives_clear_error_without_leaking_key(fake_post):
    _, responses = fake_post
    responses.append(FakeResponse(401))
    with pytest.raises(BackendError) as err:
        ClaudeBackend("secret-key", find_language("hu"), "m").translate("Hi")
    assert "secret-key" not in str(err.value)
    assert "API key" in str(err.value)


def test_network_failure_after_retry_is_reported(fake_post):
    _, responses = fake_post
    responses.extend([requests.ConnectionError("x"), requests.ConnectionError("x")])
    with pytest.raises(BackendError, match="Network error"):
        OpenAIBackend("k", find_language("hu"), "m").translate("Hi")


def test_prompt_without_context_is_just_the_line():
    system, user = build_prompt(find_language("pl"), [], "Hello")
    assert "Polish" in system
    assert user == "Hello"


def test_unknown_language_code_still_works():
    language = find_language("lt")
    assert language.code == "lt" and language.english_name == "lt"


# --- mért válaszidő a naplóban -------------------------------------------------

def test_each_request_logs_its_round_trip_time_and_size_at_info_level(fake_post, caplog):
    caplog.set_level(logging.INFO, logger="gta_felirat.backends")
    _, responses = fake_post
    answer = FakeResponse(200, {"translations": [{"text": "Egy."}, {"text": "Kettő."}]})
    responses.extend([After(0.02, requests.ConnectionError("connection reset")), After(0.187, answer)])

    DeepLBackend("abc:fx", find_language("hu")).translate_many(["One.", "Two two."])

    assert [(record.levelname, record.getMessage()) for record in caplog.records] == [
        ("INFO", "DeepL round trip: 20 ms, ConnectionError, 2 texts, 12 chars"),
        ("INFO", "DeepL round trip: 187 ms, HTTP 200, 2 texts, 12 chars")]


def test_request_log_contains_neither_the_key_nor_the_subtitle_text(fake_post, caplog):
    caplog.set_level(logging.DEBUG)
    _, responses = fake_post
    responses.append(FakeResponse(200, HELLO))

    DeepLBackend("secret-key:fx", find_language("hu")).translate("We go now.")

    assert caplog.records
    assert not any(secret in caplog.text for secret in ("secret-key", "Authorization", "We go now."))


# --- Google: időkorlát ---------------------------------------------------------

class FakeGooglePage:
    """A Google fordítóoldalának válasza abban a formában, ahogy a deep-translator feldolgozza."""

    status_code = 200

    def __init__(self, translated):
        self.text = f'<div class="result-container">{translated}</div>'

    def close(self):
        pass


def test_google_request_gets_a_timeout_and_its_round_trip_is_logged(monkeypatch, clock, caplog):
    caplog.set_level(logging.INFO, logger="gta_felirat.backends")
    requested = []

    def get(url, **options):
        requested.append(options)
        clock.now += 0.25
        return FakeGooglePage("Szia.")

    monkeypatch.setattr(requests, "get", get)

    assert GoogleBackend(find_language("hu")).translate("Hello.") == "Szia."
    assert [options["timeout"] for options in requested] == [backends.REQUEST_TIMEOUT_SECONDS]
    assert [record.getMessage() for record in caplog.records] == ["Google round trip: 250 ms, OK, 1 texts, 6 chars"]


@pytest.mark.parametrize("error", [
    requests.ReadTimeout("read timed out"), requests.ConnectionError("Max retries exceeded with url: /m?q=Hello.")])
def test_google_network_failure_is_reported_without_the_subtitle_text(monkeypatch, error):
    def get(url, **options):
        raise error

    monkeypatch.setattr(requests, "get", get)

    with pytest.raises(BackendError, match="Network error") as failure:
        GoogleBackend(find_language("hu")).translate("Hello.")

    assert "Hello" not in str(failure.value)


def test_google_page_without_a_translation_is_reported_without_the_subtitle_text(monkeypatch):
    class EmptyPage(FakeGooglePage):
        def __init__(self):
            self.text = "<div>nothing here</div>"

    monkeypatch.setattr(requests, "get", lambda url, **options: EmptyPage())

    with pytest.raises(BackendError) as failure:
        GoogleBackend(find_language("hu")).translate("Hello.")

    assert "Hello" not in str(failure.value)


def test_google_requests_from_several_threads_keep_their_own_text(monkeypatch):
    # A kliens a kérés szövegét közös állapotban tartja: egyszerre hívva összekeveredhetne
    def get(url, params, **options):
        time.sleep(0.02)  # közben a másik szál is kérne
        return FakeGooglePage("HU " + params["q"])

    monkeypatch.setattr(requests, "get", get)
    backend = GoogleBackend(find_language("hu"))
    texts = [f"Line {number}." for number in range(6)]

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(backend.translate, texts))

    assert results == ["HU " + text for text in texts]
