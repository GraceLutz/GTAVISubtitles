import re
import socket
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from gta_felirat import backends, translation_store
from gta_felirat.backends import BackendError, DeepLBackend
from gta_felirat.languages import find_language
from gta_felirat.translation_service import TranslationService
from gta_felirat.translation_store import TranslationStore
from gta_felirat.translator import CachedTranslator, TranslationError, TranslatorChoice

CHOICE = TranslatorChoice("deepl", "hu", "claude-haiku-5-5", "gpt-4.1-mini")
HUNGARIAN = find_language("hu")
TROUBLE = "We ain't lookin' for trouble, okay?"
HOW = "How about you, how you doin'?"


class BatchBackend:
    """Mint a DeepL: több sort is lefordít egyetlen kérésben; a spoil a kész válaszlistát rontja el."""

    def __init__(self, spoil=lambda answers: answers):
        self.batches = []
        self._spoil = spoil

    def translate(self, text):
        return self.translate_many([text])[0]

    def translate_many(self, texts):
        self.batches.append(list(texts))
        return self._spoil([f"HU({text})" for text in texts])


class LineBackend:
    """Mint az AI- és a Google-fordító: csak soronként fordít; a fail_on sornál megszakad a kapcsolat."""

    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def translate(self, text):
        self.calls.append(text)
        if text == self.fail_on:
            raise ConnectionError("offline")
        return f"HU({text})"


class FakeResponse:
    status_code = 200
    headers = {}

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Biztosíték: ha egy hamisítvány kimaradna, a teszt hibázzon, ne valódi szolgáltatást hívjon."""
    def refuse(*_args, **_kwargs):
        raise AssertionError("a test tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)


@pytest.fixture
def deepl_posts(monkeypatch):
    """A DeepL helyett válaszoló hamis munkamenet: minden sorra „HU(sor)”; a kérések a listába kerülnek."""
    posts = []

    def post(url, headers, json, timeout):
        posts.append({"url": url, "headers": headers, "json": json})
        return FakeResponse({"translations": [{"text": f"HU({text})"} for text in json["text"]]})

    monkeypatch.setattr(backends._session, "post", post)
    return posts


# --- több új sor egyetlen kérésben ---------------------------------------------

def test_two_new_subtitle_lines_cost_one_deepl_request_and_cached_ones_none(deepl_posts):
    translator = CachedTranslator(DeepLBackend("abc:fx", HUNGARIAN))

    first = translator.translate(f"Jason: {TROUBLE}\nLucia: {HOW}")
    second = translator.translate(f"Jason: {TROUBLE}\nLucia: {HOW}")

    assert [post["json"]["text"] for post in deepl_posts] == [[TROUBLE, HOW]]
    assert first == second == f"Jason: HU({TROUBLE})\nLucia: HU({HOW})"


def test_only_new_lines_are_requested_and_the_answer_keeps_the_order_and_the_line_count():
    backend = BatchBackend()
    translator = CachedTranslator(backend)
    translator.translate("Lucia: Two.")

    result = translator.translate("Jason: One.\nLucia: Two.\nJason: Three.")

    assert backend.batches == [["Two."], ["One.", "Three."]]
    assert result.split("\n") == ["Jason: HU(One.)", "Lucia: HU(Two.)", "Jason: HU(Three.)"]


def test_line_repeated_within_one_subtitle_is_requested_once():
    backend = BatchBackend()

    result = CachedTranslator(backend).translate("Lucia: Go!\nJason: GO!")

    assert backend.batches == [["Go!"]]
    assert result == "Lucia: HU(Go!)\nJason: HU(Go!)"


def test_speaker_line_and_language_note_share_one_request():
    backend = BatchBackend()
    translator = CachedTranslator(backend)

    first = translator.translate("In Spanish: Eyes on the road!")
    translator.translate("In Spanish: Stop!")

    assert first == "HU(In Spanish): HU(Eyes on the road!)"
    assert backend.batches == [["Eyes on the road!", "In Spanish"], ["Stop!"]]


def test_backend_without_translate_many_is_asked_line_by_line():
    backend = LineBackend()

    result = CachedTranslator(backend).translate(f"Jason: {TROUBLE}\n{HOW}")

    assert backend.calls == [TROUBLE, HOW]
    assert result == f"Jason: HU({TROUBLE})\nHU({HOW})"


@pytest.mark.parametrize("text", [
    "We go now.", "Jason: We go now.", "In Spanish: Eyes on the road!", "One.\nJason: Two.\nOne.",
    "Vagrant: Take a breath now. Lucia: Stop talking.", "  ", ""])
def test_batch_and_line_by_line_backends_give_the_same_translation(text):
    assert CachedTranslator(BatchBackend()).translate(text) == CachedTranslator(LineBackend()).translate(text)


# --- hibás válaszok --------------------------------------------------------------

@pytest.mark.parametrize("spoil", [
    lambda answers: answers[:-1], lambda answers: [*answers, "egy felesleges"], lambda answers: []])
def test_wrong_number_of_answers_is_an_error_and_nothing_is_kept(tmp_path, spoil):
    store = TranslationStore(tmp_path / "mentett.json")

    with pytest.raises(TranslationError):
        CachedTranslator(BatchBackend(spoil), store=store, namespace="deepl:hu").translate("One.\nTwo.")

    assert [store.get("deepl:hu", key) for key in ("one.", "two.")] == [None, None]


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_blank_answer_is_an_error_but_the_lines_answered_before_it_are_kept(tmp_path, blank):
    store = TranslationStore(tmp_path / "mentett.json")
    backend = BatchBackend(lambda answers: [answers[0], blank])

    with pytest.raises(TranslationError):
        CachedTranslator(backend, store=store, namespace="deepl:hu").translate("One.\nTwo.")

    assert [store.get("deepl:hu", key) for key in ("one.", "two.")] == ["HU(One.)", None]


def test_failed_batch_request_is_reported_and_leaves_nothing_on_disk(tmp_path):
    def fail(_answers):
        raise ConnectionError("offline")

    store = TranslationStore(tmp_path / "mentett.json")

    with pytest.raises(TranslationError, match="offline"):
        CachedTranslator(BatchBackend(fail), store=store, namespace="deepl:hu").translate("One.\nTwo.")

    assert not (tmp_path / "mentett.json").exists()


def test_lines_translated_before_a_failure_are_not_requested_again():
    backend = LineBackend(fail_on="Two.")
    translator = CachedTranslator(backend)
    with pytest.raises(TranslationError, match="offline"):
        translator.translate("One.\nTwo.\nThree.")
    backend.fail_on = None

    result = translator.translate("One.\nTwo.\nThree.")

    assert backend.calls == ["One.", "Two.", "Two.", "Three."]
    assert result == "HU(One.)\nHU(Two.)\nHU(Three.)"


def test_line_answered_first_comes_from_the_cache_while_the_next_one_is_still_being_translated():
    second_started, may_answer = threading.Event(), threading.Event()

    class StallingBackend(LineBackend):
        def translate(self, text):
            if text == "Two.":
                second_started.set()
                may_answer.wait(timeout=2)
            return super().translate(text)

    backend = StallingBackend()
    translator = CachedTranslator(backend)
    with ThreadPoolExecutor(max_workers=1) as pool:
        both = pool.submit(translator.translate, "One.\nTwo.")
        assert second_started.wait(timeout=2)
        first = translator.translate("One.")
        may_answer.set()
        assert both.result(timeout=2) == "HU(One.)\nHU(Two.)"

    assert first == "HU(One.)"
    assert backend.calls == ["One.", "Two."]


# --- DeepL: több sor a text tömbben ----------------------------------------------

def test_deepl_sends_all_lines_in_one_request_and_answers_in_order(deepl_posts):
    answers = DeepLBackend("abc:fx", HUNGARIAN).translate_many([TROUBLE, HOW])

    assert answers == [f"HU({TROUBLE})", f"HU({HOW})"]
    assert deepl_posts == [{"url": backends.DEEPL_FREE_URL, "headers": {"Authorization": "DeepL-Auth-Key abc:fx"},
                            "json": {"text": [TROUBLE, HOW], "source_lang": "EN", "target_lang": "HU"}}]


def test_deepl_request_for_a_single_line_is_unchanged(deepl_posts):
    assert DeepLBackend("abc:fx", HUNGARIAN).translate(TROUBLE) == f"HU({TROUBLE})"
    assert [post["json"] for post in deepl_posts] == [{"text": [TROUBLE], "source_lang": "EN", "target_lang": "HU"}]


def test_deepl_splits_more_lines_than_the_fifty_one_request_may_carry(deepl_posts):
    lines = [f"Line {number}." for number in range(51)]

    answers = DeepLBackend("abc:fx", HUNGARIAN).translate_many(lines)

    assert [len(post["json"]["text"]) for post in deepl_posts] == [50, 1]
    assert answers == [f"HU({line})" for line in lines]


@pytest.mark.parametrize("payload", [
    {"translations": []}, {"translations": [{"text": "Egy."}]}, {"message": "Bad request"},
    {"translations": [{"text": "Egy."}, {"text": "Kettő."}, {"text": "Három."}]},
    {"translations": [{"text": "Egy."}, {"detected_source_language": "EN"}]}, {"translations": None}])
def test_deepl_answer_that_does_not_match_the_request_is_an_error(monkeypatch, payload):
    monkeypatch.setattr(backends._session, "post", lambda url, headers, json, timeout: FakeResponse(payload))

    with pytest.raises(BackendError, match="DeepL"):
        DeepLBackend("abc:fx", HUNGARIAN).translate_many(["One.", "Two."])


# --- mentés: egy kérés, egy fájlírás ---------------------------------------------

def test_batch_of_translations_is_written_to_disk_once(tmp_path, monkeypatch):
    writes = []
    replace = translation_store.os.replace

    def counting_replace(source, target):
        writes.append(target)
        replace(source, target)

    monkeypatch.setattr(translation_store.os, "replace", counting_replace)
    store = TranslationStore(tmp_path / "mentett.json")

    CachedTranslator(BatchBackend(), store=store, namespace="deepl:hu").translate("One.\nTwo.\nThree.")

    assert writes == [tmp_path / "mentett.json"]


def test_saved_batch_is_reused_after_restart_without_a_request(tmp_path):
    path = tmp_path / "mentett.json"
    CachedTranslator(BatchBackend(), store=TranslationStore(path), namespace="deepl:hu").translate("One.\nTwo.")
    backend = BatchBackend()

    restarted = CachedTranslator(backend, store=TranslationStore(path), namespace="deepl:hu")

    assert restarted.translate("Jason: Two.\nOne.") == "Jason: HU(Two.)\nHU(One.)"
    assert backend.batches == []


# --- a szolgáltatáson keresztül ---------------------------------------------------

def test_service_translates_two_new_lines_with_one_request_and_shows_the_latency():
    backend = BatchBackend()
    service = TranslationService(CHOICE, CachedTranslator(backend))

    assert service.translate(f"{TROUBLE}\n{HOW}") == f"HU({TROUBLE})\nHU({HOW})"
    assert backend.batches == [[TROUBLE, HOW]]
    assert re.search(r"last translation \d+ ms", service.status())


def test_prefetched_subtitle_is_not_requested_again_when_it_is_confirmed():
    backend = BatchBackend()
    service = TranslationService(CHOICE, CachedTranslator(backend))

    with ThreadPoolExecutor(max_workers=1) as pool:
        service.prefetch(f"{TROUBLE}\n{HOW}", pool)
        confirmed = service.translate(f"{TROUBLE}\n{HOW}")

    assert confirmed == f"HU({TROUBLE})\nHU({HOW})"
    assert backend.batches == [[TROUBLE, HOW]]
