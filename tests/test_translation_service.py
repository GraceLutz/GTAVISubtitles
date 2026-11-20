import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from gta_felirat.translation_service import TranslationService
from gta_felirat.translator import CachedTranslator, TranslatorChoice

CHOICE = TranslatorChoice("auto", "hu", "claude-haiku-5-5", "gpt-4.1-mini")


class SlowBackend:
    def __init__(self, delay=0.2, fail_first=False):
        self.calls = []
        self._delay = delay
        self._fail_first = fail_first
        self._lock = threading.Lock()

    def translate(self, text):
        with self._lock:
            self.calls.append(text)
            should_fail = self._fail_first and len(self.calls) == 1
        time.sleep(self._delay)
        if should_fail:
            raise ConnectionError("hiccup")
        return f"HU({text})"


@pytest.fixture
def pool():
    executor = ThreadPoolExecutor(max_workers=3)
    yield executor
    executor.shutdown(wait=True)


def test_confirmed_translation_reuses_prefetch_without_second_request(pool):
    backend = SlowBackend(delay=0.2)
    service = TranslationService(CHOICE, CachedTranslator(backend))
    service.prefetch("We go now.", pool)
    time.sleep(0.1)  # a megerősítés félúton érkezik
    started = time.monotonic()
    assert service.translate("We go now.") == "HU(We go now.)"
    assert time.monotonic() - started < 0.18  # csak a hátralévő időt várta ki
    assert backend.calls == ["We go now."]


def test_finished_prefetch_is_served_from_cache_instantly(pool):
    backend = SlowBackend(delay=0.05)
    service = TranslationService(CHOICE, CachedTranslator(backend))
    service.prefetch("Hello.", pool)
    time.sleep(0.15)
    started = time.monotonic()
    service.translate("Hello.")
    assert time.monotonic() - started < 0.01
    assert backend.calls == ["Hello."]


def test_duplicate_prefetch_sends_one_request(pool):
    backend = SlowBackend(delay=0.1)
    service = TranslationService(CHOICE, CachedTranslator(backend))
    service.prefetch("Same line.", pool)
    service.prefetch("Same line.", pool)
    service.translate("Same line.")
    assert backend.calls == ["Same line."]


def test_failed_prefetch_is_retried_by_confirmed_translation(pool):
    backend = SlowBackend(delay=0.01, fail_first=True)
    service = TranslationService(CHOICE, CachedTranslator(backend))
    service.prefetch("Line.", pool)
    time.sleep(0.05)
    assert service.translate("Line.") == "HU(Line.)"
    assert len(backend.calls) == 2
