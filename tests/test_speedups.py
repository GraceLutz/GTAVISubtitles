import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from gta_felirat import backends, translation_service, translation_store
from gta_felirat.backends import DeepLBackend
from gta_felirat.config import AppConfig
from gta_felirat.i18n import t
from gta_felirat.languages import find_language
from gta_felirat.ocr import Box, OcrResult
from gta_felirat.subtitle_pipeline import SubtitlePipeline
from gta_felirat.translation_service import TranslationService
from gta_felirat.translation_store import TranslationStore
from gta_felirat.translator import CachedTranslator, TranslatorChoice

CHOICE = TranslatorChoice("auto", "hu", "claude-haiku-5-5", "gpt-4.1-mini")


class CountingBackend:
    def __init__(self):
        self.calls = []
        self.warm_ups = 0

    def translate(self, text):
        self.calls.append(text)
        return f"HU({text})"

    def warm_up(self):
        self.warm_ups += 1


def wait_until(condition, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


# --- mentett fordítások -------------------------------------------------------

def test_saved_translation_is_reused_after_restart_without_calling_the_service(tmp_path):
    path = tmp_path / "mentett.json"
    first = CountingBackend()
    CachedTranslator(first, store=TranslationStore(path), namespace="deepl:hu").translate("Go to hell!")

    second = CountingBackend()
    restarted = CachedTranslator(second, store=TranslationStore(path), namespace="deepl:hu")

    assert restarted.translate("Local: Go to hell!") == "Local: HU(Go to hell!)"
    assert second.calls == []


def test_saved_translations_are_kept_apart_per_service_and_language(tmp_path):
    store = TranslationStore(tmp_path / "mentett.json")
    CachedTranslator(CountingBackend(), store=store, namespace="deepl:hu").translate("Hello")
    german = CountingBackend()

    CachedTranslator(german, store=store, namespace="deepl:de").translate("Hello")

    assert german.calls == ["Hello"]


def test_failed_translation_is_not_saved(tmp_path):
    class FailingBackend:
        def translate(self, text):
            raise ConnectionError("offline")

    store = TranslationStore(tmp_path / "mentett.json")
    with pytest.raises(Exception):
        CachedTranslator(FailingBackend(), store=store, namespace="deepl:hu").translate("Hello")

    assert store.get("deepl:hu", "hello") is None


def test_corrupt_store_file_starts_empty_instead_of_crashing(tmp_path):
    path = tmp_path / "mentett.json"
    path.write_text("{ ez nem json", encoding="utf-8")

    assert TranslationStore(path).get("deepl:hu", "hello") is None


def test_store_drops_oldest_entries_over_the_limit(tmp_path):
    store = TranslationStore(tmp_path / "mentett.json", max_entries=2)
    for word in ("a", "b", "c"):
        store.put("deepl:hu", word, word.upper())

    reloaded = TranslationStore(tmp_path / "mentett.json", max_entries=2)
    assert [reloaded.get("deepl:hu", w) for w in ("a", "b", "c")] == [None, "B", "C"]


def test_saved_file_keeps_its_format_when_several_lines_are_saved_at_once(tmp_path):
    path = tmp_path / "mentett.json"

    TranslationStore(path).put_many("deepl:hu", {"go!": "Gyerünk!", "wait.": "Várj."})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "version": 1, "entries": {"deepl:hu": {"go!": "Gyerünk!", "wait.": "Várj."}}}


def test_saved_translation_can_be_read_while_the_file_is_still_being_written(tmp_path, monkeypatch):
    store = TranslationStore(tmp_path / "mentett.json")
    writing, may_finish = threading.Event(), threading.Event()

    def slow_replace(_source, _target):
        writing.set()
        may_finish.wait(timeout=2)

    monkeypatch.setattr(translation_store.os, "replace", slow_replace)
    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(store.put_many, "deepl:hu", {"go!": "Gyerünk!"})
        assert writing.wait(timeout=2)
        try:
            read = pool.submit(store.get, "deepl:hu", "go!").result(timeout=1)
        finally:
            may_finish.set()
        writer.result(timeout=2)

    assert read == "Gyerünk!"


def test_translations_saved_from_several_threads_all_reach_the_file(tmp_path):
    path = tmp_path / "mentett.json"
    store = TranslationStore(path)
    batches = [{f"line {worker}-{number}": "x" for number in range(20)} for worker in range(4)]

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda batch: store.put_many("deepl:hu", batch), batches))

    reloaded = TranslationStore(path)
    assert all(reloaded.get("deepl:hu", key) == "x" for batch in batches for key in batch)


# --- kapcsolat melegen tartása ------------------------------------------------

def test_idle_connection_is_warmed_once_in_the_background():
    backend = CountingBackend()
    service = TranslationService(CHOICE, CachedTranslator(backend))

    service.keep_warm()
    assert wait_until(lambda: backend.warm_ups == 1)
    service.keep_warm()
    time.sleep(0.05)

    assert backend.warm_ups == 1


def test_recent_translation_means_no_extra_warm_up(monkeypatch):
    backend = CountingBackend()
    service = TranslationService(CHOICE, CachedTranslator(backend))
    service.translate("Hello")

    service.keep_warm()
    time.sleep(0.05)

    assert backend.warm_ups == 0


def test_warm_up_again_after_long_silence(monkeypatch):
    backend = CountingBackend()
    translator = CachedTranslator(backend)
    service = TranslationService(CHOICE, translator)
    translator.last_network_at = time.monotonic() - translation_service.KEEP_WARM_SECONDS - 1

    service.keep_warm()

    assert wait_until(lambda: backend.warm_ups == 1)


def test_deepl_warm_up_is_a_free_usage_request(monkeypatch):
    calls = []

    class Ok:
        status_code = 200

    def get(url, headers, timeout):
        calls.append((url, headers))
        return Ok()

    monkeypatch.setattr(backends._session, "get", get)
    DeepLBackend("abc:fx", find_language("hu")).warm_up()

    assert calls == [(backends.DEEPL_FREE_USAGE_URL, {"Authorization": "DeepL-Auth-Key abc:fx"})]


# --- csak a feliratsáv rögzítése ----------------------------------------------

class BandReader:
    def __init__(self):
        self.calls = []

    def read_band(self, band, area):
        self.calls.append((band.shape, area))
        return OcrResult(text="Jason: Go.", box=Box(area.left + 10, area.top + 5, area.left + 200, area.top + 40))


def test_pipeline_reads_only_the_subtitle_band_and_keeps_full_frame_positions():
    reader = BandReader()
    config = AppConfig()
    service = TranslationService(CHOICE, CachedTranslator(CountingBackend()))
    pipeline = SubtitlePipeline(config, reader, service)
    area = pipeline.band_area(1920, 1080)
    band = np.zeros((area.height, area.width, 3), np.uint8)

    try:
        pipeline.submit_band(lambda: band, area, (1920, 1080), now=0.0)
        assert wait_until(lambda: (pipeline.poll(0.2) or True) and pipeline._ocr_future is None)
    finally:
        pipeline.shutdown()

    assert reader.calls == [((area.height, area.width, 3), area)]
    assert area.top == int(config.subtitle_region.top * 1080)


# --- 10 mp-es felső határ a fordításra ----------------------------------------

class TextReader:
    def read_band(self, band, area):
        return OcrResult(text="Jason: We go now.", box=Box(area.left, area.top, area.left + 100, area.top + 30))


def test_pipeline_gives_up_on_a_translation_after_the_deadline(monkeypatch):
    from gta_felirat import subtitle_pipeline

    class HangingBackend:
        def translate(self, text):
            time.sleep(0.5)
            return "késő"

    monkeypatch.setattr(subtitle_pipeline, "TRANSLATION_DEADLINE_SECONDS", 0.05)
    config = AppConfig()
    pipeline = SubtitlePipeline(config, TextReader(), TranslationService(CHOICE, CachedTranslator(HangingBackend())))
    area = pipeline.band_area(1920, 1080)
    band = np.zeros((area.height, area.width, 3), np.uint8)
    try:
        for step in range(2):  # két egyező olvasás kell a megerősítéshez
            now = step * 1.0
            pipeline.submit_band(lambda: band, area, (1920, 1080), now)
            assert wait_until(lambda: (pipeline.poll(now) or True) and pipeline._ocr_future is None)
        assert wait_until(lambda: (pipeline.poll(2.0) or True) and pipeline.shown is not None, timeout=0.4)
    finally:
        pipeline.shutdown()

    prefix = t("pipeline.timeout_prefix", seconds=subtitle_pipeline.TRANSLATION_DEADLINE_SECONDS)
    assert pipeline.shown.text.startswith(prefix)


def test_backend_skips_retry_when_the_time_budget_is_used_up(monkeypatch):
    import requests

    attempts = []

    def post(url, headers, json, timeout):
        attempts.append(timeout)
        raise requests.ConnectionError("lassú hálózat")

    monkeypatch.setattr(backends._session, "post", post)
    monkeypatch.setattr(backends.time, "sleep", lambda _s: None)
    # Megszakadt kapcsolatnál azonnal újrapróbálunk, de csak ha legalább MIN_RETRY_SECONDS maradt
    monkeypatch.setattr(backends, "TRANSLATION_BUDGET_SECONDS", 1.5)

    with pytest.raises(backends.BackendError):
        DeepLBackend("abc:fx", find_language("hu")).translate("Hello")

    assert len(attempts) == 1
    assert attempts[0] <= 1.5
