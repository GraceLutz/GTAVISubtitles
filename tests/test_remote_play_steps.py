"""A Remote Play mód lépései ablak nélkül: melyik úton megy az olvasás, és mit kap a kirajzolás."""

import codecs
import threading
import time
from dataclasses import replace

import pytest
from band_drawing import AREA, DIALOGUE_BASELINE, FRAME, band_with

from gta_felirat import remote_play
from gta_felirat.app_base import SubtitleAppBase
from gta_felirat.band_feed import SOURCE_NONE, CapturedBand
from gta_felirat.config import AppConfig
from gta_felirat.ocr import Box, OcrLine, OcrResult
from gta_felirat.translator import CachedTranslator
from gta_felirat.window_capture import ScreenRect

TROUBLE = "We ain't lookin' for trouble, okay?"
LINE_BOX = Box(700, 957, 1220, 994)
WAIT_SECONDS = 5  # felső határ a munkaszálakra várva; a tesztek ezredmásodpercek alatt végeznek
NARROW_WIDTH = 1280  # ekkora képen a soralapú út nem megbízható


def hu(text):
    """Hamis fordítás, amely nem hasonlít az angolra."""
    return "HU<" + codecs.encode(text, "rot13") + ">"


class QuickBackend:
    def translate(self, text):
        return hu(text)


class FakeReader:
    """Hamis OCR: a teljes olvasás és a sorolvasás is ugyanazt a mondatot adja; számolja, melyiket kérték."""

    def __init__(self, row_error=None):
        self.row_error = row_error
        self.full_reads = 0
        self.row_reads = 0

    def read_band(self, band, area):
        self.full_reads += 1
        return OcrResult(TROUBLE, LINE_BOX, (OcrLine(TROUBLE, LINE_BOX),))

    def read_rows(self, band, area, boxes):
        self.row_reads += 1
        if self.row_error is not None:
            raise self.row_error
        return [(TROUBLE, 0.99) for _box in boxes]


class FakeFeed:
    """A folyamatos rögzítés helyett: a teszt adja be a sávokat, mintha a rögzítő szál hívná az on_band-et."""

    def __init__(self, hwnd, area_of, *, on_band=None):
        self.on_band = on_band
        self.band = None
        self.source = SOURCE_NONE
        self.is_started = False
        self.is_stopped = False

    def start(self):
        self.is_started = True

    def stop(self):
        self.is_stopped = True

    def latest(self):
        return self.band

    def deliver(self, image, at, frame_size=FRAME):
        sequence = 1 if self.band is None else self.band.sequence + 1
        self.band = CapturedBand(image, AREA, frame_size, at, sequence, "test")
        self.source = "test"
        self.on_band(self.band)


class FakeOverlay:
    def __init__(self, root, layer, visible_in_recordings):
        self.needs_redraw = True
        self.hwnd = 2
        self.renders = []
        self.is_destroyed = False

    def place(self, rect):
        pass

    def hide(self):
        pass

    def keep_on_top(self):
        pass

    def set_capture_visible(self, visible):
        pass

    def render(self, shown, hide_original, preview_region, backdrop):
        self.renders.append((shown, hide_original, backdrop))
        self.needs_redraw = False

    def destroy(self):
        self.is_destroyed = True


class FakeScreenGrabber:
    def close(self):
        pass


class FakeSettings:
    def __init__(self):
        self.root = object()
        self.is_closed = False

    def refresh_status(self):
        pass

    def sync(self, config):
        pass

    def close(self):
        self.is_closed = True


class Bench:
    """Az alkalmazás hamis ablakokkal; a lépéseit a teszt hívja, szimulált idővel."""

    def __init__(self, monkeypatch, reader, config):
        self.width = FRAME[0]
        self.reader = reader
        monkeypatch.setattr(remote_play, "find_window", lambda title, **_options: 1)
        monkeypatch.setattr(remote_play, "client_rect", lambda hwnd: ScreenRect(0, 0, self.width, FRAME[1]))
        monkeypatch.setattr(remote_play, "window_exists", lambda hwnd: True)
        monkeypatch.setattr(remote_play, "ScreenGrabber", FakeScreenGrabber)
        monkeypatch.setattr(remote_play, "WindowContentGrabber", lambda hwnd: None)
        monkeypatch.setattr(remote_play, "BandFeed", FakeFeed)
        monkeypatch.setattr(remote_play, "OverlayWindow", FakeOverlay)
        monkeypatch.setattr(remote_play, "tk_scheduler", lambda root, callback: lambda: None)
        monkeypatch.setattr(SubtitleAppBase, "_create_settings_window", lambda self, **extra: FakeSettings())
        self.app = remote_play.RemotePlayApp(replace(AppConfig(only_spoken_lines=False), **config), reader,
                                             translator=CachedTranslator(QuickBackend()))
        self.feed, self.overlay = self.app._feed, self.app._overlay
        self.now = 10.0

    def show(self, image):
        """Egy új sáv érkezik a rögzítő száltól, majd a megjelenítési szál lép egyet."""
        self.now += 0.04
        self.feed.deliver(image, self.now, (self.width, FRAME[1]))
        self.app._step(self.now + 0.001)

    def step_until(self, is_done):
        """A megjelenítési szál lépéseit ismétli (valós időben várva a munkaszálakra), amíg a feltétel teljesül."""
        deadline = time.monotonic() + WAIT_SECONDS
        while time.monotonic() < deadline:
            self.now += 0.004
            self.app._step(self.now)
            if is_done():
                return True
            time.sleep(0.005)
        return False

    def play_until(self, is_done):
        """Folyamatosan érkeznek az új sávok ugyanazzal a felirattal (mint élőben), amíg a feltétel teljesül."""
        deadline = time.monotonic() + WAIT_SECONDS
        seed = 100
        while time.monotonic() < deadline:
            seed += 1
            self.show(subtitle_band(seed))
            if is_done():
                return True
            time.sleep(0.005)
        return False

    def shown(self):
        return self.overlay.renders[-1][0] if self.overlay.renders else None

    def has_translation(self):
        shown = self.shown()
        return shown is not None and any(line.text == hu(TROUBLE) for line in shown.lines)


@pytest.fixture
def make_bench(monkeypatch):
    benches = []

    def make(reader=None, **config):
        benches.append(Bench(monkeypatch, reader or FakeReader(), config))
        return benches[-1]

    yield make
    for bench in benches:
        bench.app.stop()
        bench.app._pipeline.shutdown()


def subtitle_band(seed=1):
    return band_with((TROUBLE, DIALOGUE_BASELINE), seed=seed)


def test_wide_picture_from_the_feed_is_read_row_by_row(make_bench):
    bench = make_bench()

    for seed in range(4):  # ugyanaz a sor több rögzítésen: a képpontjai stabilak
        bench.show(subtitle_band(seed))
    translated = bench.step_until(bench.has_translation)

    assert translated
    assert bench.reader.row_reads >= 1
    assert bench.reader.full_reads <= 1  # legfeljebb az indulás pillanatában, mielőtt a soralapú út átvette


def test_new_row_is_covered_before_it_is_read_and_the_cover_uses_the_band_it_was_found_on(make_bench):
    bench = make_bench(FakeReader(row_error=None))
    bench.show(band_with())  # üres kép: a soralapú út bekapcsol
    band = subtitle_band()

    bench.show(band)

    shown, hide_original, backdrop = bench.overlay.renders[-1]
    assert hide_original
    assert len(shown.hidden) == 1 and shown.lines == ()
    assert backdrop.image is band


def test_still_picture_is_read_in_full_only_now_and_then(make_bench):
    # Álló képnél a folyamatos rögzítés nem ad új sávot: a teljes olvasás csak biztosíték, ne égesse a processzort
    bench = make_bench()
    bench.show(subtitle_band())
    bench.feed.band = None  # a rögzítés szerint a sáv elavult: áll a kép

    for _ in range(50):  # 50 lépés, 4 ms-onként: 0,2 mp
        bench.now += 0.004
        bench.app._step(bench.now)
        time.sleep(0.002)

    assert bench.reader.full_reads <= 2


def test_narrow_picture_is_read_in_full(make_bench):
    bench = make_bench()
    bench.width = NARROW_WIDTH

    translated = bench.play_until(bench.has_translation)

    assert translated
    assert bench.reader.row_reads == 0
    assert bench.reader.full_reads >= 2  # a teljes olvasású úton két egyező olvasás erősíti meg a feliratot


def test_row_reading_can_be_switched_off_in_the_config(make_bench):
    bench = make_bench(row_reading=False)

    translated = bench.play_until(bench.has_translation)

    assert translated
    assert bench.reader.row_reads == 0


def test_failing_row_reader_falls_back_to_reading_in_full(make_bench):
    bench = make_bench(FakeReader(row_error=RuntimeError("no model")))
    bench.show(band_with())

    translated = bench.play_until(bench.has_translation)

    assert translated
    assert bench.reader.row_reads >= 1  # megpróbálta soronként, mielőtt feladta
    assert bench.reader.full_reads >= 2


def test_translation_survives_the_switch_from_rows_to_full_reading(make_bench):
    bench = make_bench()
    for seed in range(4):
        bench.show(subtitle_band(seed))
    assert bench.step_until(bench.has_translation)

    bench.width = NARROW_WIDTH  # az ablakot összehúzták: innen a teljes olvasás megy
    still_translated = bench.play_until(lambda: bench.reader.full_reads >= 2 and bench.has_translation())

    assert still_translated


def test_feed_is_started_only_when_fast_capture_is_on(make_bench):
    class Loop:
        def after(self, delay, callback):
            pass

        def mainloop(self):
            pass

    for fast_capture in (True, False):
        bench = make_bench(fast_capture=fast_capture)
        bench.app._settings.root = Loop()

        bench.app.run()

        assert bench.feed.is_started is fast_capture
        assert bench.feed.is_stopped


def test_stopping_ends_the_row_threads_and_closes_the_windows(make_bench):
    before = set(threading.enumerate())
    bench = make_bench()
    started = [thread for thread in threading.enumerate()
               if thread not in before and thread.name in ("row-recognizer", "tk-waker")]

    bench.app.stop()
    for thread in started:
        thread.join(WAIT_SECONDS)

    assert started and not any(thread.is_alive() for thread in started)
    assert bench.overlay.is_destroyed and bench.app._settings.is_closed

