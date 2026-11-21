import codecs
import threading
import time

import pytest
from band_drawing import AREA, DIALOGUE_BASELINE, FRAME, band_with

from gta_felirat.band_feed import CapturedBand
from gta_felirat.config import AppConfig
from gta_felirat.ocr import EMPTY_RESULT
from gta_felirat.row_path import RowPath, WakingRowPipeline
from gta_felirat.row_pipeline import RowPipeline
from gta_felirat.translation_service import TranslationService
from gta_felirat.translator import CachedTranslator, TranslatorChoice

TROUBLE = "We ain't lookin' for trouble, okay?"
WAIT_SECONDS = 5  # felső határ a munkaszálakra várva; a tesztek ezredmásodpercek alatt végeznek
STEP_SECONDS = 0.04  # két rögzítés között eltelő idő


def hu(text):
    """Hamis fordítás, amely nem hasonlít az angolra."""
    return "HU<" + codecs.encode(text, "rot13") + ">"


class QuickBackend:
    def translate(self, text):
        return hu(text)


class RowReader:
    """Hamis felismerő: minden kért sorra ugyanazt a szöveget adja, és megjegyzi, hány sort kértek tőle."""

    def __init__(self, text=TROUBLE, error=None):
        self.text = text
        self.error = error
        self.requests = []

    def read_rows(self, band, area, boxes):
        self.requests.append(list(boxes))
        if self.error is not None:
            raise self.error
        return [(self.text, 0.99) for _box in boxes]


class Scene:
    """A soralapú út a tesztben: valódi pipeline, sorfigyelő és munkaszálak; hamis felismerő és fordító."""

    def __init__(self, reader):
        self.reader = reader
        self.woken = threading.Event()
        service = TranslationService(TranslatorChoice("deepl", "hu", "", ""), CachedTranslator(QuickBackend()))
        self.pipeline = RowPipeline(AppConfig(only_spoken_lines=False), None, service)
        self.path = RowPath(self.pipeline, reader.read_rows, AppConfig().min_text_score, self.woken.set)
        self.captures = 0

    def capture(self, band, at):
        self.captures += 1
        captured = CapturedBand(band, AREA, FRAME, at, self.captures, "test")
        self.path.on_band(captured)
        return captured

    def pump_until(self, is_done, now):
        """A megjelenítési szál lépéseit ismétli (valós időben várva a munkaszálakra), amíg a feltétel teljesül."""
        deadline = time.monotonic() + WAIT_SECONDS
        while time.monotonic() < deadline:
            self.path.pump(now, FRAME)
            if is_done():
                return True
            time.sleep(0.005)
        return False

    def close(self):
        self.path.close()
        self.pipeline.shutdown()


@pytest.fixture
def make_scene():
    scenes = []

    def make(reader=None):
        scenes.append(Scene(reader or RowReader()))
        return scenes[-1]

    yield make
    for scene in scenes:
        scene.close()


def test_new_row_is_hidden_as_soon_as_its_band_is_handed_in(make_scene):
    scene = make_scene()

    scene.capture(band_with((TROUBLE, DIALOGUE_BASELINE)), at=10.0)
    changed = scene.path.pump(10.01, FRAME)

    assert changed
    assert len(scene.pipeline.shown.hidden) == 1  # még olvasatlan, de már takarjuk
    assert scene.pipeline.shown.lines == ()


def test_row_is_read_once_and_translated(make_scene):
    scene = make_scene()
    band = band_with((TROUBLE, DIALOGUE_BASELINE))

    for step in range(3):  # ugyanaz a sor három rögzítésen: a képpontjai stabilak, nem kell újraolvasni
        scene.capture(band.copy(), at=10.0 + step * STEP_SECONDS)
    translated = scene.pump_until(lambda: scene.pipeline.shown is not None and scene.pipeline.shown.lines, now=10.2)

    assert translated
    assert scene.pipeline.shown.lines[0].text == hu(TROUBLE)
    assert [len(boxes) for boxes in scene.reader.requests] == [1]


def test_display_thread_is_woken_when_a_row_appears(make_scene):
    scene = make_scene()

    scene.capture(band_with((TROUBLE, DIALOGUE_BASELINE)), at=10.0)

    assert scene.woken.wait(WAIT_SECONDS)


def test_backdrop_is_the_band_the_shown_state_was_made_from(make_scene):
    scene = make_scene()
    assert scene.path.backdrop() is None  # még nem jött sáv

    captured = scene.capture(band_with((TROUBLE, DIALOGUE_BASELINE)), at=10.0)
    scene.path.pump(10.01, FRAME)

    backdrop = scene.path.backdrop()
    assert backdrop.image is captured.image
    assert backdrop.area == AREA
    assert scene.path.backdrop() is backdrop  # ugyanaz a háttér: a takarás gyorstára újra felhasználhatja


def test_failing_reader_switches_the_row_path_off_for_good(make_scene):
    scene = make_scene(RowReader(error=RuntimeError("no model")))
    assert scene.path.can_read(FRAME[0])

    scene.capture(band_with((TROUBLE, DIALOGUE_BASELINE)), at=10.0)
    gave_up = scene.pump_until(lambda: not scene.path.can_read(FRAME[0]), now=10.05)

    assert gave_up
    assert scene.pipeline.snapshot is None  # a pipeline a régi, teljes olvasású úton folytathatja


def test_narrow_picture_is_not_read_by_rows(make_scene):
    assert not make_scene().path.can_read(1280)


def test_suspended_path_forgets_its_rows_and_the_waiting_snapshot(make_scene):
    scene = make_scene()
    scene.capture(band_with((TROUBLE, DIALOGUE_BASELINE)), at=10.0)
    scene.path.pump(10.01, FRAME)
    scene.capture(band_with((TROUBLE, DIALOGUE_BASELINE)), at=10.04)  # ez még a postaládában vár

    scene.path.suspend()

    assert scene.pipeline.snapshot is None
    assert scene.path.backdrop() is None
    assert not scene.path.pump(10.05, FRAME)  # a felfüggesztés előtti pillanatkép nem éled újra
    assert scene.pipeline.snapshot is None


def test_display_thread_is_woken_when_a_full_read_is_done():
    class FullReader:
        def read_band(self, band, area):
            return EMPTY_RESULT

    woken = threading.Event()
    service = TranslationService(TranslatorChoice("deepl", "hu", "", ""), CachedTranslator(QuickBackend()))
    pipeline = WakingRowPipeline(AppConfig(), FullReader(), service, wake=woken.set)

    pipeline.submit_band(lambda: band_with(), AREA, FRAME, 10.0)

    assert woken.wait(WAIT_SECONDS)
    pipeline.shutdown()


def test_closing_ends_the_worker_threads(make_scene):
    before = {thread.name for thread in threading.enumerate()}
    scene = make_scene()
    started = [thread for thread in threading.enumerate()
               if thread.name in ("row-recognizer", "tk-waker") and thread.name not in before]

    scene.path.close()
    for thread in started:
        thread.join(WAIT_SECONDS)

    assert started and not any(thread.is_alive() for thread in started)
