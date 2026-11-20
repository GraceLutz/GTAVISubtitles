import time

from gta_felirat.config import AppConfig
from gta_felirat.ocr import Box, OcrResult
from gta_felirat.subtitle_pipeline import SubtitlePipeline
from gta_felirat.translation_service import TranslationService
from gta_felirat.translator import CachedTranslator, TranslatorChoice
from gta_felirat.window_capture import ScreenRect, band_sample_points, is_unobstructed

REMOTE_PLAY = 100
OVERLAY = 200
BROWSER = 300
BAND = ScreenRect(left=500, top=800, width=900, height=120)


def windows_at(covered_by=None, covered_x_from=None):
    """Hamis Windows-lekérdezés: a megadott x-től jobbra egy másik ablak takar."""
    def root_at(x, y):
        if covered_by is not None and x >= covered_x_from:
            return covered_by
        return REMOTE_PLAY
    return root_at


def test_sample_points_lie_inside_the_band():
    points = band_sample_points(BAND)

    assert len(points) >= 5
    for x, y in points:
        assert BAND.left <= x < BAND.left + BAND.width
        assert BAND.top <= y < BAND.top + BAND.height


def test_band_is_readable_when_only_remote_play_is_there():
    assert is_unobstructed(BAND, {REMOTE_PLAY, OVERLAY}, root_at=windows_at())


def test_our_own_overlay_on_the_band_does_not_count_as_covering():
    assert is_unobstructed(BAND, {REMOTE_PLAY, OVERLAY}, root_at=windows_at(OVERLAY, BAND.left))


def test_band_partly_covered_by_a_browser_is_not_read():
    covered_from = BAND.left + BAND.width - 50

    assert not is_unobstructed(BAND, {REMOTE_PLAY, OVERLAY}, root_at=windows_at(BROWSER, covered_from))


# --- eltakart sáv a pipeline-ban ----------------------------------------------

class TextReader:
    def read_band(self, band, area):
        return OcrResult(text="Jason: We go now.", box=Box(area.left, area.top, area.left + 100, area.top + 30))


class EchoBackend:
    def translate(self, text):
        return f"HU({text})"


def wait_until(condition, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_hidden_band_counts_as_no_subtitle_so_the_old_one_disappears():
    config = AppConfig()
    choice = TranslatorChoice("auto", "hu", "claude-haiku-5-5", "gpt-4.1-mini")
    pipeline = SubtitlePipeline(config, TextReader(), TranslationService(choice, CachedTranslator(EchoBackend())))
    area = pipeline.band_area(1920, 1080)
    try:
        for step in range(2):
            now = step * 1.0
            pipeline.submit_band(lambda: None, area, (1920, 1080), now)
            assert wait_until(lambda: (pipeline.poll(now) or True) and pipeline._ocr_future is None)
        assert wait_until(lambda: (pipeline.poll(2.0) or True) and pipeline.shown is not None)

        changed = False
        for step in range(2):
            changed = pipeline.report_hidden(2.0 + step * (config.hold_seconds + 1)) or changed
    finally:
        pipeline.shutdown()

    assert changed
    assert pipeline.shown is None
