import time

import numpy as np
import pytest

from gta_felirat import band_capture
from gta_felirat.band_capture import (FAILED_GRABS_BEFORE_FALLBACK, SINGLE_THREAD_PIXELS, PrintWindowSource,
                                      WindowContentGrabber, WindowGrabError, bgrx_to_bgr, cut_band, fits)
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


# --- az ablak saját tartalmának rögzítése (PrintWindow): ezt a fölé húzott ablakok nem zavarják ---

MISSING_WINDOW = 0  # ilyen ablak nincs: a valódi rögzítés rajta mindig sikertelen
CLIENT_SIZE = (1920, 1080)
GRAB_PERIOD = 1 / 15
SUBTITLE_AREA = Box(230, 842, 1689, 1058)


def subtitle_area_of(_width, _height):
    return SUBTITLE_AREA


def bgrx_surface(width, height):
    """Rajzfelület: a kék csatorna az oszlop, a zöld a sor száma, a piros 200, a negyedik bájt kitöltés."""
    pixels = np.zeros((height, width, 4), np.uint8)
    pixels[:, :, 0] = np.arange(width)[None, :]
    pixels[:, :, 1] = np.arange(height)[:, None]
    pixels[:, :, 2:] = (200, 77)
    return pixels


def test_picture_converted_in_several_stripes_equals_dropping_the_fourth_channel():
    pixels = np.random.default_rng(3).integers(0, 255, (500, 300, 4), dtype=np.uint8)
    view = pixels[10:490, 20:280]  # nem összefüggő nézet, mint a rajzfelületből kivágott sáv
    assert view.shape[0] * view.shape[1] > 2 * SINGLE_THREAD_PIXELS

    image = bgrx_to_bgr(view)

    assert np.array_equal(image, view[:, :, :3])
    assert image.flags.c_contiguous and image.flags.owndata


def test_band_inside_the_surface_is_cut_out_as_bgr():
    band = cut_band(bgrx_surface(100, 60), Box(10, 20, 40, 50))

    assert band.shape == (30, 30, 3)
    assert band[0, 0].tolist() == [10, 20, 200]
    assert band[-1, -1].tolist() == [39, 49, 200]


def test_part_of_the_band_outside_the_surface_is_black():
    band = cut_band(bgrx_surface(100, 60), Box(90, 50, 120, 70))

    assert band.shape == (20, 30, 3)
    assert band[9, 9].tolist() == [99, 59, 200]
    assert band[10:].max() == 0 and band[:, 10:].max() == 0


def test_band_starting_before_the_surface_keeps_its_size_and_place():
    band = cut_band(bgrx_surface(100, 60), Box(-5, -3, 10, 7))

    assert band.shape == (10, 15, 3)
    assert band[3, 5].tolist() == [0, 0, 200]
    assert band[:3].max() == 0 and band[:, :5].max() == 0


def test_cut_band_does_not_share_memory_with_the_surface():
    surface = bgrx_surface(100, 60)
    band = cut_band(surface, Box(10, 20, 40, 50))

    surface[:] = 0

    assert band[0, 0].tolist() == [10, 20, 200]


@pytest.mark.parametrize("area, expected", [
    (Box(0, 0, 1920, 1080), True),
    (Box(230, 842, 1689, 1058), True),
    (Box(230, 842, 1921, 1058), False),  # jobbra kilóg
    (Box(230, 842, 1689, 1081), False),  # lent kilóg
    (Box(-1, 842, 1689, 1058), False),
    (Box(500, 842, 500, 1058), False),  # üres
])
def test_area_fits_only_when_it_is_a_real_box_inside_the_client_area(area, expected):
    assert fits(area, *CLIENT_SIZE) is expected


def test_grabbing_a_window_that_does_not_exist_gives_nothing_and_closing_twice_is_harmless():
    grabber = WindowContentGrabber(MISSING_WINDOW)

    assert grabber.grab(SUBTITLE_AREA) is None
    grabber.close()
    grabber.close()


def test_printwindow_source_treats_a_minimised_window_as_no_new_picture():
    source = PrintWindowSource(MISSING_WINDOW, GRAB_PERIOD)  # a kliensterület lekérdezése itt is None-t ad

    reads = [source.read(subtitle_area_of) for _ in range(FAILED_GRABS_BEFORE_FALLBACK + 2)]

    assert reads == [None] * (FAILED_GRABS_BEFORE_FALLBACK + 2)


def test_printwindow_source_gives_up_after_repeated_failed_grabs(monkeypatch):
    monkeypatch.setattr(band_capture, "client_rect", lambda _hwnd: ScreenRect(0, 0, *CLIENT_SIZE))
    source = PrintWindowSource(MISSING_WINDOW, GRAB_PERIOD)

    for _ in range(FAILED_GRABS_BEFORE_FALLBACK - 1):
        assert source.read(subtitle_area_of) is None
    with pytest.raises(WindowGrabError):
        source.read(subtitle_area_of)


def test_printwindow_source_skips_an_area_that_does_not_fit_the_window(monkeypatch):
    monkeypatch.setattr(band_capture, "client_rect", lambda _hwnd: ScreenRect(0, 0, 800, 600))
    source = PrintWindowSource(MISSING_WINDOW, GRAB_PERIOD)

    reads = [source.read(subtitle_area_of) for _ in range(FAILED_GRABS_BEFORE_FALLBACK + 2)]

    assert reads == [None] * (FAILED_GRABS_BEFORE_FALLBACK + 2)
