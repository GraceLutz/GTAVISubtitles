import numpy as np

from gta_felirat.band_capture import BLACK_FRAMES_BEFORE_FALLBACK, FAILED_GRABS_BEFORE_FALLBACK, BandCapture
from gta_felirat.ocr import Box
from gta_felirat.window_capture import ScreenRect

AREA = Box(0, 0, 40, 10)
RECT = ScreenRect(100, 200, 40, 10)


class FakeWindowGrabber:
    def __init__(self, images):
        self._images = list(images)
        self.calls = 0

    def grab(self, area):
        self.calls += 1
        return self._images.pop(0) if len(self._images) > 1 else self._images[0]


class FakeScreenGrabber:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def grab(self, rect):
        self.calls += 1
        return np.full((rect.height, rect.width, 3), self.value, np.uint8)


def band(value):
    return np.full((AREA.height, AREA.width, 3), value, np.uint8)


def test_window_content_is_used_when_it_works():
    capture = BandCapture(FakeWindowGrabber([band(120)]), FakeScreenGrabber(50))
    assert capture.grab(RECT, AREA)[0, 0, 0] == 120
    assert capture.uses_window_content


def test_falls_back_to_screen_when_printwindow_keeps_failing():
    screen = FakeScreenGrabber(50)
    capture = BandCapture(FakeWindowGrabber([None]), screen)
    for _ in range(FAILED_GRABS_BEFORE_FALLBACK):
        image = capture.grab(RECT, AREA)
    assert not capture.uses_window_content
    assert image[0, 0, 0] == 50


def test_falls_back_when_window_stays_black_but_screen_shows_the_game():
    capture = BandCapture(FakeWindowGrabber([band(0)]), FakeScreenGrabber(80))
    for _ in range(BLACK_FRAMES_BEFORE_FALLBACK):
        capture.grab(RECT, AREA)
    assert not capture.uses_window_content


def test_dark_scene_does_not_cause_fallback():
    capture = BandCapture(FakeWindowGrabber([band(0)]), FakeScreenGrabber(0))
    for _ in range(BLACK_FRAMES_BEFORE_FALLBACK * 2):
        capture.grab(RECT, AREA)
    assert capture.uses_window_content


def test_after_fallback_window_grabber_is_no_longer_called():
    window = FakeWindowGrabber([None])
    capture = BandCapture(window, FakeScreenGrabber(50))
    for _ in range(FAILED_GRABS_BEFORE_FALLBACK + 3):
        capture.grab(RECT, AREA)
    assert window.calls == FAILED_GRABS_BEFORE_FALLBACK


def test_reports_whether_the_last_band_came_from_the_window():
    capture = BandCapture(FakeWindowGrabber([band(120), None]), FakeScreenGrabber(50))
    capture.grab(RECT, AREA)
    assert capture.last_from_window
    capture.grab(RECT, AREA)
    assert not capture.last_from_window
