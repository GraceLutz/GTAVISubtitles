import numpy as np

from gta_felirat.band_capture import (BLACK_CHECKS_BEFORE_FALLBACK, BLACK_FRAMES_BEFORE_FALLBACK,
                                      FAILED_GRABS_BEFORE_FALLBACK, BandCapture)
from gta_felirat.ocr import Box
from gta_felirat.window_capture import ScreenRect

AREA = Box(0, 0, 40, 10)
RECT = ScreenRect(100, 200, 40, 10)
# Ennyi fekete sáv után dönt úgy a rögzítő, hogy az ablak tartalma nem olvasható (ha a képernyő a játékot mutatja)
LONG_BLACK = BLACK_FRAMES_BEFORE_FALLBACK * BLACK_CHECKS_BEFORE_FALLBACK


class FakeWindowGrabber:
    def __init__(self, images):
        self._images = list(images)
        self.calls = 0

    def grab(self, area):
        self.calls += 1
        return self._images.pop(0) if len(self._images) > 1 else self._images[0]


class FakeScreenGrabber:
    """A képernyő: egyenletes fényességű sáv; `values` megadásával hívásonként más (az utolsó ismétlődik)."""

    def __init__(self, value, values=None):
        self._values = list(values or [value])
        self.calls = 0

    def grab(self, rect):
        self.calls += 1
        value = self._values.pop(0) if len(self._values) > 1 else self._values[0]
        return np.full((rect.height, rect.width, 3), value, np.uint8)


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


def test_falls_back_when_window_stays_black_but_screen_keeps_showing_the_game():
    capture = BandCapture(FakeWindowGrabber([band(0)]), FakeScreenGrabber(80))

    for _ in range(LONG_BLACK - 1):
        capture.grab(RECT, AREA)
    assert capture.uses_window_content  # egyetlen összevetés még nem elég

    image = capture.grab(RECT, AREA)

    assert not capture.uses_window_content
    assert image[0, 0, 0] == 80  # innentől a képernyőről jön a kép


def test_dark_scene_does_not_cause_fallback():
    capture = BandCapture(FakeWindowGrabber([band(0)]), FakeScreenGrabber(0))
    for _ in range(LONG_BLACK * 2):
        capture.grab(RECT, AREA)
    assert capture.uses_window_content


def test_picture_returning_at_the_moment_of_one_check_is_not_a_broken_capture():
    # Valós eset: küldetés végi sötét képernyő; az egyik összevetés pillanatában a képernyőn már volt valami
    # világos, a program a képernyőre váltott, és ezzel a takarás az egész munkamenetre kikapcsolt
    screen = FakeScreenGrabber(0, values=[80, 0, 80, 0, 80, 0])
    capture = BandCapture(FakeWindowGrabber([band(0)]), screen)

    for _ in range(LONG_BLACK * 2):
        capture.grab(RECT, AREA)

    assert capture.uses_window_content


def test_a_few_bright_pixels_on_the_screen_are_not_the_game():
    class NotificationOnBlack:
        def grab(self, rect):
            screen = np.zeros((rect.height, rect.width, 3), np.uint8)
            screen[:2, :6] = 255  # a sáv 3%-a: értesítés sarka, egy felirat betűi
            return screen

    capture = BandCapture(FakeWindowGrabber([band(0)]), NotificationOnBlack())

    for _ in range(LONG_BLACK * 2):
        capture.grab(RECT, AREA)

    assert capture.uses_window_content


def test_screen_that_cannot_be_judged_never_causes_the_fallback():
    # Más ablak takarja a sávot, vagy a saját feliratrétegünk rajzol rá: a képernyő nem bizonyít semmit
    capture = BandCapture(FakeWindowGrabber([band(0)]), FakeScreenGrabber(80), can_judge_screen=lambda rect: False)

    for _ in range(LONG_BLACK * 2):
        capture.grab(RECT, AREA)

    assert capture.uses_window_content


def test_window_that_turns_black_for_good_after_a_real_picture_still_falls_back():
    window = FakeWindowGrabber([band(120)] + [band(0)])
    capture = BandCapture(window, FakeScreenGrabber(80))

    for _ in range(LONG_BLACK + 1):
        capture.grab(RECT, AREA)

    assert not capture.uses_window_content


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
