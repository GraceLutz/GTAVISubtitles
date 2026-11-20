import numpy as np

from gta_felirat.config import AppConfig
from gta_felirat.ocr import Box
from gta_felirat.remote_play import effective_hide_original
from gta_felirat.window_capture import blank_out

BAND_AREA = Box(left=200, top=800, right=1700, bottom=1060)


def white_band():
    return np.full((BAND_AREA.height, BAND_AREA.width, 3), 255, np.uint8)


def test_our_own_translation_is_blanked_out_before_ocr():
    translation = Box(left=700, top=960, right=1200, bottom=1010)  # a Remote Play ablakon belül

    band = blank_out(white_band(), BAND_AREA, translation, margin=4)

    inside = band[960 - 800 + 10, 700 - 200 + 10]
    outside = band[10, 10]
    assert inside.tolist() == [0, 0, 0]
    assert outside.tolist() == [255, 255, 255]


def test_margin_also_covers_the_shadow_and_outline():
    translation = Box(left=700, top=960, right=1200, bottom=1010)

    band = blank_out(white_band(), BAND_AREA, translation, margin=4)

    assert band[1010 - 800 + 3, 1200 - 200 + 3].tolist() == [0, 0, 0]
    assert band[1010 - 800 + 6, 1200 - 200 + 6].tolist() == [255, 255, 255]


def test_band_is_untouched_when_the_translation_is_elsewhere():
    original = white_band()

    band = blank_out(original, BAND_AREA, Box(0, 0, 100, 100), margin=4)

    assert np.array_equal(band, original)


def test_original_capture_is_not_modified():
    original = white_band()

    blank_out(original, BAND_AREA, Box(700, 960, 1200, 1010), margin=4)

    assert original.min() == 255


def test_overlay_is_visible_to_recorders_by_default():
    assert AppConfig().visible_in_recordings


def test_english_is_never_covered_while_visible_in_recordings():
    # Ha a felvételen is látszik, az OCR a takarást is látná, és nem olvasna semmit
    assert not effective_hide_original(AppConfig(hide_original=True, visible_in_recordings=True))
    assert effective_hide_original(AppConfig(hide_original=True, visible_in_recordings=False))
