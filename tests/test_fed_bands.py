import numpy as np

from gta_felirat.band_feed import SOURCE_WGC, CapturedBand
from gta_felirat.ocr import Box
from gta_felirat.remote_play import fresh_band

AREA = Box(230, 842, 1689, 1058)
FRAME_SIZE = (1920, 1080)


def captured(sequence, frame_size=FRAME_SIZE):
    image = np.zeros((AREA.height, AREA.width, 3), np.uint8)
    return CapturedBand(image, AREA, frame_size, captured_at=10.0, sequence=sequence, source=SOURCE_WGC)


def test_new_band_is_handed_to_the_reader():
    band = captured(sequence=8)

    assert fresh_band(band, read_sequence=7, frame_size=FRAME_SIZE) is band


def test_band_that_was_already_read_is_not_read_again():
    assert fresh_band(captured(sequence=7), read_sequence=7, frame_size=FRAME_SIZE) is None


def test_band_of_another_window_size_waits_for_the_next_capture():
    # Átméretezés közben a sáv még a régi ablakméretre készült: a helye nem stimmelne
    assert fresh_band(captured(sequence=8, frame_size=(1280, 720)), read_sequence=7, frame_size=FRAME_SIZE) is None


def test_no_band_means_the_classic_capture_path():
    assert fresh_band(None, read_sequence=7, frame_size=FRAME_SIZE) is None
