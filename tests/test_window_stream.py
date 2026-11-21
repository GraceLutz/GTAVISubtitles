from contextlib import contextmanager

import numpy as np
import pytest

from gta_felirat import band_feed
from gta_felirat.band_feed import EARLY_FRAME_SLACK_SECONDS, WGC_POLL_SECONDS, FramePacer, StreamSource, band_in_frame
from gta_felirat.ocr import Box
from gta_felirat.window_stream import WindowStream, WindowStreamError, mapped_rows_to_bgr

HWND = 4242  # hamis ablakazonosító: a hamis képfolyammal sosem jut el a Windowshoz
MISSING_WINDOW = 0  # ilyen ablak nincs
INTERVAL = 1 / 30
SOURCE_FRAME_GAP = 1 / 60  # a játék képe ilyen sűrűn frissül
ROW_PITCH = 32  # a videókártya ennyi bájtra tömi ki az 5 pixeles (20 bájtos) sort
PADDING = 0xEE
SCRATCH = np.zeros(4096, np.uint8)  # érvényes memória a hibás paraméterek próbájához


def mapped_texture(width, height, row_pitch=ROW_PITCH):
    """A leképezett textúra mása: a kék csatorna az oszlop, a zöld a sor száma; a sorok végén kitöltés áll.

    Pontosan akkora, amekkorát a valódi leképezés garantál: az utolsó sor után már nincs kitöltés.
    """
    memory = np.full(row_pitch * (height - 1) + width * 4, PADDING, np.uint8)
    for row in range(height):
        pixels = memory[row * row_pitch:row * row_pitch + width * 4].reshape(width, 4)
        pixels[:, 0], pixels[:, 1], pixels[:, 2], pixels[:, 3] = np.arange(width), row, 200, 255
    return memory


# --- a videókártyáról leképezett sorok ------------------------------------------

def test_padded_rows_become_a_bgr_image_without_the_padding():
    memory = mapped_texture(width=5, height=3)

    image = mapped_rows_to_bgr(memory.ctypes.data, ROW_PITCH, 5, 3)

    assert image.shape == (3, 5, 3) and image.dtype == np.uint8
    assert image[0, 0].tolist() == [0, 0, 200]
    assert image[2, 4].tolist() == [4, 2, 200]
    assert PADDING not in image


def test_rows_without_padding_are_read_too():
    memory = mapped_texture(width=8, height=2, row_pitch=32)

    image = mapped_rows_to_bgr(memory.ctypes.data, 32, 8, 2)

    assert image[1, 7].tolist() == [7, 1, 200]


def test_image_keeps_its_pixels_after_the_mapping_is_gone():
    memory = mapped_texture(width=5, height=3)
    image = mapped_rows_to_bgr(memory.ctypes.data, ROW_PITCH, 5, 3)

    memory[:] = 0

    assert image[2, 4].tolist() == [4, 2, 200]
    assert image.flags.c_contiguous


@pytest.mark.parametrize("address, row_pitch, width, height", [
    (None, ROW_PITCH, 5, 3),  # a leképezés nem adott memóriát
    (0, ROW_PITCH, 5, 3),
    (SCRATCH.ctypes.data, 16, 5, 3),  # egy sor (20 bájt) nem fér el 16 bájtban
    (SCRATCH.ctypes.data, ROW_PITCH, 0, 3),
    (SCRATCH.ctypes.data, ROW_PITCH, 5, 0),
])
def test_unusable_mapping_is_an_error_instead_of_a_bad_read(address, row_pitch, width, height):
    with pytest.raises(WindowStreamError):
        mapped_rows_to_bgr(address, row_pitch, width, height)


# --- a rögzítés létrehozása -----------------------------------------------------

def test_stream_of_a_window_that_does_not_exist_fails_with_its_own_error():
    with pytest.raises(WindowStreamError):
        WindowStream(MISSING_WINDOW, INTERVAL)


def test_stream_error_is_an_os_error_so_the_feed_falls_back_on_it():
    assert issubclass(WindowStreamError, OSError)


# --- saját képkocka-ritkítás ----------------------------------------------------

def test_first_frame_is_always_read():
    assert FramePacer(INTERVAL).accept(now=5.0)


def test_every_second_frame_is_dropped_when_the_source_is_twice_as_fast():
    pacer = FramePacer(INTERVAL)

    accepted = [pacer.accept(now=index * SOURCE_FRAME_GAP) for index in range(6)]

    assert accepted == [True, False, True, False, True, False]


def test_frame_arriving_slightly_early_still_counts_as_due():
    pacer = FramePacer(INTERVAL)
    pacer.accept(now=0.0)

    assert pacer.accept(now=INTERVAL - EARLY_FRAME_SLACK_SECONDS / 2)


def test_frame_arriving_well_before_its_time_is_dropped():
    pacer = FramePacer(INTERVAL)
    pacer.accept(now=0.0)

    assert not pacer.accept(now=INTERVAL - 2 * EARLY_FRAME_SLACK_SECONDS)


def test_dropped_frame_does_not_postpone_the_next_one():
    pacer = FramePacer(INTERVAL)
    pacer.accept(now=0.0)
    pacer.accept(now=INTERVAL / 2)

    assert pacer.accept(now=INTERVAL)


# --- a sáv helye a képkockán belül ----------------------------------------------

TITLE_BAR_CLIENT = Box(8, 31, 1288, 751)  # 1280×720-as kliensterület, fölötte címsor, körben szegély
WINDOWED_FRAME = (1296, 759)
BAND = Box(150, 560, 1130, 705)


def test_band_is_shifted_by_the_title_bar_and_the_border():
    assert band_in_frame(TITLE_BAR_CLIENT, BAND, WINDOWED_FRAME) == Box(158, 591, 1138, 736)


def test_band_keeps_its_place_in_a_borderless_window():
    full_screen = Box(0, 0, 1920, 1080)
    band = Box(230, 842, 1689, 1058)

    assert band_in_frame(full_screen, band, (1920, 1080)) == band


def test_band_reaching_outside_the_client_area_is_not_read():
    too_wide = Box(150, 560, 1281, 705)

    assert band_in_frame(TITLE_BAR_CLIENT, too_wide, WINDOWED_FRAME) is None


def test_empty_band_is_not_read():
    assert band_in_frame(TITLE_BAR_CLIENT, Box(150, 560, 150, 705), WINDOWED_FRAME) is None


def test_band_is_not_read_while_the_frame_is_still_smaller_than_the_resized_window():
    frame_before_the_resize = (1000, 700)

    assert band_in_frame(TITLE_BAR_CLIENT, BAND, frame_before_the_resize) is None


# --- a képfolyamra épülő forrás (hamis képfolyammal) ----------------------------

class FakeStream:
    """Hamis WindowStream: mindig van új képkockája; feljegyzi, melyik téglalapot olvasták ki belőle."""

    def __init__(self, limits_rate=True):
        self.limits_rate = limits_rate
        self.boxes = []

    @contextmanager
    def frame(self):
        yield WINDOWED_FRAME

    def read_box(self, box):
        self.boxes.append(box)
        return np.zeros((box.height, box.width, 3), np.uint8)


def stream_source(monkeypatch, stream, client=TITLE_BAR_CLIENT):
    monkeypatch.setattr(band_feed, "WindowStream", lambda _hwnd, _min_interval: stream)
    monkeypatch.setattr(band_feed, "client_box_in_frame", lambda _hwnd: client)
    return StreamSource(HWND, INTERVAL)


def test_band_is_read_from_its_place_in_the_frame_and_reported_in_client_coordinates(monkeypatch):
    stream = FakeStream()
    source = stream_source(monkeypatch, stream)

    band = source.read(lambda _width, _height: BAND)

    assert stream.boxes == [Box(158, 591, 1138, 736)]
    assert (band.area, band.frame_size, band.image.shape) == (BAND, (1280, 720), (145, 980, 3))


def test_area_callback_gets_the_client_size_not_the_frame_size(monkeypatch):
    asked_for = []
    source = stream_source(monkeypatch, FakeStream())

    source.read(lambda width, height: asked_for.append((width, height)) or BAND)

    assert asked_for == [(1280, 720)]


def test_minimised_window_gives_no_band(monkeypatch):
    stream = FakeStream()
    source = stream_source(monkeypatch, stream, client=None)

    assert source.read(lambda _width, _height: BAND) is None
    assert stream.boxes == []


def test_source_rests_until_the_next_frame_can_come_when_the_system_limits_the_rate(monkeypatch):
    source = stream_source(monkeypatch, FakeStream(limits_rate=True))

    assert source.poll_period == WGC_POLL_SECONDS
    assert source.band_period == pytest.approx(INTERVAL - WGC_POLL_SECONDS)


def test_source_keeps_asking_often_when_it_has_to_drop_frames_itself(monkeypatch):
    source = stream_source(monkeypatch, FakeStream(limits_rate=False))

    assert source.poll_period == source.band_period == WGC_POLL_SECONDS
