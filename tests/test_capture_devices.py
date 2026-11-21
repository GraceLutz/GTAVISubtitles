import numpy as np
import pytest

from gta_felirat import capture
from gta_felirat.capture import CaptureError, FoundDevice, list_devices, open_device

FRAME = np.zeros((1080, 1920, 3), np.uint8)
DSHOW, MSMF = capture.cv2.CAP_DSHOW, capture.cv2.CAP_MSMF


class FakeCamera:
    """cv2.VideoCapture helyett: a `working` párokra (sorszám, felület) nyílik meg és ad képet."""

    working: set = set()
    silent: set = set()  # megnyílik, de nem ad képet
    opened: list = []
    open_now: int = 0
    most_open: int = 0

    def __init__(self, index, api):
        self.index, self.api = index, api
        self.settings = {}
        self.released = False
        FakeCamera.opened.append((index, api))
        FakeCamera.open_now += 1
        FakeCamera.most_open = max(FakeCamera.most_open, FakeCamera.open_now)

    def isOpened(self):
        return (self.index, self.api) in FakeCamera.working

    def read(self):
        has_picture = self.isOpened() and (self.index, self.api) not in FakeCamera.silent
        return (True, FRAME) if has_picture else (False, None)

    def set(self, prop, value):
        self.settings[prop] = value
        return True

    def release(self):
        if not self.released:
            self.released = True
            FakeCamera.open_now -= 1


@pytest.fixture
def cameras(monkeypatch):
    FakeCamera.working, FakeCamera.silent, FakeCamera.opened = set(), set(), []
    FakeCamera.open_now = FakeCamera.most_open = 0
    monkeypatch.setattr(capture.sys, "platform", "win32")
    monkeypatch.setattr(capture.cv2, "VideoCapture", FakeCamera)
    monkeypatch.setattr(capture.time, "sleep", lambda _seconds: None)
    return FakeCamera


def test_device_is_opened_through_directshow_first(cameras):
    cameras.working = {(0, DSHOW), (0, MSMF)}

    device = open_device(0, 1920, 1080, 60)

    assert device.api == DSHOW
    assert device.settings[capture.cv2.CAP_PROP_FRAME_WIDTH] == 1920
    assert cameras.open_now == 1


def test_device_directshow_cannot_open_is_tried_through_media_foundation(cameras):
    # Egyes olcsó USB-s HDMI-rögzítők csak az egyik Windows-os videós felületen nyílnak meg
    cameras.working = {(0, MSMF)}

    assert open_device(0, 1920, 1080, 60).api == MSMF
    assert cameras.most_open == 1


def test_device_that_opens_through_directshow_is_not_reopened_elsewhere(cameras):
    # Ugyanaz a sorszám a másik felületen másik eszköz is lehet: nem váltunk, ha a DirectShow megnyílt
    cameras.working = {(0, DSHOW), (0, MSMF)}
    cameras.silent = {(0, DSHOW)}

    assert open_device(0, 1920, 1080, 60).api == DSHOW
    assert cameras.opened == [(0, DSHOW)]


def test_chosen_interface_is_the_only_one_tried(cameras):
    cameras.working = {(0, DSHOW), (0, MSMF)}

    assert open_device(0, 1920, 1080, 60, api="msmf").api == MSMF
    assert cameras.opened == [(0, MSMF)]


def test_chosen_interface_that_cannot_open_gives_an_error(cameras):
    cameras.working = {(0, MSMF)}

    with pytest.raises(CaptureError):
        open_device(0, 1920, 1080, 60, api="dshow")
    assert cameras.opened == [(0, DSHOW)]


def test_device_no_interface_can_open_gives_an_error_with_the_index(cameras):
    with pytest.raises(CaptureError, match="0"):
        open_device(0, 1920, 1080, 60)

    assert [api for _index, api in cameras.opened] == [DSHOW, MSMF]
    assert cameras.open_now == 0


def test_device_is_released_when_setting_it_up_fails(cameras, monkeypatch):
    cameras.working = {(0, DSHOW)}

    def broken(self, prop, value):
        raise RuntimeError("driver")

    monkeypatch.setattr(FakeCamera, "set", broken)
    with pytest.raises(RuntimeError):
        open_device(0, 1920, 1080, 60)
    assert cameras.open_now == 0


def test_device_list_is_made_per_interface(cameras):
    cameras.working = {(0, DSHOW), (0, MSMF), (1, MSMF)}
    cameras.silent = {(1, MSMF)}

    assert list_devices() == [FoundDevice("dshow", 0, (1920, 1080)), FoundDevice("msmf", 0, (1920, 1080)),
                              FoundDevice("msmf", 1, None)]
    assert cameras.most_open == 1
    assert cameras.open_now == 0


def test_device_list_of_one_interface(cameras):
    cameras.working = {(0, DSHOW), (2, MSMF)}

    assert list_devices("msmf") == [FoundDevice("msmf", 2, (1920, 1080))]


def test_other_systems_use_the_default_interface(cameras, monkeypatch):
    monkeypatch.setattr(capture.sys, "platform", "linux")
    cameras.working = {(1, capture.cv2.CAP_ANY)}

    assert open_device(1, 1920, 1080, 60).api == capture.cv2.CAP_ANY
    assert list_devices() == [FoundDevice("any", 1, (1920, 1080))]
