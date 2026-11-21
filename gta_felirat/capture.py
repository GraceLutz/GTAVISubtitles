"""Capture card kép beolvasása külön szálon (mindig a legfrissebb képkocka)."""

import logging
import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .i18n import t

MAX_PROBED_DEVICES = 8
PICTURE_CHECK_READS = 10  # ennyi olvasásból legalább egynek képet kell adnia (különben „nincs kép”)
READ_FAILURE_LIMIT = 60
RETRY_SLEEP_SECONDS = 0.05

API_AUTO, API_DSHOW, API_MSMF, API_ANY = "auto", "dshow", "msmf", "any"
API_IDS = {API_DSHOW: cv2.CAP_DSHOW, API_MSMF: cv2.CAP_MSMF, API_ANY: cv2.CAP_ANY}
API_NAMES = {API_DSHOW: "DirectShow", API_MSMF: "Media Foundation", API_ANY: "OpenCV"}

logger = logging.getLogger(__name__)


class CaptureError(RuntimeError):
    """A capture card nem nyitható meg vagy nem ad képet."""


@dataclass(frozen=True)
class FoundDevice:
    api: str  # "dshow", "msmf" (Windows) vagy "any"
    index: int  # a sorszám ezen a felületen: a két felület ugyanazt az eszközt más számon is láthatja
    size: tuple[int, int] | None  # az első kép (szélesség, magasság); None: megnyílt, de nem adott képet


def _apis(api: str) -> tuple[str, ...]:
    """A kipróbálandó felületek sorrendben.

    auto: előbb DirectShow, és csak ha azon az eszköz meg sem nyílik, Media Foundation (ezt használja a
    Windows Kamera alkalmazás): egyes olcsó USB-s HDMI-rögzítők csak ezen nyílnak meg.
    """
    if sys.platform != "win32":
        return (API_ANY,)
    return (API_DSHOW, API_MSMF) if api == API_AUTO else (api,)


def _first_picture(capture: cv2.VideoCapture) -> np.ndarray | None:
    for _ in range(PICTURE_CHECK_READS):
        ok, frame = capture.read()
        if ok and frame is not None:
            return frame
        time.sleep(RETRY_SLEEP_SECONDS)
    return None


def open_device(index: int, width: int, height: int, fps: int, api: str = API_AUTO) -> cv2.VideoCapture:
    for name in _apis(api):
        capture = cv2.VideoCapture(index, API_IDS[name])
        if not capture.isOpened():
            capture.release()
            continue
        try:
            # USB-s capture cardok 1080p60-at jellemzően csak MJPG-ben adnak
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            capture.set(cv2.CAP_PROP_FPS, fps)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except BaseException:
            capture.release()
            raise
        logger.info(t("capture.device_opened", index=index, api=API_NAMES[name]))
        return capture
    raise CaptureError(t("capture.device_open_failed", index=index))


def listing_apis(api: str = API_AUTO) -> tuple[str, ...]:
    """A felületek, amelyeken eszközt keresünk (auto: mindkettő; a sorszámok felületenként mások lehetnek)."""
    return _apis(api)


def list_devices(api: str = API_AUTO) -> list[FoundDevice]:
    """A megnyitható eszközök a megadott felületen (auto: mindkét felületen, felületenként külön)."""
    return [device for name in listing_apis(api) for device in _probe(name)]


def _probe(name: str) -> list[FoundDevice]:
    found = []
    for index in range(MAX_PROBED_DEVICES):
        capture = cv2.VideoCapture(index, API_IDS[name])
        try:
            if capture.isOpened():
                frame = _first_picture(capture)
                found.append(FoundDevice(name, index, None if frame is None else (frame.shape[1], frame.shape[0])))
        finally:
            capture.release()
    return found


class FrameGrabber:
    """Háttérszálon olvassa a képet, így a megjelenítés sosem vár a kártyára."""

    def __init__(self, capture: cv2.VideoCapture) -> None:
        self._capture = capture
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._error: CaptureError | None = None
        self._running = False
        self._thread = threading.Thread(target=self._run, name="frame-grabber", daemon=True)

    def start(self) -> "FrameGrabber":
        self._running = True
        self._thread.start()
        return self

    def latest(self) -> np.ndarray | None:
        with self._lock:
            if self._error is not None:
                raise self._error
            return self._frame

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        self._capture.release()

    def _run(self) -> None:
        failures = 0
        while self._running:
            ok, frame = self._capture.read()
            if ok and frame is not None:
                failures = 0
                with self._lock:
                    self._frame = frame
                continue
            failures += 1
            if failures >= READ_FAILURE_LIMIT:
                with self._lock:
                    self._error = CaptureError(t("capture.no_signal"))
                logger.error(t("capture.read_failed", count=failures))
                return
            time.sleep(RETRY_SLEEP_SECONDS)
