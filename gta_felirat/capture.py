"""Capture card kép beolvasása külön szálon (mindig a legfrissebb képkocka)."""

import logging
import sys
import threading
import time

import cv2
import numpy as np

MAX_PROBED_DEVICES = 8
READ_FAILURE_LIMIT = 60
RETRY_SLEEP_SECONDS = 0.05

logger = logging.getLogger(__name__)


class CaptureError(RuntimeError):
    """A capture card nem nyitható meg vagy nem ad képet."""


def _backend() -> int:
    return cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY


def open_device(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(index, _backend())
    if not capture.isOpened():
        raise CaptureError(f"A(z) {index}. videóeszköz nem nyitható meg")
    # USB-s capture cardok 1080p60-at jellemzően csak MJPG-ben adnak
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def list_devices() -> list[tuple[int, int, int]]:
    """Visszaadja a működő eszközök (index, szélesség, magasság) listáját."""
    found = []
    for index in range(MAX_PROBED_DEVICES):
        capture = cv2.VideoCapture(index, _backend())
        if capture.isOpened():
            ok, frame = capture.read()
            if ok and frame is not None:
                found.append((index, frame.shape[1], frame.shape[0]))
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
                    self._error = CaptureError("A capture card nem ad képet (kábel / eszköz?)")
                logger.error("Képolvasás sikertelen %d alkalommal egymás után", failures)
                return
            time.sleep(RETRY_SLEEP_SECONDS)
