"""A fő ciklus: kép → OCR → stabil felirat → fordítás → feliratpanel."""

import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .app_base import SubtitleAppBase
from .capture import CaptureError
from .config import AppConfig
from .facecam import FacecamStyle, place_facecam
from .facecam_drag import FacecamDragger
from .ocr import SubtitleReader, region_to_box
from .translator import CachedTranslator

WINDOW_NAME = "GTA felirat"
KEY_ESC = 27
REGION_COLOR = (0, 200, 255)
IDLE_WAIT_MS = 10

logger = logging.getLogger(__name__)


class FrameSource(Protocol):
    def latest(self) -> np.ndarray | None: ...
    def stop(self) -> None: ...


@dataclass(frozen=True)
class ViewOptions:
    fullscreen: bool
    show_region: bool = False
    overlay_enabled: bool = True
    webcam_enabled: bool = True


class OverlayApp(SubtitleAppBase):
    """Capture card mód: saját ablak, benne a játékkép és a feliratpanel."""

    def __init__(self, config: AppConfig, source: FrameSource, reader: SubtitleReader, *,
                 translator: CachedTranslator | None = None, webcam: FrameSource | None = None,
                 config_path: Path | None = None, settings_window: bool = True) -> None:
        super().__init__(config, reader, translator, config_path)
        self._source = source
        self._webcam = webcam
        self._facecam = FacecamDragger(
            FacecamStyle(config.webcam_x, config.webcam_y, config.webcam_width_ratio, config.webcam_mirror),
            on_save=self._save_webcam_layout)
        self._view = ViewOptions(fullscreen=config.fullscreen)
        self._settings = self._create_settings_window() if settings_window else None

    # --- fő ciklus --------------------------------------------------------
    def run(self) -> None:
        self._open_window()
        try:
            while self._tick():
                pass
        finally:
            self._pipeline.shutdown()
            self._source.stop()
            if self._webcam is not None:
                self._webcam.stop()
            if self._settings is not None:
                self._settings.close()
            cv2.destroyAllWindows()

    def _tick(self) -> bool:
        if self._settings is not None:
            self._settings.pump()
        frame = self._source.latest()
        if frame is None:
            return self._handle_key(cv2.waitKey(IDLE_WAIT_MS))
        now = time.monotonic()
        self._pipeline.submit_frame(frame, now)
        self._pipeline.poll(now)
        self._translation.keep_warm()
        cv2.imshow(WINDOW_NAME, self._compose(frame))
        if not self._handle_key(cv2.waitKey(1)):
            return False
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1

    # --- rajzolás ---------------------------------------------------------
    def _compose(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        output = frame
        if self._view.overlay_enabled:
            preview = region_to_box(self._config.subtitle_region, width, height) if self._view.show_region else None
            output = self._subtitles.draw(frame, self._pipeline.shown, self._config.hide_original, preview)
        if self._view.show_region:
            area = region_to_box(self._config.subtitle_region, width, height)
            output = output.copy()
            cv2.rectangle(output, (area.left, area.top), (area.right, area.bottom), REGION_COLOR, 2)
        return self._draw_webcam(output)

    def _draw_webcam(self, frame: np.ndarray) -> np.ndarray:
        if self._webcam is None or not self._view.webcam_enabled:
            return frame
        try:
            cam_frame = self._webcam.latest()
        except CaptureError as err:
            logger.warning("Webkamera kikapcsolva: %s", err)
            self._webcam.stop()
            self._webcam = None
            return frame
        if cam_frame is None:
            return frame
        self._facecam.set_geometry(frame.shape[1], frame.shape[0], cam_frame.shape[1], cam_frame.shape[0])
        return place_facecam(frame, cam_frame, self._facecam.style, highlighted=self._facecam.is_dragging)

    def _save_webcam_layout(self, style: FacecamStyle) -> None:
        self._save({"webcam_x": style.x, "webcam_y": style.y, "webcam_width_ratio": style.width_ratio})

    # --- egér és billentyűzet ---------------------------------------------
    def _on_mouse(self, event: int, x: int, y: int, flags: int, _param) -> None:
        # A feliratpanel kapja először az egeret, kivéve ha épp a webkamerás képet húzzuk
        if not self._facecam.is_dragging and self._view.overlay_enabled:
            if self._subtitles.handle_mouse(event, x, y, flags):
                return
        if self._webcam is not None and self._view.webcam_enabled:
            self._facecam.handle_mouse(event, x, y, flags)

    def _open_window(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)
        self._apply_fullscreen()

    def _apply_fullscreen(self) -> None:
        mode = cv2.WINDOW_FULLSCREEN if self._view.fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, mode)

    def _handle_key(self, key: int) -> bool:
        key &= 0xFF
        if key in (ord("q"), KEY_ESC):
            return False
        if key == ord("f"):
            self._view = replace(self._view, fullscreen=not self._view.fullscreen)
            self._apply_fullscreen()
        elif key == ord("h"):
            self.apply_updates({"hide_original": not self._config.hide_original})
        elif key == ord("r"):
            self._view = replace(self._view, show_region=not self._view.show_region)
        elif key == ord("o"):
            self._view = replace(self._view, overlay_enabled=not self._view.overlay_enabled)
        elif key == ord("c"):
            self._view = replace(self._view, webcam_enabled=not self._view.webcam_enabled)
        elif key == ord("p") and self._settings is not None:
            self._settings.toggle()
        return True


class StaticFrameSource:
    """Teszteléshez: egy képernyőképet ad vissza capture card helyett."""

    def __init__(self, image: np.ndarray) -> None:
        self._image = image

    def latest(self) -> np.ndarray | None:
        return self._image

    def stop(self) -> None:
        return None
