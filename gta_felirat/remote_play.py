"""Remote Play mód: a PS Remote Play (vagy bármely más) ablak fölé rakott feliratréteg.

Nincs capture card és nincs saját játékablak: a Remote Play marad előtérben,
így a kontroller is ott működik. Képet csak akkor rögzítünk, amikor OCR jön.
"""

import logging
import time
from pathlib import Path

from .app_base import SubtitleAppBase
from .config import AppConfig
from .ocr import SubtitleReader, region_to_box
from .overlay_window import OverlayWindow
from .translator import CachedTranslator
from .ocr import Box
from .window_capture import (ScreenGrabber, ScreenRect, blank_out, client_rect, find_window, is_unobstructed,
                             window_exists)

DEFAULT_WINDOW_TITLE = "PS Remote Play"
TICK_MS = 15
KEEP_ON_TOP_SECONDS = 1.0
# A saját fordításunk kitakarása az OCR elől ennyivel bővebben (árnyék, körvonal)
OWN_TEXT_MARGIN_PX = 8

logger = logging.getLogger(__name__)


class RemotePlayApp(SubtitleAppBase):
    def __init__(self, config: AppConfig, reader: SubtitleReader, *, window_title: str = DEFAULT_WINDOW_TITLE,
                 translator: CachedTranslator | None = None, config_path: Path | None = None) -> None:
        super().__init__(config, reader, translator, config_path)
        self._hwnd = find_window(window_title)
        self._grabber = ScreenGrabber()
        self._preview = False
        self._next_keep_on_top = 0.0
        self._running = True
        self._band_covered = False
        self._settings = self._create_settings_window(on_preview=self._set_preview, on_close=self.stop)
        self._overlay = OverlayWindow(self._settings.root, self._subtitles, config.visible_in_recordings)

    def run(self) -> None:
        root = self._settings.root
        root.after(TICK_MS, self._tick)
        try:
            root.mainloop()
        except KeyboardInterrupt:
            logger.info("Ctrl+C: kilépek")
        finally:
            self._pipeline.shutdown()
            self._grabber.close()

    def stop(self) -> None:
        self._running = False
        self._overlay.destroy()
        self._settings.close()

    def _set_preview(self, enabled: bool) -> None:
        self._preview = enabled
        self._overlay.needs_redraw = True

    def _on_config_changed(self) -> None:
        self._overlay.set_capture_visible(self._config.visible_in_recordings)
        self._overlay.needs_redraw = True

    def _tick(self) -> None:
        if not self._running:
            return
        try:
            self._step(time.monotonic())
        except KeyboardInterrupt:
            self.stop()  # Ctrl+C a terminálban: a Tk különben elnyelné
            return
        except Exception:
            logger.exception("Hiba a Remote Play rétegben")
        if self._running:
            self._settings.root.after(TICK_MS, self._tick)

    def _step(self, now: float) -> None:
        self._settings.refresh_status()
        rect = client_rect(self._hwnd)
        if rect is None:
            if not window_exists(self._hwnd):
                logger.info("A Remote Play ablakot bezárták, kilépek")
                self.stop()
                return
            self._overlay.hide()  # le van kicsinyítve
            return
        self._overlay.place(rect)
        if self._pipeline.ocr_due(now):
            self._submit_subtitle_band(rect, now)
        if self._pipeline.poll(now):
            self._overlay.needs_redraw = True
        if self._overlay.needs_redraw:
            preview = region_to_box(self._config.subtitle_region, rect.width, rect.height) if self._preview else None
            self._overlay.render(self._pipeline.shown, effective_hide_original(self._config), preview)
        self._translation.keep_warm()
        if now >= self._next_keep_on_top:
            self._next_keep_on_top = now + KEEP_ON_TOP_SECONDS
            self._overlay.keep_on_top()

    def _submit_subtitle_band(self, rect: ScreenRect, now: float) -> None:
        """Csak a feliratsávot rögzítjük (~4× gyorsabb), és azt is az OCR-szálon."""
        area = self._pipeline.band_area(rect.width, rect.height)
        if area.width <= 0 or area.height <= 0:
            return
        band_rect = ScreenRect(rect.left + area.left, rect.top + area.top, area.width, area.height)
        covered = not is_unobstructed(band_rect, {self._hwnd, self._overlay.hwnd})
        if covered != self._band_covered:
            self._band_covered = covered
            logger.info("A feliratsávot %s", "másik ablak takarja, nem olvasom" if covered else "újra látom")
        if covered:
            if self._pipeline.report_hidden(now):
                self._overlay.needs_redraw = True
            return
        own_text = self._own_text_box()
        self._pipeline.submit_band(lambda: self._grab_band(band_rect, area, own_text), area,
                                   (rect.width, rect.height), now)

    def _grab_band(self, band_rect: ScreenRect, area: Box, own_text: Box | None):
        """Az OCR-szálon fut: rögzít, és ha kell, kitakarja a saját kirajzolt fordításunkat."""
        band = self._grabber.grab(band_rect)
        return band if own_text is None else blank_out(band, area, own_text, OWN_TEXT_MARGIN_PX)

    def _own_text_box(self) -> Box | None:
        """Ahol most a fordításunk látszik, ha a réteg a képrögzítésben is látszik (különben nem kell takarni)."""
        if not self._config.visible_in_recordings or self._subtitles.last_rect is None:
            return None
        x, y, width, height = self._subtitles.last_rect
        return Box(x, y, x + width, y + height)


def effective_hide_original(config: AppConfig) -> bool:
    """Felvételre látszó rétegnél az angolt nem takarjuk: a takarást az OCR is látná, és nem olvasna semmit."""
    return config.hide_original and not config.visible_in_recordings
