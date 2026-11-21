"""Remote Play mód: a PS Remote Play (vagy bármely más) ablak fölé rakott feliratréteg.

Nincs capture card és nincs saját játékablak: a Remote Play marad előtérben, így a kontroller is ott működik.

Az olvasás három úton mehet, a leggyorsabbtól a legbiztosabbig; a lassabb mindig tartaléka a gyorsabbnak:
1. soronként (row_path): a folyamatos rögzítés minden sávján sorkereső fut, és csak az új sort olvassuk be;
2. a folyamatos rögzítés (band_feed) legfrissebb sávjának teljes olvasása, azonnali takarással (eager_hide);
3. olvasásonkénti rögzítés az OCR-szálon (band_capture), ha a folyamatos rögzítésnek nincs használható sávja.

A megjelenítési (Tk) szálat a munkaszálak ébresztik, amikor dolga van (új sáv, felismert szöveg, kész olvasás
vagy fordítás); a rendes időzítő csak a ráérős teendőket viszi (ablakkövetés, lejáró takarások).
"""

import logging
import time
from pathlib import Path

import cv2

from .app_base import SubtitleAppBase
from .band_capture import BandCapture, WindowContentGrabber
from .band_feed import SOURCE_NONE, BandFeed, CapturedBand
from .config import AppConfig
from .eager_hide import EagerCover, with_eager_boxes
from .feed_gate import FeedGate
from .frosted_cover import Backdrop
from .handoff import tk_scheduler
from .i18n import t
from .layer_plan import Shown
from .ocr import Box, SubtitleReader, region_to_box
from .overlay_window import OverlayWindow
from .row_path import RowPath, WakingRowPipeline
from .translator import CachedTranslator
from .window_capture import (ScreenGrabber, ScreenRect, blank_out, client_rect, find_window, is_unobstructed,
                             window_exists)

DEFAULT_WINDOW_TITLE = "PS Remote Play"
# A rendes időzítő üteme. Ennél sűrűbben fölösleges: az új eredményekre a munkaszálak ébresztik a megjelenítést
# (a 4 ms-os ütem 1 ms-os rendszeridőzítővel mérve ~13% processzormagot égetett el üresjáratban)
TICK_MS = 15
KEEP_ON_TOP_SECONDS = 1.0
# Amíg a folyamatos rögzítés él, de áll a kép (nem jön új képkocka), a teljes olvasás csak biztosítéknak megy, ritkán
SAFETY_READ_SECONDS = 0.5
# A saját fordításunk kitakarása az OCR elől ennyivel bővebben (árnyék, körvonal)
OWN_TEXT_MARGIN_PX = 8

logger = logging.getLogger(__name__)


class RemotePlayApp(SubtitleAppBase):
    _pipeline: WakingRowPipeline

    def __init__(self, config: AppConfig, reader: SubtitleReader, *, window_title: str = DEFAULT_WINDOW_TITLE,
                 match_title_start: bool = False, translator: CachedTranslator | None = None,
                 config_path: Path | None = None) -> None:
        self._row_path: RowPath | None = None  # a pipeline előbb készül el, mint az ébresztő (ahhoz Tk-ablak kell)
        super().__init__(config, reader, translator, config_path)
        # Ezekhez a kis képekhez az OpenCV munkaszálai csak processzort égetnek (mérve: ~7× annyit, gyorsulás nélkül)
        cv2.setNumThreads(1)
        self._hwnd = find_window(window_title, by_start=match_title_start)
        self._grabber = ScreenGrabber()
        # Elsőként a Remote Play ablak saját tartalmát olvassuk: abban nincs benne a feliratréteg
        self._capture = BandCapture(WindowContentGrabber(self._hwnd), self._grabber, self._shows_only_the_game)
        # Folyamatos sávrögzítés külön szálon: az olvasás és a takarás nem vár a rögzítésre.
        # Amíg nincs használható sávja, a fenti, olvasásonkénti rögzítés megy.
        self._feed = BandFeed(self._hwnd, self._pipeline.band_area, on_band=self._on_band)
        self._gate = FeedGate(self._feed)
        self._reads_rows = False  # most soronként olvasunk-e; a Tk-szál írja, a rögzítő szál olvassa
        self._can_read_rows: bool | None = None  # a naplóba utoljára kiírt olvasási mód
        self._read_sequence = 0  # a legutóbb teljes olvasásra adott sáv sorszáma
        self._next_full_read = 0.0
        self._logged_source = SOURCE_NONE  # a naplóba utoljára kiírt rögzítési mód
        self._eager = EagerCover()  # a teljes olvasású út háttere és azonnali takarása
        self._drawn_backdrop: Backdrop | None = None
        self._drawn_shown: Shown | None = None
        self._preview = False
        self._next_keep_on_top = 0.0
        self._running = True
        self._band_covered = False
        self._settings = self._create_settings_window(on_preview=self._set_preview, on_close=self.stop)
        self._overlay = OverlayWindow(self._settings.root, self._subtitles, config.visible_in_recordings)
        self._row_path = RowPath(self._pipeline, reader.read_rows, config.min_text_score,
                                 tk_scheduler(self._settings.root, self._on_wake))

    def _create_pipeline(self, config: AppConfig, reader: SubtitleReader) -> WakingRowPipeline:
        """Soralapú pipeline; amíg nem kap pillanatképet, pontosan úgy működik, mint a régi."""
        return WakingRowPipeline(config, reader, self._translation, wake=self._wake_display)

    def _wake_display(self) -> None:
        """Bármely szálból: a megjelenítési szál rögtön lép egyet, nem várja meg a rendes időzítőt."""
        row_path = self._row_path
        if row_path is not None:
            row_path.wake()

    def run(self) -> None:
        root = self._settings.root
        root.after(TICK_MS, self._tick)
        if self._config.fast_capture:
            self._feed.start()  # el nem indítva nincs sávja: a régi, olvasásonkénti rögzítés megy
        try:
            root.mainloop()
        except KeyboardInterrupt:
            logger.info(t("remote.ctrl_c_quit"))
        finally:
            self._feed.stop()
            self._row_path.close()
            self._pipeline.shutdown()
            self._grabber.close()

    def stop(self) -> None:
        self._running = False
        self._row_path.close()  # az ébresztő szálnak a gyökérablak lebontása előtt kell leállnia
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
        self._run_step()
        if self._running:
            self._settings.root.after(TICK_MS, self._tick)

    def _on_wake(self) -> None:
        """Egy munkaszál jelzett (új sáv, felismert szöveg, kész olvasás vagy fordítás)."""
        if self._running:
            self._run_step()

    def _run_step(self) -> None:
        try:
            self._step(time.monotonic())
        except KeyboardInterrupt:
            self.stop()  # Ctrl+C a terminálban: a Tk különben elnyelné
        except Exception:
            logger.exception(t("remote.layer_error"))

    def _step(self, now: float) -> None:
        self._settings.refresh_status()
        rect = client_rect(self._hwnd)
        if rect is None:
            if not window_exists(self._hwnd):
                logger.info(t("remote.window_closed"))
                self.stop()
                return
            self._overlay.hide()  # le van kicsinyítve
            return
        self._overlay.place(rect)
        self._report_capture_source()
        reads_rows = self._choose_reading(rect.width)
        if self._read(rect, now, reads_rows):
            self._overlay.needs_redraw = True
        self._draw(rect, now, reads_rows)
        self._translation.keep_warm()
        if now >= self._next_keep_on_top:
            self._next_keep_on_top = now + KEEP_ON_TOP_SECONDS
            self._overlay.keep_on_top()

    def _choose_reading(self, frame_width: int) -> bool:
        """Soronként olvasunk-e most: a folyamatos rögzítés élő sávjából, elég széles ablakban, amíg nem hibázik."""
        can_read_rows = (self._config.fast_capture and self._config.row_reading
                         and self._row_path.can_read(frame_width))
        if can_read_rows != self._can_read_rows:
            self._can_read_rows = can_read_rows
            logger.info(t("remote.reading_rows") if can_read_rows else t("remote.reading_full"))
        reads_rows = can_read_rows and self._gate.band() is not None
        if reads_rows != self._reads_rows:  # pl. megállt a kép: amíg nem jön új sáv, a teljes olvasás a biztosíték
            self._reads_rows = reads_rows
            if not reads_rows:
                self._hand_over_to_full_reading()
            self._overlay.needs_redraw = True
        return reads_rows

    def _hand_over_to_full_reading(self) -> None:
        """A teljes olvasás ott folytatja, ahol a soralapú abbahagyta: ugyanazzal a háttérrel, felirattal, fordítással."""
        snapshot = self._pipeline.snapshot
        if snapshot is not None:
            self._publish(snapshot.image, snapshot.area, snapshot.captured_at)
        self._row_path.suspend()

    def _read(self, rect: ScreenRect, now: float, reads_rows: bool) -> bool:
        """Begyűjti az olvasás és a fordítás eredményeit, és ha kell, új olvasást indít. True: változott a kiírás."""
        if reads_rows:
            changed = self._row_path.pump(now, (rect.width, rect.height))
            return self._pipeline.poll(now) or changed  # a váltás előtt indult utolsó teljes olvasás még beérhet
        # Előbb begyűjtjük a kész olvasást, és csak utána indítjuk a következőt: így az új olvasás
        # ugyanebben a lépésben elindul, nem egy teljes ütemmel (és egy kirajzolással) később
        changed = self._pipeline.poll(now)
        if self._pipeline.ocr_due(now) and now >= self._next_full_read:
            self._next_full_read = now + SAFETY_READ_SECONDS if self._gate.is_still else 0.0
            self._submit_subtitle_band(rect, now)
        return changed

    def _draw(self, rect: ScreenRect, now: float, reads_rows: bool) -> None:
        reads_window_content = self._gate.band() is not None or self._capture.uses_window_content
        hide_original = effective_hide_original(self._config, ocr_sees_overlay=not reads_window_content)
        backdrop, shown = self._picture(now, reads_rows, hide_original)
        if shown is not None and backdrop is not self._drawn_backdrop:
            self._overlay.needs_redraw = True  # a takarás kövesse a mozgó játékképet
        if shown != self._drawn_shown:
            self._overlay.needs_redraw = True  # más látszik vagy más van takarva (pl. egy új sor azonnali takarása)
        if self._overlay.needs_redraw:
            preview = region_to_box(self._config.subtitle_region, rect.width, rect.height) if self._preview else None
            self._overlay.render(shown, hide_original, preview, backdrop)
            self._drawn_backdrop, self._drawn_shown = backdrop, shown

    def _picture(self, now: float, reads_rows: bool, hide_original: bool) -> tuple[Backdrop | None, Shown | None]:
        """(a takarás háttere, a kiírandó állapot) az éppen használt olvasási út szerint."""
        shown = self._pipeline.shown
        if not hide_original:
            return None, shown
        if reads_rows:  # a pipeline maga takarja az új sorokat; a háttér az a sáv, amelyből a kiírás készült
            return self._row_path.backdrop(), shown
        backdrop, eager = self._eager.view(shown, now)
        return backdrop, with_eager_boxes(shown, eager)

    def _report_capture_source(self) -> None:
        """Naplózza, ha változott, hogyan rögzítjük a feliratsávot (ebből látszik, melyik úton megy a kép)."""
        source = self._feed.source
        if source == self._logged_source:
            return
        self._logged_source = source
        if source == SOURCE_NONE:
            logger.info(t("remote.capture_per_read"))
        else:
            logger.info(t("remote.capture_continuous", source=source))

    def _submit_subtitle_band(self, rect: ScreenRect, now: float) -> None:
        """A legfrissebb feliratsáv teljes olvasása; ha a folyamatos rögzítésnek nincs sávja, az OCR-szál maga rögzít."""
        latest = self._gate.band()
        if latest is not None:
            band = fresh_band(latest, self._read_sequence, (rect.width, rect.height))
            if band is not None:  # különben a következő lépésben újra megnézzük, jött-e új kép
                self._read_sequence = band.sequence
                self._pipeline.submit_band(lambda: band.image, band.area, band.frame_size, now)
            return
        area = self._pipeline.band_area(rect.width, rect.height)
        if area.width <= 0 or area.height <= 0:
            return
        band_rect = ScreenRect(rect.left + area.left, rect.top + area.top, area.width, area.height)
        # Az ablak saját tartalmát a fölé húzott ablakok sem zavarják; csak képernyőről olvasva kell nézni
        covered = (not self._capture.uses_window_content
                   and not is_unobstructed(band_rect, {self._hwnd, self._overlay.hwnd}))
        if covered != self._band_covered:
            self._band_covered = covered
            logger.info(t("remote.band_covered") if covered else t("remote.band_visible"))
        if covered:
            if self._pipeline.report_hidden(now):
                self._overlay.needs_redraw = True
            return
        own_text = self._own_text_box()
        self._pipeline.submit_band(lambda: self._grab_band(band_rect, area, own_text), area,
                                   (rect.width, rect.height), now)

    def _grab_band(self, band_rect: ScreenRect, area: Box, own_text: Box | None):
        """Az OCR-szálon fut: rögzít, és ha kell, kitakarja a saját kirajzolt fordításunkat."""
        band = self._capture.grab(band_rect, area)
        if self._capture.last_from_window:
            self._publish(band, area, time.monotonic())
            self._wake_display()  # az új sor takarása ne várjon az olvasás végéig
            return band
        return band if own_text is None else blank_out(band, area, own_text, OWN_TEXT_MARGIN_PX)

    def _on_band(self, captured: CapturedBand) -> None:
        """A rögzítő szálon fut, minden új sávnál."""
        self._gate.on_band(captured)
        if self._reads_rows:
            self._row_path.on_band(captured)  # sorkeresés; az eredményt a Tk-szál veszi át
        else:
            self._publish(captured.image, captured.area, captured.captured_at)
        self._wake_display()  # a takarás háttere minden új sávval frissüljön

    def _publish(self, band, area: Box, captured_at: float) -> None:
        """A teljes olvasású út kapja: friss háttér a takaráshoz, és (ha takarunk) az új sorok azonnali takarása."""
        region = self._config.subtitle_region if self._config.hide_original else None
        self._eager.publish(band, area, region, captured_at)

    def _shows_only_the_game(self, band_rect: ScreenRect) -> bool:
        """Az OCR-szálon: a képernyőn a sáv helyén most csak a játék látszhat-e (más ablak nem takarja, mi nem rajzolunk rá)."""
        return (self._subtitles.last_rect is None
                and is_unobstructed(band_rect, {self._hwnd, self._overlay.hwnd}))

    def _own_text_box(self) -> Box | None:
        """Ahol most a fordításunk látszik, ha a réteg a képrögzítésben is látszik (különben nem kell takarni)."""
        if not self._config.visible_in_recordings or self._subtitles.last_rect is None:
            return None
        x, y, width, height = self._subtitles.last_rect
        return Box(x, y, x + width, y + height)


def fresh_band(latest: CapturedBand | None, read_sequence: int, frame_size: tuple[int, int]) -> CapturedBand | None:
    """Az olvasásra adható sáv: csak új kép, és csak ha az ablak mostani méretére készült."""
    if latest is None or latest.sequence == read_sequence or latest.frame_size != frame_size:
        return None
    return latest


def effective_hide_original(config: AppConfig, ocr_sees_overlay: bool = True) -> bool:
    """Az angolt csak akkor takarhatjuk, ha az OCR nem látja a takarást (különben semmit sem olvasna).

    Az OCR akkor látja a réteget, ha a képernyőről olvas és a réteg a felvételen is látszik.
    """
    return config.hide_original and not (config.visible_in_recordings and ocr_sees_overlay)
