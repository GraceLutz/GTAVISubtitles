"""Közös feliratfeldolgozás: OCR ütemezése → stabil felirat → (elő)fordítás.

A capture cardos ablak és a Remote Play réteg is ezt használja; csak a kép
forrása és a kirajzolás módja különbözik.
"""

import logging
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from .config import AppConfig
from .ocr import Box, OcrResult, SubtitleReader, is_spoken_line, region_to_box
from .subtitle_layer import Shown
from .subtitle_tracker import NewSubtitle, SubtitleCleared, SubtitleTracker, normalize, similarity
from .translation_service import TranslationService
from .translator import TranslationError

FAILED_PREFIX = "(fordítás nem sikerült) "
# Egy fordításra legfeljebb ennyit várunk, utána feladjuk (újrapróbálással együtt)
TRANSLATION_DEADLINE_SECONDS = 10.0
TIMEOUT_PREFIX = "(a fordítás 10 mp alatt nem jött meg) "
# Előre indított + megerősített fordítás egyszerre is futhat, holtpont nélkül
TRANSLATE_WORKERS = 3

logger = logging.getLogger(__name__)


class SubtitlePipeline:
    def __init__(self, config: AppConfig, reader: SubtitleReader, translation: TranslationService) -> None:
        self._config = config
        self._reader = reader
        self._translation = translation
        self._tracker = SubtitleTracker(config.stable_reads, config.similarity_threshold, config.hold_seconds)
        self._ocr_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr")
        self._translate_pool = ThreadPoolExecutor(max_workers=TRANSLATE_WORKERS, thread_name_prefix="translate")
        self._ocr_future: Future[OcrResult] | None = None
        self._translate_future: Future[str] | None = None
        self._pending_source_text = ""
        self._translate_started_at = 0.0
        self._last_box: Box | None = None
        self._last_prefetch = ""
        self._next_ocr_at = 0.0
        self._frame_size = (1920, 1080)
        self.shown: Shown | None = None

    def update_config(self, config: AppConfig) -> None:
        self._config = config

    def ocr_due(self, now: float) -> bool:
        """Kell-e új kép? (A Remote Play módban csak ilyenkor rögzítünk képernyőt.)"""
        return self._ocr_future is None and now >= self._next_ocr_at

    def submit_frame(self, frame: np.ndarray, now: float) -> None:
        if not self.ocr_due(now):
            return
        self._frame_size = (frame.shape[1], frame.shape[0])
        self._next_ocr_at = now + self._config.ocr_interval_seconds
        self._ocr_future = self._ocr_pool.submit(self._reader.read, frame.copy())

    def band_area(self, frame_width: int, frame_height: int) -> Box:
        """A feliratsáv helye a teljes képen: Remote Play módban csak ezt rögzítjük."""
        return region_to_box(self._config.subtitle_region, frame_width, frame_height)

    def submit_band(self, capture: Callable[[], np.ndarray], area: Box, frame_size: tuple[int, int],
                    now: float) -> None:
        """A feliratsáv rögzítése és olvasása az OCR-szálon: a megjelenítési szál nem vár rá."""
        if not self.ocr_due(now):
            return
        self._frame_size = frame_size
        self._next_ocr_at = now + self._config.ocr_interval_seconds
        self._ocr_future = self._ocr_pool.submit(lambda: self._reader.read_band(capture(), area))

    def report_hidden(self, now: float) -> bool:
        """A feliratsáv nem látszik (másik ablak takarja): üres olvasásnak számít.

        Így a régi felirat a szokásos várakozás után eltűnik. True, ha a kiírt felirat megváltozott.
        """
        if not self.ocr_due(now):
            return False
        self._next_ocr_at = now + self._config.ocr_interval_seconds
        if isinstance(self._tracker.update("", now), SubtitleCleared):
            self.shown = None
            return True
        return False

    def poll(self, now: float) -> bool:
        """Begyűjti a kész OCR-t és fordítást. True, ha a kiírt felirat megváltozott."""
        before = self.shown
        self._collect_ocr(now)
        self._collect_translation()
        return self.shown is not before

    def retranslate_current(self) -> None:
        if self._tracker.current:
            self._start_translation(self._tracker.current)

    def shutdown(self) -> None:
        self._ocr_pool.shutdown(wait=False, cancel_futures=True)
        self._translate_pool.shutdown(wait=False, cancel_futures=True)

    def _collect_ocr(self, now: float) -> None:
        if self._ocr_future is None or not self._ocr_future.done():
            return
        future, self._ocr_future = self._ocr_future, None
        try:
            result = future.result()
        except Exception:
            logger.exception("OCR hiba")
            return
        if self._is_own_translation(result.text):
            return  # a saját kiírt fordításunkat olvastuk vissza, nem új felirat
        text = result.text
        if self._config.only_spoken_lines and not is_spoken_line(text):
            text = ""  # küldetésszöveg, tábla, menü: nem kimondott szöveg, nem fordítjuk
        self._prefetch(text)
        event = self._tracker.update(text, now)
        if isinstance(event, NewSubtitle):
            self._last_box = result.box
            self._start_translation(event.text)
        elif isinstance(event, SubtitleCleared):
            self.shown = None

    def _is_own_translation(self, raw_text: str) -> bool:
        if self.shown is None or not raw_text.strip():
            return False
        return similarity(normalize(raw_text), normalize(self.shown.text)) >= self._config.similarity_threshold

    def _prefetch(self, raw_text: str) -> None:
        """Új szöveg első olvasásakor már fordítunk; a megerősítés a kész eredményt kapja."""
        text = normalize(raw_text)
        threshold = self._config.similarity_threshold
        if not text or similarity(text, self._tracker.current) >= threshold:
            return
        if similarity(text, self._last_prefetch) >= threshold:
            return  # ugyanannak a sornak egy OCR-változata, nem küldjük újra
        self._last_prefetch = text
        self._translation.prefetch(text, self._translate_pool)

    def _start_translation(self, text: str) -> None:
        logger.info("EN: %s", text)
        self._pending_source_text = text
        self._translate_started_at = time.monotonic()
        self._translate_future = self._translate_pool.submit(self._translation.translate, text)

    def _give_up_translation(self) -> None:
        """10 mp után nem várunk tovább: az angol szöveget mutatjuk jelzéssel."""
        self._translate_future = None
        source_text = self._pending_source_text
        if source_text != self._tracker.current:
            return
        logger.warning("A fordítás %.0f mp alatt nem jött meg, feladom", TRANSLATION_DEADLINE_SECONDS)
        anchor = self._last_box or region_to_box(self._config.subtitle_region, *self._frame_size)
        self.shown = Shown(text=TIMEOUT_PREFIX + source_text, anchor=anchor)

    def _collect_translation(self) -> None:
        if self._translate_future is None:
            return
        if not self._translate_future.done():
            if time.monotonic() - self._translate_started_at >= TRANSLATION_DEADLINE_SECONDS:
                self._give_up_translation()
            return
        future, self._translate_future = self._translate_future, None
        source_text = self._pending_source_text
        if source_text != self._tracker.current:
            return  # közben már másik felirat jelent meg
        try:
            translated = future.result()
            logger.info("→ %s", translated)
        except TranslationError as err:
            logger.warning("%s", err)
            translated = FAILED_PREFIX + source_text
        anchor = self._last_box or region_to_box(self._config.subtitle_region, *self._frame_size)
        self.shown = Shown(text=translated, anchor=anchor)
