"""A küldetésutasítás sora (pl. „Go to your car”): a párbeszédtől függetlenül, a saját sorára fordítjuk.

A párbeszédsorok alatt, középen álló sor. A felirat-követő (subtitle_tracker) nem tud róla: a párbeszéd
változása nem fordítja újra, és a küldetéssor változása nem tünteti el a párbeszéd fordítását.
Amíg a fordítása nem kész, legfeljebb PENDING_HIDE_SECONDS-ig takarjuk; ha a fordítás nem sikerül,
az angol sor látszik.

Szálak: minden metódust a megjelenítési (Tk) szál hív; a fordítás a kapott végrehajtón fut, és a
kész fordításról a `wake` jelez (szálbiztos, nem blokkol).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import Executor, Future

from .line_placement import PENDING_HIDE_SECONDS, TranslatedLine
from .ocr import Box, OcrLine
from .subtitle_tracker import is_same_subtitle, normalize
from .translator import TranslationError

logger = logging.getLogger(__name__)


class ObjectiveLine:
    """A most látható küldetéssor és a fordítása."""

    def __init__(self, translate: Callable[[str], str], pool: Executor, threshold: float, *,
                 wake: Callable[[], None] | None = None) -> None:
        self._translate = translate
        self._pool = pool
        self._threshold = threshold  # ennyire hasonló olvasat ugyanaz a sor (config.similarity_threshold)
        self._wake = wake
        self._line: OcrLine | None = None
        self._since = 0.0  # mióta látszik ez a szöveg
        self._future: Future[str] | None = None
        self._translated: str | None = None

    def see(self, line: OcrLine | None, now: float) -> bool:
        """A mostani kép küldetéssora (None: nincs). True, ha a kiírandó állapot megváltozhatott."""
        if line is None:
            changed = self._line is not None
            self._line, self._future, self._translated = None, None, None
            return changed
        if self._line is not None and is_same_subtitle(normalize(line.text), normalize(self._line.text),
                                                       self._threshold):
            changed = line.box != self._line.box
            self._line = OcrLine(self._line.text, line.box)  # az első olvasat marad: a fordítás ahhoz tartozik
            return changed
        self._line, self._since, self._translated = line, now, None
        self._future = self._pool.submit(self._translate, line.text)
        wake = self._wake
        if wake is not None:
            self._future.add_done_callback(lambda _done: wake())  # a fordító szálon fut: csak jelez
        return True

    def collect(self) -> bool:
        """Átveszi a kész fordítást. True, ha megérkezett (a kiírást frissíteni kell)."""
        future = self._future
        if future is None or not future.done():
            return False
        self._future = None
        try:
            self._translated = future.result()
        except TranslationError as err:
            logger.warning("%s", err)
            self._translated = None  # az angol sor látszik
        return self._translated is not None

    def pair(self) -> tuple[TranslatedLine, OcrLine] | None:
        """A kiírandó fordítás a küldetéssor helyén, ha kész."""
        if self._line is None or self._translated is None:
            return None
        return TranslatedLine(self._line.text, self._translated), self._line

    def hidden(self, now: float) -> list[Box]:
        """A még fordításra váró küldetéssor doboza, legfeljebb PENDING_HIDE_SECONDS-ig."""
        if self._line is None or self._future is None or now - self._since >= PENDING_HIDE_SECONDS:
            return []
        return [self._line.box]

    def deadline(self, now: float) -> float | None:
        """Mikor jár le a takarás (ha most takarunk)."""
        return self._since + PENDING_HIDE_SECONDS if self.hidden(now) else None
