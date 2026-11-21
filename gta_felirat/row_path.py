"""A soralapú olvasás bekötése: rögzítő szál → sorfigyelő → felismerő szál → RowPipeline, a Tk-szál ébresztésével.

A részek a handoff, a row_tracker és a row_pipeline modulban vannak; ez az osztály csak összeköti őket,
és eldönti, mikor nem használható az út (keskeny kép, hibázó munkaszál): olyankor a hívó a régi,
teljes olvasású úton megy tovább ugyanazzal a pipeline-nal.

Szálak: az on_band() a rögzítő szálon fut (sorkeresés, ~1–3 ms sávonként), a felismerés saját
munkaszálon. Minden más metódust a megjelenítési (Tk) szál hív; a RowPath-ot is azon kell létrehozni
és lezárni (close() a gyökérablak lebontása előtt).
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from .band_feed import CapturedBand
from .frosted_cover import Backdrop
from .handoff import BandWatcher, Inbox, LatestBox, ReadRows, RowRecognizer, TkWaker, start_worker
from .i18n import t
from .ocr import Box
from .row_finder import is_wide_enough, make_row_finder
from .row_pipeline import RowPipeline
from .row_tracker import BandFrame, BandSnapshot

logger = logging.getLogger(__name__)


class WakingRowPipeline(RowPipeline):
    """RowPipeline, amely a teljes olvasás elkészültekor is jelez a megjelenítési szálnak (nem csak a kész
    fordításnál): így a ritka, rendes időzítő mellett sem késik a régi úton olvasott felirat."""

    def submit_band(self, capture: Callable[[], np.ndarray], area: Box, frame_size: tuple[int, int],
                    now: float) -> None:
        before = self._ocr_future
        super().submit_band(capture, area, frame_size, now)
        future, wake = self._ocr_future, self._wake
        if future is not None and future is not before and wake is not None:
            future.add_done_callback(lambda _done: wake())  # az OCR-szálon fut: csak jelez


class RowPath:
    def __init__(self, pipeline: RowPipeline, read_rows: ReadRows, min_score: float,
                 schedule: Callable[[], None], *, whole_band: bool = False) -> None:
        """read_rows: SubtitleReader.read_rows; min_score: config.min_text_score.
        schedule: handoff.tk_scheduler(...) – ezzel fut le a Tk-szálon a hívó kirajzolási lépése, amikor dolga van.
        whole_band: config.whole_band_text – a sáv középen kívüli szövegét is sorként keressük és olvassuk."""
        self._pipeline = pipeline
        self._waker = TkWaker(schedule)
        self._inbox = Inbox(self._waker.wake)
        self._jobs: LatestBox[BandSnapshot] = LatestBox()
        self._watcher = BandWatcher(make_row_finder(whole_band), self._inbox, self._jobs)
        start_worker("row-recognizer", self._jobs, RowRecognizer(read_rows, self._inbox, min_score).submit)
        self._is_broken = False
        self._backdrop: tuple[BandSnapshot, Backdrop] | None = None

    def wake(self) -> None:
        """Jelzés a Tk-szálnak; bármely szálból hívható, és nem blokkol (a pipeline a kész fordításnál hívja)."""
        self._waker.wake()

    def set_whole_band(self, whole_band: bool) -> None:
        """config.whole_band_text változott: a sorkereső a következő sávtól eszerint keres."""
        self._watcher.set_finder(make_row_finder(whole_band))

    def can_read(self, frame_width: int) -> bool:
        """Használható-e most a soralapú út ekkora képen."""
        return not self._is_broken and is_wide_enough(frame_width)

    def on_band(self, captured: CapturedBand) -> None:
        """A rögzítő szálon, minden új sávnál: sorkeresés és sorkövetés; az eredmény a postaládába kerül."""
        self._watcher.submit(BandFrame(captured.image, captured.area, captured.frame_size, captured.captured_at))

    def pump(self, now: float, frame_size: tuple[int, int]) -> bool:
        """A munkaszálak eredményei a pipeline-ba kerülnek. True, ha a kiírt felirat megváltozott.

        frame_size: a játékablak mostani mérete; a más méretre készült pillanatképet a pipeline eldobja.
        """
        mail = self._inbox.drain()
        if mail.failures:
            self._is_broken = True  # a részleteket a munkaszál már naplózta; innentől a teljes olvasás megy
            logger.warning(t("remote.rows_failed", worker=mail.failures[0].worker))
            self.suspend()
            return True
        self._pipeline.set_frame_size(frame_size)
        return self._pipeline.receive(mail.snapshot, mail.texts, now)

    def backdrop(self) -> Backdrop | None:
        """A takarás háttere: az a sáv, amelyből a mostani kiírás készült (None: még nincs)."""
        snapshot = self._pipeline.snapshot
        if snapshot is None:
            return None
        if self._backdrop is None or self._backdrop[0] is not snapshot:
            self._backdrop = (snapshot, Backdrop(snapshot.image, snapshot.area))
        return self._backdrop[1]

    def suspend(self) -> None:
        """A hívó a régi útra vált: a sorokat és a még fel nem dolgozott eredményeket elfelejtjük."""
        self._inbox.drain()
        self._pipeline.suspend_rows()
        self._backdrop = None

    def close(self) -> None:
        """Leállítja a felismerő és az ébresztő szálat."""
        self._jobs.close()
        self._waker.close()
