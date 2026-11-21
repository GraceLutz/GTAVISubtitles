"""Átadás a szálak között a soralapú úton: postaládák, a Tk-szál ébresztése és a két munkafolyamat.

Ki mit csinál:
- a rögzítő szál mindig a legfrissebb feliratsávot tartja, és minden új sávot átad a BandWatcher-nek
  (közvetlenül, a saját szálán, vagy egy LatestBox-on át egy külön munkaszálnak: start_worker);
- BandWatcher: sorkeresés és sorkövetés → BandSnapshot az Inbox-ba; ha új sor látszik, a pillanatkép
  a felismerő postaládájába (LatestBox) is bekerül;
- RowRecognizer (saját munkaszálon: start_worker): csak a még olvasatlan sorokat olvassa, mindig a
  legfrissebb képről; az eredménye (RowText-ek) az Inbox-ba megy;
- a megjelenítési (Tk) szál az egyetlen, amely a pipeline állapotát írja és rajzol: kiüríti az Inbox-ot
  (drain), és amit kapott, átadja a RowPipeline.receive()-nek. Az Inbox felébreszti (TkWaker), amikor
  dolga van, így nem kell sűrűn kérdezgetnie.

Szabályok:
- munkaszál nem nyúl sem a pipeline-hoz, sem a Tk-hoz: csak megváltoztathatatlan eredményt tesz az Inbox-ba;
- az Inbox és a LatestBox bármely szálból használható; a BandWatcher.submit-et mindig ugyanaz az egy szál
  hívja, és a RowRecognizer.submit-et is (a saját munkaszála);
- a TkWaker-t és az ütemezőjét (tk_scheduler) a Tk-szálon kell létrehozni és lezárni; a wake() bármely
  szálból hívható, és nem blokkol;
- az ébresztés csak gyorsítás: a hívó tartsa meg a rendes (lassú) időzítőjét, amely akkor is kiüríti az
  Inbox-ot, ha egy ébresztés elmarad (indulás, leállás), és viszi tovább az időfüggő állapotokat;
- leállás: a postaládák close()-a után a start_worker szálai maguktól kilépnek; a TkWaker.close() a
  gyökérablak lebontása előtt hívandó. Minden szál démon: a program kilépését egyik sem tartja fel.

Idő: a BandFrame.captured_at ugyanabból az órából való, mint a pipeline `now` értéke (time.monotonic()).

Összerakva (a Tk-szálon; a finder a sorkereső köré írt illesztő, a watcher.submit-et a rögzítő szál hívja):

    waker = TkWaker(tk_scheduler(root, on_mail))
    inbox, jobs = Inbox(waker.wake), LatestBox()
    watcher = BandWatcher(finder, inbox, jobs)
    start_worker("row-recognizer", jobs, RowRecognizer(reader.read_rows, inbox, min_score).submit)
    def on_mail():  # az ébresztés és a rendes időzítő is ezt hívja
        mail = inbox.drain()
        changed = pipeline.receive(mail.snapshot, mail.texts, time.monotonic())
"""

from __future__ import annotations

import logging
import threading
import tkinter
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Generic, TypeVar

import numpy as np

from .i18n import t
from .ocr import Box
from .speaker_color import sample_speaker_colors
from .row_tracker import (EMPTY_MAIL, RECOGNIZER, STABLE_CAPTURES, WATCHER, BandFrame, BandSnapshot, Mail, RowFinder,
                          RowText, RowTracker, WorkerFailed)

# Ennyi rögzítésenként a felismerő akkor is megkapja a képet, ha nincs új sor: így próbálja újra, amit elsőre
# nem tudott elolvasni (másodpercenként 30 rögzítésnél ~0,27 mp-enként; becsült érték, nem mért)
RETRY_CAPTURES = 8
# Egy olvashatatlan sort legfeljebb ennyiszer olvasunk: mozgó háttér előtt másodszorra sikerülhet, de ami nem
# szöveg, arra ne menjen el a processzoridő (becsült érték, nem mért)
READ_ATTEMPTS = 4

ItemT = TypeVar("ItemT")
ReadRows = Callable[[np.ndarray, Box, list[Box]], Sequence[tuple[str, float]]]

logger = logging.getLogger(__name__)


class Inbox:
    """Munkaszálak → Tk-szál. Pillanatképből csak a legfrissebb marad (nem torlódhat fel, ha a Tk-szál
    elfoglalt); a felismert szövegek és a hibajelzések mind megmaradnak."""

    def __init__(self, wake: Callable[[], None]) -> None:
        """wake: szálbiztos, nem blokkoló jelzés a Tk-szálnak (TkWaker.wake)."""
        self._wake = wake
        self._lock = threading.Lock()
        self._mail = EMPTY_MAIL

    def put_snapshot(self, snapshot: BandSnapshot, *, urgent: bool) -> None:
        """urgent: változtak a sorok, a Tk-szálnak rögtön látnia kell; különben ráér a rendes időzítőjével."""
        with self._lock:
            self._mail = replace(self._mail, snapshot=snapshot)
        if urgent:
            self._wake()

    def put_texts(self, texts: Sequence[RowText]) -> None:
        with self._lock:
            self._mail = replace(self._mail, texts=self._mail.texts + tuple(texts))
        self._wake()

    def put_failure(self, failure: WorkerFailed) -> None:
        with self._lock:
            self._mail = replace(self._mail, failures=self._mail.failures + (failure,))
        self._wake()

    def drain(self) -> Mail:
        """A Tk-szál hívja: minden, ami az előző ürítés óta érkezett (EMPTY_MAIL, ha semmi)."""
        with self._lock:
            mail, self._mail = self._mail, EMPTY_MAIL
        return mail


class LatestBox(Generic[ItemT]):
    """Egyelemű postaláda egy munkaszálnak: ha a szál lemaradt, csak a legfrissebb feladat marad."""

    def __init__(self) -> None:
        self._changed = threading.Condition()
        self._item: ItemT | None = None
        self._is_closed = False

    def put(self, item: ItemT) -> None:
        with self._changed:
            self._item = item
            self._changed.notify()

    def take(self) -> ItemT | None:
        """Megvárja a következő feladatot; None, ha a postaládát lezárták (a munkaszál ekkor kilép)."""
        with self._changed:
            while self._item is None and not self._is_closed:
                self._changed.wait()
            if self._is_closed:
                return None
            item, self._item = self._item, None
            return item

    def close(self) -> None:
        with self._changed:
            self._is_closed = True
            self._changed.notify_all()


def start_worker(name: str, box: LatestBox[ItemT], handle: Callable[[ItemT], None]) -> threading.Thread:
    """Vékony (démon) szálhurok indítása: a postaláda feladatait egyenként adja a feldolgozónak, amíg a
    postaládát le nem zárják. A feldolgozó ne dobjon kivételt (a két submit metódus ilyen)."""
    def run() -> None:
        while (item := box.take()) is not None:
            handle(item)

    thread = threading.Thread(target=run, name=name, daemon=True)
    thread.start()
    return thread


def tk_scheduler(interp: tkinter.Misc, callback: Callable[[], None]) -> Callable[[], None]:
    """Ütemező a TkWaker-hez: hívására a callback lefut a Tk-szál eseményhurkában, amint az ráér.

    A Tk-szálon kell létrehozni (parancsot regisztrál). A visszaadott függvényt más szál hívja: megvárja,
    amíg a Tk-szál átveszi a kérést; futó mainloop nélkül ~1 mp múlva RuntimeError-t ad (a TkWaker kezeli).
    """
    command = interp.register(callback)
    return lambda: interp.tk.call("after", 0, command)


class TkWaker:
    """wake(): a Tk-szál hamarosan lefuttatja a callbackjét. Bármely szálból hívható, és sosem blokkol.

    A tkinter más szálból jövő hívása megvárja, amíg a Tk-szál eseményhurka átveszi, ezért azt egy
    külön ébresztő szál végzi: a munkaszálak sosem várnak a kirajzolásra. Több jelzés egy ébresztéssé
    olvad össze. Ha nincs futó mainloop (indulás, leállás), az ébresztés elmarad: a hívó rendes
    időzítője úgyis kiüríti a postaládát.
    """

    def __init__(self, schedule: Callable[[], None]) -> None:
        """schedule: a tényleges ütemezés (tk_scheduler(...)); az ébresztő szálon fut."""
        self._schedule = schedule
        self._signal = threading.Event()
        self._is_closed = False
        self._thread = threading.Thread(target=self._run, name="tk-waker", daemon=True)
        self._thread.start()

    def wake(self) -> None:
        self._signal.set()

    def close(self) -> None:
        """Nem lesz több ébresztés. A Tk-szálon, a gyökérablak lebontása előtt hívandó."""
        self._is_closed = True
        self._signal.set()

    def join(self, timeout: float) -> None:
        """Megvárja az ébresztő szál végét (close() után)."""
        self._thread.join(timeout)

    def _run(self) -> None:
        while True:
            self._signal.wait()
            if self._is_closed:
                return
            self._signal.clear()
            try:
                self._schedule()
            except RuntimeError as err:  # nincs futó mainloop: majd a rendes időzítő ürít
                logger.debug("Tk wake skipped, no main loop is running: %s", err)
            except tkinter.TclError as err:  # az alkalmazást már lebontották: nincs kit ébreszteni
                logger.debug("Tk waker stops, the application is gone: %s", err)
                return


class _FailureReport:
    """Egy munkafolyamat hibáinak jelentése: hibasorozatból csak az elsőt (ne teljen meg vele a napló)."""

    def __init__(self, worker: str, inbox: Inbox) -> None:
        self._worker = worker
        self._inbox = inbox
        self._is_failing = False

    def failed(self, err: Exception) -> None:
        """Kivételkezelőből hívandó: a napló így a hívási láncot is megkapja."""
        if not self._is_failing:
            logger.exception(t("pipeline.ocr_error"))
            self._inbox.put_failure(WorkerFailed(self._worker, repr(err)))
        self._is_failing = True

    def succeeded(self) -> None:
        self._is_failing = False


class BandWatcher:
    """Egy rögzített sáv feldolgozása: sorkeresés, sorkövetés, és jelzés annak, akinek dolga van vele."""

    def __init__(self, finder: RowFinder, inbox: Inbox, jobs: LatestBox[BandSnapshot]) -> None:
        """jobs: a felismerő postaládája (RowRecognizer.submit egy start_worker szálon)."""
        self._detect = finder.detect
        self._tracker = RowTracker(finder.is_same_row)
        self._inbox = inbox
        self._jobs = jobs
        self._geometry: tuple[Box, tuple[int, int]] | None = None
        self._marker: tuple[tuple[int, Box, bool, int], ...] | None = None  # None: a következő kép biztosan sürgős
        self._visible: frozenset[int] = frozenset()
        self._captures_since_job = 0
        self._failures = _FailureReport(WATCHER, inbox)

    def submit(self, frame: BandFrame) -> None:
        """Munkaszál-határ (a rögzítő szál vagy egy start_worker szál hívja): hibánál jelent, de sosem dob kivételt."""
        try:
            self.process(frame)
        except Exception as err:  # a sorkereső hibája nem állíthatja meg a hívó szálat
            self._failures.failed(err)
        else:
            self._failures.succeeded()

    def process(self, frame: BandFrame) -> BandSnapshot:
        """Egy sáv → pillanatkép a Tk-szálnak, és munka a felismerőnek, ha új sor látszik."""
        geometry = (frame.area, frame.frame_size)
        if geometry != self._geometry:  # átméretezték az ablakot, vagy más a figyelt sáv: a régi sorok érvénytelenek
            self._geometry, self._marker, self._visible = geometry, None, frozenset()
            self._tracker.reset()
        rows = self._tracker.update(self._detect(frame.image, frame.area, frame.frame_size), frame.captured_at)
        snapshot = BandSnapshot(frame.image, frame.area, frame.frame_size, frame.captured_at, rows)
        marker = tuple((row.row_id, row.box, row.sure, row.missing) for row in rows)
        is_settling = any(row.stable_captures <= STABLE_CAPTURES for row in rows)
        # Előbb a pillanatkép, utána a felismerés: a Tk-oldalon a sornak előbb kell léteznie, mint a szövegének
        self._inbox.put_snapshot(snapshot, urgent=marker != self._marker or is_settling)
        visible = frozenset(row.row_id for row in rows if row.missing == 0)
        self._captures_since_job += 1
        # Új vagy visszatért sor: a felismerő megnézi, olvasta-e már; időnként akkor is, ha nincs ilyen (újrapróbálás)
        if visible - self._visible or (visible and self._captures_since_job >= RETRY_CAPTURES):
            self._jobs.put(snapshot)
            self._captures_since_job = 0
        self._marker, self._visible = marker, visible
        return snapshot


class RowRecognizer:
    """A még olvasatlan sorok felismertetése, mindig a legfrissebb pillanatképről."""

    def __init__(self, read_rows: ReadRows, inbox: Inbox, min_score: float) -> None:
        """read_rows: a SubtitleReader.read_rows alakja: (sáv, a sáv helye, sordobozok) → (szöveg, bizonyosság).
        min_score: az ennél bizonytalanabb (vagy üres) olvasatot később újra megpróbáljuk (config.min_text_score)."""
        self._read_rows = read_rows
        self._inbox = inbox
        self._min_score = min_score
        self._reads: dict[int, int] = {}  # sor → hány olvasás volt; olvasható eredmény után rögtön READ_ATTEMPTS
        self._failures = _FailureReport(RECOGNIZER, inbox)
        self.recognitions = 0  # ennyi sorolvasás volt eddig (méréshez)

    def submit(self, snapshot: BandSnapshot) -> None:
        """Munkaszál-határ (a start_worker szála hívja): hibánál jelent, de sosem dob kivételt."""
        try:
            texts = self.recognise(snapshot)
        except Exception as err:  # az OCR hibája nem állíthatja meg a szálat; a sorok olvasatlanok maradnak
            self._failures.failed(err)
            return
        self._failures.succeeded()
        if texts:
            self._inbox.put_texts(texts)

    def recognise(self, snapshot: BandSnapshot) -> tuple[RowText, ...]:
        """A pillanatkép látható sorai közül azoknak a szövege, amelyeket még (vagy újra) olvasni kell."""
        reads = {row.row_id: self._reads.get(row.row_id, 0) for row in snapshot.rows}  # az eltűnt sorok kimaradnak
        todo = [row for row in snapshot.rows if row.missing == 0 and reads[row.row_id] < READ_ATTEMPTS]
        results = self._read_rows(snapshot.image, snapshot.area, [row.box for row in todo]) if todo else []
        texts = tuple(RowText(row.row_id, text, score, speaker_colors=sample_speaker_colors(
                          snapshot.image, snapshot.area, text, row.box))
                      for row, (text, score) in zip(todo, results, strict=True))
        done = {item.row_id: READ_ATTEMPTS if item.is_legible(self._min_score) else reads[item.row_id] + 1
                for item in texts}
        self._reads = {**reads, **done}
        self.recognitions += len(todo)
        return texts
