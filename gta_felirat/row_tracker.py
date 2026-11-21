"""Feliratsorok követése a képpontjaik alapján: a soralapú út adatai és a sorkövető.

A feliratsávról minden rögzítés után egy gyors (klasszikus) sorkereső fut. Ez a modul mondja meg,
hogy egy talált sor ugyanaz-e, mint az előző rögzítésen (akkor is, ha a játék közben feljebb
görgette), vagy új. Új sor = új azonosító: csak azt kell felismertetni (OCR), a többi megtartja
a szövegét és a fordítását. Ha a rögzítések között hosszabb szünet volt, minden sor újnak számít.

A sorkereső és az aláírás külön modulok dolga; ez a modul nem ismeri őket. Az aláírás itt
átlátszatlan érték: két aláírást a RowTracker-nek átadott is_same_row függvény hasonlít össze.

Szálak: a RowTracker-t egyetlen szál használja (az, amelyik a sávokat feldolgozza, lásd
handoff.BandWatcher), ezért nincs benne zár. Az itt leírt adatok mind megváltoztathatatlanok:
szabadon utazhatnak a rögzítő, a felismerő és a megjelenítési (Tk) szál között.
A sáv képét (BandFrame.image) a rögzítés után senki sem írhatja.

Idő: minden időpont ugyanabból az órából való, mint a pipeline `now` értéke (time.monotonic()).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .ocr import Box

GONE_GRACE_CAPTURES = 2  # ennyi rögzítésről hiányozhat egy sor (egy-két rossz képkocka), mielőtt eltűntnek vesszük
SETTLE_SECONDS = 0.3  # ennél fiatalabb sor „cseréje” még ugyanannak a sornak a megjelenése (áttűnés)
STABLE_CAPTURES = 2  # ennyi egyező rögzítés után stabilak a sor képpontjai (második OCR-olvasás helyett)
# Ennél hosszabb szünet után (pl. közben a régi úton olvastunk) a korábbi sorokban nem bízunk: ami a szünetben
# történt, nem láttuk, ezért minden sor újnak számít, és újraolvassuk (becsült érték: a felirat ~0,8 mp-ig marad)
MAX_CAPTURE_GAP_SECONDS = 1.0
# Két doboz „ugyanaz a hely”, ha a közös részük a kettő együttes területének legalább ekkora része.
# Becsült érték: ugyanaz a sor ±3 pixeles ingadozással ~0,85; más hosszúságú (középre igazított) szöveg kevesebb.
SAME_PLACE_OVERLAP = 0.8
WATCHER = "watcher"  # WorkerFailed.worker: a sorkereső hibázott
RECOGNIZER = "recognizer"  # WorkerFailed.worker: a sorok olvasása (OCR) hibázott


@dataclass(frozen=True)
class DetectedRow:
    """A sorkereső kimenete egy sorra, egy rögzítésen."""

    box: Box  # a teljes kép koordinátáiban, a betűk körül kis margóval
    signature: Any  # átlátszatlan: csak az is_same_row függvény érti
    whiteness: float  # a feliratszínű pixelek aránya a dobozban (fragment_filter.white_fraction)
    sure: bool = True  # a szigorú ellenőrzéseken is átment: olvasás előtt is eltakarható


@dataclass(frozen=True)
class RowFinder:
    """A sorkereső két függvénye (a sorkereső és az aláírás modulja köré írt illesztő adja).

    Mindkettő azon a szálon fut, amelyik a sávokat feldolgozza; a képet nem módosíthatják.
    """

    detect: Callable[[np.ndarray, Box, tuple[int, int]], Sequence[DetectedRow]]  # (sáv, a helye, képméret)
    is_same_row: Callable[[Any, Any], bool]  # (régi aláírás, új aláírás): ugyanazok a képpontok-e


@dataclass(frozen=True)
class RowObs:
    """Egy képernyőn lévő sor; a row_id addig ugyanaz, amíg ugyanazok a képpontok látszanak."""

    row_id: int
    box: Box
    whiteness: float
    first_seen: float  # annak a rögzítésnek az ideje, amelyiken megjelent
    stable_captures: int  # ennyi rögzítésen látszott már ugyanazokkal a képpontokkal
    sure: bool
    missing: int = 0  # ennyi rögzítés óta nem látszik (türelmi idő); 0: most is a képen van


@dataclass(frozen=True, eq=False)  # eq=False: a kép numpy-tömbjének == összehasonlítása hibát dobna
class BandFrame:
    """Egy rögzített feliratsáv, ahogy a rögzítő száltól kapjuk (a band_feed.CapturedBand ugyanilyen mezőkkel adja)."""

    image: np.ndarray  # BGR; ebből olvas a felismerő, és ebből készül a takarás háttere
    area: Box  # a sáv helye a teljes képen
    frame_size: tuple[int, int]  # a teljes kép (az ablak tartalmának) mérete: a dobozok ehhez tartoznak
    captured_at: float


@dataclass(frozen=True, eq=False)
class BandSnapshot(BandFrame):
    """Minden, ami EGY rögzítésből származik: a kép, és a sorok, amelyeket rajta találtunk."""

    rows: tuple[RowObs, ...] = ()  # fentről lefelé


@dataclass(frozen=True)
class RowText:
    """Egy sor szövege: a felismerő (OCR-szál) eredménye."""

    row_id: int
    text: str  # üres, ha nincs olvasható szöveg
    score: float
    provisional: bool = False  # még nem olvastuk: attól a sortól örökölte, amelyiknek pontosan a helyére került

    def is_legible(self, min_score: float) -> bool:
        """Olvasható-e: van szövege, és az OCR elég biztos benne."""
        return bool(self.text.strip()) and self.score >= min_score


@dataclass(frozen=True)
class WorkerFailed:
    """Egy munkafolyamat hibázott (a részletek a naplóban); a hívó dönthet úgy, hogy a régi útra vált."""

    worker: str  # WATCHER vagy RECOGNIZER
    error: str


@dataclass(frozen=True)
class Mail:
    """Amit a Tk-szál a postaláda (handoff.Inbox) egy ürítésekor kap."""

    snapshot: BandSnapshot | None = None  # a legfrissebb pillanatkép, ha jött új
    texts: tuple[RowText, ...] = ()  # az azóta felismert sorok, érkezési sorrendben
    failures: tuple[WorkerFailed, ...] = ()


EMPTY_MAIL = Mail()


def overlaps(first: Box, second: Box) -> bool:
    """Van-e közös területe a két doboznak."""
    return (min(first.right, second.right) > max(first.left, second.left)
            and min(first.bottom, second.bottom) > max(first.top, second.top))


def is_same_place(first: Box, second: Box) -> bool:
    """Szinte teljesen fedi-e egymást a két doboz: ugyanaz a képernyősor, legfeljebb néhány pixel eltéréssel."""
    width = min(first.right, second.right) - max(first.left, second.left)
    height = min(first.bottom, second.bottom) - max(first.top, second.top)
    if width <= 0 or height <= 0:
        return False
    shared = width * height
    combined = first.width * first.height + second.width * second.height - shared
    return shared >= SAME_PLACE_OVERLAP * combined


@dataclass(frozen=True, eq=False)  # eq=False: az aláírás összehasonlítása nem a mi dolgunk (és numpy-tömb is lehet)
class _Track:
    obs: RowObs
    signature: Any


class RowTracker:
    """Megmondja, melyik talált sor ugyanaz, mint eddig, és melyik új. Egyetlen szál használja."""

    def __init__(self, is_same_row: Callable[[Any, Any], bool]) -> None:
        """is_same_row(régi aláírás, új aláírás): ugyanazok a képpontok-e (akár más sorhelyen)."""
        self._is_same_row = is_same_row
        self._tracks: tuple[_Track, ...] = ()
        self._ids = itertools.count(1)
        self._updated_at = float("-inf")

    def reset(self) -> None:
        """Új ablakméret vagy sáv: a régi sorok nem érvényesek. Az azonosítók ezután sem ismétlődnek."""
        self._tracks = ()

    def update(self, detected: Sequence[DetectedRow], now: float) -> tuple[RowObs, ...]:
        """Egy rögzítés sorai → a képernyőn lévő (és a türelmi idejüket töltő) sorok, fentről lefelé."""
        if now - self._updated_at > MAX_CAPTURE_GAP_SECONDS:
            self.reset()  # hosszú szünet után a régi sorok már nem érvényesek
        self._updated_at = now
        matches = self._match(detected)
        continued = {track.obs.row_id for track in matches if track is not None}
        orphans = [track for track in self._tracks if track.obs.row_id not in continued]
        tracks = [_continue(track, row) if track is not None else self._start(row, orphans, now)
                  for row, track in zip(detected, matches)]
        tracks += _waiting(orphans, tracks)
        self._tracks = tuple(sorted(tracks, key=lambda track: track.obs.box.top))
        return tuple(track.obs for track in self._tracks)

    def _match(self, detected: Sequence[DetectedRow]) -> list[_Track | None]:
        """Minden talált sorhoz az a követett sor, amelyiknek ugyanazok a képpontjai (mindegyik csak egyszer)."""
        free = list(self._tracks)
        matches: list[_Track | None] = []
        for row in detected:
            track = next((candidate for candidate in free
                          if self._is_same_row(candidate.signature, row.signature)), None)
            if track is not None:
                free.remove(track)
            matches.append(track)
        return matches

    def _start(self, row: DetectedRow, orphans: Sequence[_Track], now: float) -> _Track:
        obs = RowObs(next(self._ids), row.box, row.whiteness, _first_seen(row, orphans, now), 1, row.sure)
        return _Track(obs, row.signature)


def _continue(track: _Track, row: DetectedRow) -> _Track:
    """Ugyanaz a sor a mostani rögzítésen: a helye változhatott, a szövege nem."""
    old = track.obs
    obs = replace(old, box=row.box, whiteness=row.whiteness, sure=old.sure or row.sure,
                  stable_captures=old.stable_captures + 1, missing=0)
    return _Track(obs, row.signature)


def _waiting(orphans: Sequence[_Track], tracks: Sequence[_Track]) -> list[_Track]:
    """A most nem látszó sorok rövid türelmi időt kapnak, hacsak más sor nem került a helyükre."""
    return [_Track(replace(track.obs, missing=track.obs.missing + 1), track.signature) for track in orphans
            if track.obs.missing < GONE_GRACE_CAPTURES
            and not any(overlaps(track.obs.box, other.obs.box) for other in tracks)]


def _first_seen(row: DetectedRow, orphans: Sequence[_Track], now: float) -> float:
    """Az éppen megjelenő (áttűnő) sor aláírása még változhat: ilyenkor az eltakarás órája nem indul újra."""
    for track in orphans:
        if now - track.obs.first_seen < SETTLE_SECONDS and overlaps(track.obs.box, row.box):
            return track.obs.first_seen
    return now
