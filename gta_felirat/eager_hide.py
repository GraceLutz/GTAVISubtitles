"""Új angol feliratsorok azonnali eltakarása, még az olvasás előtt.

Az olvasás (szövegkeresés + felismerés) 30–140 ms; addig az új angol sor látszana. A klasszikus
sorkereső (row_detector, ~1–3 ms) viszont már a rögzítés után megmondja, hol áll biztosan
feliratsor. Ezeket rövid időre (EAGER_HIDE_SECONDS) rögtön eltakarjuk; mire letelik, az olvasás
eredménye megvan, és a szokásos szabályok döntenek (felirat-e, van-e fordítása). Ami nem
felirat, az a rövid idő után újra látszik, és amíg a helyén marad, többet nem takarjuk.

Amit viszont a feldolgozás már feliratként takart, az rövid ideig (CONFIRMED_HOLD_SECONDS) akkor
is takarva marad, ha egy-egy olvasás kihagyja (pl. egy kétsoros mondat második sorát): különben
az angol sor fel-felvillanna a fordítás alatt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np

from .config import Region
from .frosted_cover import Backdrop
from .layer_plan import Shown
from .ocr import Box, frame_center
from .row_detector import detect_rows

EAGER_HIDE_SECONDS = 0.35  # olvasás előtt legfeljebb eddig takarunk egy új sort
CONFIRMED_HOLD_SECONDS = 1.0  # a feldolgozás által már takart sort eddig tartjuk takarva, ha kimarad egy olvasásból
SAME_ROW_TOLERANCE_PX = 8  # ennyin belül ugyanott álló doboz ugyanaz a sor (a kereső dobozai 1–5 px-t ingadoznak)
MIN_COVERED_RATIO = 0.5  # ha a sor magasságának ekkora részét már takarja a feldolgozás, nem kell külön takarni

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeenRow:
    box: Box  # a teljes kép koordinátáiban
    first_seen: float  # annak a rögzítésnek az ideje, amelyiken először látszott
    covered_at: float | None = None  # mikor takarta utoljára a feldolgozás (vagyis az olvasás szerint felirat)


def sure_rows(band: np.ndarray, area: Box, frame_width: float, dialogue_bottom: float) -> tuple[Box, ...]:
    """A sávban biztosan feliratsornak látszó sorok dobozai a teljes képen (a párbeszédsorok zónájában).

    dialogue_bottom: a párbeszédsorok alsó határa a teljes képen; alatta a küldetésutasítás sora áll.
    """
    rows = detect_rows(band, area.left, frame_width, dialogue_bottom=dialogue_bottom - area.top)
    return tuple(Box(row.left + area.left, row.top + area.top, row.right + area.left, row.bottom + area.top)
                 for row in rows if row.sure)


class EagerHide:
    """Számon tartja, melyik sor mióta látszik, és megmondja, melyiket kell még takarni."""

    def __init__(self) -> None:
        self._rows: tuple[SeenRow, ...] = ()

    def observe(self, boxes: tuple[Box, ...], captured_at: float) -> None:
        """Egy rögzítés biztos sorai: az ugyanott álló sor megtartja, amit eddig tudtunk róla."""
        self._rows = tuple(self._carried_over(box, captured_at) for box in boxes)

    def confirm(self, shown: Shown | None, now: float) -> None:
        """Megjegyzi, mely sorokat takarja most a feldolgozás: azokról már tudjuk, hogy feliratsorok.

        Ha a feldolgozás szerint nincs felirat (shown None), a tartás is megszűnik: az nem kimaradt olvasás.
        """
        if shown is None:
            self._rows = tuple(replace(row, covered_at=None) for row in self._rows)
            return
        # Csak a lefordított sor erősít meg: amit a feldolgozás csak takar (shown.hidden), azt szándékosan
        # engedi el, ha késik a fordítása, és azt nem szabad tovább tartani
        anchors = _translated_anchors(shown)
        self._rows = tuple(replace(row, covered_at=now) if _is_under(row.box, anchors) else row
                           for row in self._rows)

    def active(self, now: float) -> tuple[Box, ...]:
        """A takarandó sorok: az újak (EAGER_HIDE_SECONDS-ig) és a nemrég még feliratként takartak."""
        return tuple(row.box for row in self._rows if _is_new(row, now) or _is_held(row, now))

    def reset(self) -> None:
        self._rows = ()

    def _carried_over(self, box: Box, captured_at: float) -> SeenRow:
        for row in self._rows:
            if _is_same_place(row.box, box):
                return replace(row, box=box)
        return SeenRow(box, captured_at)


def _is_new(row: SeenRow, now: float) -> bool:
    return now - row.first_seen < EAGER_HIDE_SECONDS


def _is_held(row: SeenRow, now: float) -> bool:
    return row.covered_at is not None and now - row.covered_at < CONFIRMED_HOLD_SECONDS


@dataclass(frozen=True, eq=False)
class _Capture:
    """Egy rögzítés minden adata egyben: a megjelenítési szál sosem lát új hátteret régi sorokkal."""

    backdrop: Backdrop  # ebből készül az angol szöveg elmosott takarása
    rows: tuple[Box, ...]  # a biztos feliratsorok ezen a sávon
    captured_at: float


class EagerCover:
    """Híd a rögzítő és a megjelenítési szál között: az egyik közzéteszi egy rögzítés sávját és biztos sorait,
    a másik megkérdezi, mi a takarás háttere, és mit kell most a feldolgozástól függetlenül takarni.

    A két szál egyetlen hivatkozás cseréjével ad át adatot, ezért nincs szükség zárra.
    """

    def __init__(self) -> None:
        self._eager = EagerHide()
        self._published: _Capture | None = None  # a rögzítő (vagy az OCR-) szál írja
        self._applied: _Capture | None = None
        self._is_broken = False

    def publish(self, band: np.ndarray, area: Box, region: Region | None, captured_at: float) -> None:
        """A rögzítés után rögtön, még az olvasás előtt: a sáv és a rajta talált biztos sorok (~1–3 ms).

        region: a beállított feliratsáv (ebből tudjuk a kép közepét és a párbeszédsorok alsó határát);
        None: nem keresünk sorokat (az angol felirat takarása ki van kapcsolva).
        Ha a sorkereső hibázik, az azonnali takarás kikapcsol: ez csak gyorsítás, a szokásos út megy tovább.
        """
        rows = () if region is None or self._is_broken else self._sure_rows(band, area, region)
        self._published = _Capture(Backdrop(band, area), rows, captured_at)

    def _sure_rows(self, band: np.ndarray, area: Box, region: Region) -> tuple[Box, ...]:
        try:
            center = frame_center(region, area)
            return sure_rows(band, area, center.frame_width, center.dialogue_bottom)
        except Exception:
            self._is_broken = True
            logger.exception("Eager hide is switched off: the row detector failed")
            return ()  # a korábban látott sorokat sem takarjuk tovább

    def view(self, shown: Shown | None, now: float) -> tuple[Backdrop | None, tuple[Box, ...]]:
        """A megjelenítési szálon: (a takarás háttere, a most takarandó sorok). A sorok: az újak, és amit a
        feldolgozás (shown) nemrég még feliratként takart."""
        published = self._published
        if published is None:
            return None, ()
        if published is not self._applied:
            self._applied = published
            self._eager.observe(published.rows, published.captured_at)
        self._eager.confirm(shown, now)
        return published.backdrop, self._eager.active(now)


def _is_same_place(first: Box, second: Box) -> bool:
    edges = ((first.left, second.left), (first.top, second.top),
             (first.right, second.right), (first.bottom, second.bottom))
    return all(abs(a - b) <= SAME_ROW_TOLERANCE_PX for a, b in edges)


def with_eager_boxes(shown: Shown | None, boxes: tuple[Box, ...]) -> Shown | None:
    """A kiírandó állapot kiegészítve azokkal a sorokkal, amelyeket a feldolgozás még nem takar."""
    extra = tuple(box for box in boxes if not _is_covered(shown, box))
    if not extra:
        return shown
    if shown is None:
        return Shown("", _union(extra), (), extra)
    return replace(shown, hidden=shown.hidden + extra)


def _is_covered(shown: Shown | None, box: Box) -> bool:
    if shown is None:
        return False
    return _is_under(box, _translated_anchors(shown) + list(shown.hidden))


def _translated_anchors(shown: Shown) -> list[Box]:
    if not shown.lines and shown.text:
        return [shown.anchor]  # egy tömbben kiírt szöveg: az egész angol felirat helyét takarja
    return [line.anchor for line in shown.lines]


def _is_under(box: Box, covers: list[Box]) -> bool:
    return any(_spans(other, box) and _overlap_ratio(box, other) >= MIN_COVERED_RATIO for other in covers)


def _spans(other: Box, box: Box) -> bool:
    """A másik doboz vízszintesen is végig takarja-e a sort (egy szélesebb új sort a régi, keskenyebb nem)."""
    return other.left <= box.left + SAME_ROW_TOLERANCE_PX and other.right >= box.right - SAME_ROW_TOLERANCE_PX


def _overlap_ratio(box: Box, other: Box) -> float:
    """A doboz magasságának mekkora részét fedi a másik (ha vízszintesen is átfednek)."""
    if min(box.right, other.right) <= max(box.left, other.left) or box.height <= 0:
        return 0.0
    return max(0, min(box.bottom, other.bottom) - max(box.top, other.top)) / box.height


def _union(boxes: tuple[Box, ...]) -> Box:
    return Box(min(b.left for b in boxes), min(b.top for b in boxes),
               max(b.right for b in boxes), max(b.bottom for b in boxes))
