"""A késleltetett tiszta kép alapja: a játékképek és a feliratállapotok rövid története, időbélyeggel.

A feliratréteg mindig kicsit lemarad a játéktól: az új angol sort csak a felismerése után takarja, és a fordítás
is idő. A tiszta kép ezért régebbi képkockát mutat, és arra azt az állapotot rajzolja, amelyik a képkockán látható
sorokat fedi, lehetőleg kész fordítással (pick_shown).

A képkocka sorait a tiszta kép maga keresi meg (row_finder): így a takarás oda kerül, ahol a sor azon a képen
áll, akkor is, ha a játék közben feljebb görgette (új beszélő), vagy a következő felirat már ott áll. A
legutolsó állapot erre nem jó: az többnyire már az átmenet pillanata, a következő sor takarásával.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from .layer_plan import Shown, ShownLine

if TYPE_CHECKING:
    import numpy as np

    from .game_box import GameBox
    from .ocr import Box

SETTLE_SECONDS = 0.4  # ennyi idő alatt a pipeline biztosan felismeri és megerősíti a képen megjelent feliratot
HISTORY_SLACK_SECONDS = 1.0  # a késleltetésen túl ennyi régebbi állapotot is megtartunk
MIN_COVER_RATIO = 0.5  # a sor magasságának és szélességének ekkora részét fedje a takarás vagy a fordítás
# Egy fordítás csak hasonló szélességű sorhoz tartozhat: ugyanott a következő, más hosszú felirat már nem az övé.
# A képkocka sora a teljes szöveg szélessége plusz ráhagyás, az angol sor helye a felismert szöveg, ezért a tűrés.
WIDTH_TOLERANCE = 0.2
WIDTH_SLACK_PX = 30


@dataclass(frozen=True)
class FrameRow:
    """A képkockán talált sor."""
    box: Box
    # A sorkereső biztos benne, hogy feliratsor: akkor is takarjuk, ha egyik állapot sem fedi. A bizonytalan
    # sort (pl. rendszámtábla) csak akkor, ha valamelyik közeli állapot fordította vagy takarta
    sure: bool = True
    game_box: GameBox | None = None  # a játék sötét feliratdoboza a sor mögött (game_box)


@dataclass(frozen=True)
class ShownEntry:
    at: float  # time.monotonic()
    shown: Shown | None
    source: str  # a pipeline aktuális angol szövege (a felirat azonosítója); "" ha nincs felirat


class ShownHistory:
    """A kiírt állapotok időrendben; a Tk-szál írja, a tiszta kép szála olvassa."""

    def __init__(self, keep_seconds: float) -> None:
        self._keep_seconds = keep_seconds
        self._entries: deque[ShownEntry] = deque()
        self._lock = threading.Lock()

    def record(self, at: float, shown: Shown | None, source: str) -> None:
        with self._lock:
            last = self._entries[-1] if self._entries else None
            if last is not None and last.shown == shown and last.source == source:
                return
            self._entries.append(ShownEntry(at, shown, source))
            while len(self._entries) > 1 and self._entries[1].at < at - self._keep_seconds:
                self._entries.popleft()  # az utolsó, még érvényes régi állapot marad

    def between(self, start: float, end: float) -> list[ShownEntry]:
        """A start pillanatában érvényes állapot és az end-ig követő többi, időrendben."""
        with self._lock:
            entries = list(self._entries)
        first = next((i for i in range(len(entries) - 1, -1, -1) if entries[i].at <= start), 0)
        return [entry for entry in entries[first:] if entry.at <= end]

    def for_frame(self, frame_at: float, until: float, rows: list[FrameRow] | None) -> Shown | None:
        """A képkockára rajzolandó állapot. rows: a képkockán talált sorok (üres vagy None:
        nem találtunk / nem kerestünk; ilyenkor csak az idő szerint dönthetünk, mint az élő réteg)."""
        entries = self.between(frame_at, until)
        if not rows:
            return _in_force(entries, frame_at + SETTLE_SECONDS)
        source = _source_at(entries, frame_at + SETTLE_SECONDS)
        return pick_shown([entry for entry in entries if entry.shown is not None], rows, source)


def _in_force(entries: list[ShownEntry], moment: float) -> Shown | None:
    found = [entry for entry in entries if entry.at <= moment]
    return found[-1].shown if found else None


def _source_at(entries: list[ShownEntry], moment: float) -> str:
    found = [entry for entry in entries if entry.at <= moment]
    return found[-1].source if found else ""


def is_covered(row: Box, boxes: list[Box]) -> bool:
    for box in boxes:
        height = min(row.bottom, box.bottom) - max(row.top, box.top)
        width = min(row.right, box.right) - max(row.left, box.left)
        if height >= MIN_COVER_RATIO * row.height and width >= MIN_COVER_RATIO * row.width:
            return True
    return False


def is_same_line(row: Box, anchor: Box) -> bool:
    """A fordítás angol sora (anchor) lehet-e a képkocka sora: fedi, és hasonló széles."""
    widest = max(row.width, anchor.width)
    return is_covered(row, [anchor]) and abs(row.width - anchor.width) <= WIDTH_TOLERANCE * widest + WIDTH_SLACK_PX


def _translates(shown: Shown, row: Box) -> bool:
    if shown.lines:
        return any(is_same_line(row, line.anchor) for line in shown.lines)
    return is_covered(row, text_boxes(shown))


def text_boxes(shown: Shown) -> list[Box]:
    """Ahol fordítás áll: soronként, vagy egy tömbben az eredeti felirat helyén."""
    if shown.lines:
        return [line.anchor for line in shown.lines]
    return [shown.anchor] if shown.text else []


def _score(entry: ShownEntry, rows: list[Box], source: str, order: int) -> tuple:
    shown = entry.shown
    anchors = text_boxes(shown)
    translated_rows = [_translates(shown, row) for row in rows]
    missing = sum(not (translates or is_covered(row, list(shown.hidden)))
                  for row, translates in zip(rows, translated_rows))
    translated = sum(translated_rows)
    stray = sum(not any(is_covered(row, [anchor]) or is_covered(anchor, [row]) for row in rows)
                for anchor in anchors)  # fordítás ott, ahol a képen nincs sor
    # Elöl a fedés, aztán hogy ugyanaz a felirat-e, aztán a fordítás; egyenlőségnél a korábbi (a következő
    # felirat ugyanazon a helyen később áll be)
    return (-missing, entry.source == source, translated, -stray, -order)


def _without_lines_of_other_rows(shown: Shown, rows: list[Box]) -> Shown | None:
    """Elhagyja azokat a fordított sorokat, amelyek helyén a képkockán egy más (eltérő szélességű) sor áll."""
    def belongs_elsewhere(line: ShownLine) -> bool:
        return any(is_covered(row, [line.anchor]) and not is_same_line(row, line.anchor) for row in rows)

    kept = tuple(line for line in shown.lines if not belongs_elsewhere(line))
    if kept == shown.lines:
        return shown
    if not kept and not shown.hidden:
        return None
    anchor = _union([line.anchor for line in kept] or list(shown.hidden))
    return replace(shown, text="\n".join(line.text for line in kept), anchor=anchor, lines=kept)


def pick_shown(entries: list[ShownEntry], rows: list[FrameRow], source: str) -> Shown | None:
    """Az az állapot, amelyik a képkocka sorait fedi, lehetőleg fordítással; a fedetlen feliratsorokra
    takarás kerül."""
    found = [row.box for row in rows]
    if not source:
        # A felismerés még nem ért utol: annak a feliratnak a fordítása kell, amelyik elsőként fordítja a sort
        source = next((entry.source for entry in entries
                       if any(_translates(entry.shown, row) for row in found)), "")
    best = max(entries, key=lambda entry: _score(entry, found, source, entries.index(entry)), default=None)
    shown = best.shown if best is not None else None
    if shown is not None and shown.lines:
        shown = _without_lines_of_other_rows(shown, found)
    boxes = [] if shown is None else text_boxes(shown) + list(shown.hidden)
    known = [box for entry in entries for box in text_boxes(entry.shown) + list(entry.shown.hidden)]
    missing = tuple(row.box for row in rows
                    if (row.sure or is_covered(row.box, known)) and not is_covered(row.box, boxes))
    if not missing:
        return shown
    if shown is None:
        return Shown("", _union(list(missing)), (), missing)
    return replace(shown, hidden=shown.hidden + missing)


def _union(boxes: list[Box]) -> Box:
    return type(boxes[0])(left=min(b.left for b in boxes), top=min(b.top for b in boxes),
                          right=max(b.right for b in boxes), bottom=max(b.bottom for b in boxes))


class FrameBuffer:
    """A legutóbbi képkockák (időbélyeg, kép); a rögzítő szál tölti és olvassa."""

    def __init__(self, keep_seconds: float) -> None:
        self._keep_seconds = keep_seconds
        self._frames: deque[tuple[float, np.ndarray]] = deque()

    def push(self, at: float, frame: np.ndarray) -> None:
        self._frames.append((at, frame))
        while len(self._frames) > 1 and self._frames[1][0] < at - self._keep_seconds:
            self._frames.popleft()

    def at_or_before(self, moment: float) -> tuple[float, np.ndarray] | None:
        """A legkésőbbi képkocka, amely legkésőbb ekkor készült; None, ha még nincs ilyen régi."""
        found = None
        for item in self._frames:
            if item[0] > moment:
                break
            found = item
        return found

    def between(self, start: float, end: float) -> list[tuple[float, np.ndarray]]:
        """A [start, end] közötti képkockák időrendben."""
        return [item for item in self._frames if start <= item[0] <= end]

    def clear(self) -> None:
        self._frames.clear()
