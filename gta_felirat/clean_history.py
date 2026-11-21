"""A késleltetett tiszta kép alapja: a játékképek és a feliratállapotok rövid története, időbélyeggel.

A feliratréteg mindig kicsit lemarad a játéktól: az új angol sort csak a felismerése után takarja, és a fordítás
is idő. A tiszta kép ezért régebbi képkockát mutat, és arra azt az állapotot rajzolja, amelyik ahhoz a felirathoz
végül kialakult (a kész fordítást), nem azt, ami a képkocka pillanatában éppen kint volt.

Melyik feliratot látjuk a képkockán: amelyik SETTLE_SECONDS-szal később a pipeline aktuális szövege volt (eddigre a
felismerés és a megerősítés biztosan utolérte a képet). Ennek a feliratnak a legutolsó ismert állapota kerül rá.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from .layer_plan import Shown

SETTLE_SECONDS = 0.4  # ennyi idő alatt a pipeline biztosan felismeri és megerősíti a képen megjelent feliratot
HISTORY_SLACK_SECONDS = 1.0  # a késleltetésen túl ennyi régebbi állapotot is megtartunk


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

    def settled(self, frame_at: float) -> Shown | None:
        """A képkockán látszó felirat végleges (legutolsó ismert) állapota."""
        with self._lock:
            entries = list(self._entries)
        seen_at = frame_at + SETTLE_SECONDS
        index = next((i for i in range(len(entries) - 1, -1, -1) if entries[i].at <= seen_at), None)
        if index is None:
            return None
        source = entries[index].source
        while index + 1 < len(entries) and entries[index + 1].source == source:
            index += 1  # ugyanannak a feliratnak a későbbi állapota: a fordítás közben elkészült
        return entries[index].shown


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

    def clear(self) -> None:
        self._frames.clear()
