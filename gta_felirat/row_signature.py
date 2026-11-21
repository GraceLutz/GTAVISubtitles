"""Olcsó változásjelző feliratsoronként: a sor betűmag-maszkja, a kép közepéhez horgonyozva.

A GTA minden feliratsort a kép közepére igazít, ezért ugyanaz a szöveg minden felvételen és minden
sorhelyen (a sorok fölfelé gördülnek) ugyanott áll vízszintesen. A jel ezért a sor maszkját a kép
közepétől mért vízszintes helyével együtt őrzi; két jelet ennél a horgonynál egymásra fektetve,
néhány függőleges eltolást kipróbálva hasonlítunk össze (függőlegesen a festék súlypontja a horgony,
az ±1–2 pixelt ingadozik).

    hasonlóság = |A és B| / min(|A|, |B|)      (1,0: az egyik maszk tartalmazza a másikat)

Szándékosan a kisebb maszkkal osztunk: világos háttér előtt a betűk közti fehér zugok is bekerülnek
a maszkba, így ugyanannak a szövegnek nagyobb lehet a maszkja; maguk a betűk viszont mindkettőben
ott vannak, vagyis a kisebb maszk (majdnem) benne van a nagyobban.

Mérve a felvételeken: azonos szöveg hasonlóságának mediánja 0,996, különböző szövegé legfeljebb 0,426.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import cv2
import numpy as np

MASK_ON = 255  # a maszk bekapcsolt pixelének értéke
MAX_SHIFT_Y = 2  # ennyi pixel függőleges eltolást próbálunk ki mindkét irányban
SAME_OVERLAP = 0.65  # azonos szöveg: 810 egymást követő párból 808-ban >= 0,80; különböző szöveg: <= 0,43
MAX_WIDTH_DIFFERENCE = 16  # px, 1920 széles képen: ugyanaz a szöveg ugyanolyan széles (±3 px)
LOCAL_WINDOW = 14  # px, 1920 széles képen: nagyjából egy betű
MIN_LOCAL_WINDOW = 2  # kis képen se legyen az ablak ennél keskenyebb
LOCAL_CHANGE = 18  # ennyi eltérő pixel egy ablakon belül: egy betű kicserélődött („Go.” → „Go!”)

Verdict = Literal["different", "same", "verify"]
DIFFERENT: Final = "different"  # más szöveg
SAME: Final = "same"  # ugyanaz a szöveg: a felismert szöveg és a fordítása újra felhasználható
VERIFY: Final = "verify"  # most újra felhasználható, de a sort egyszer olvassuk újra


@dataclass(frozen=True, eq=False)  # eq=False: a numpy maszkok == összehasonlítása hibát dobna
class Signature:
    """Egy feliratsor változásjele."""

    mask: np.ndarray  # uint8, 0/255: a sor betűmagjai (saját, csak olvasható másolat)
    x0: int  # a maszk első oszlopa a kép közepéhez képest (px)
    y0: int  # a maszk első sora a festék súlypontjához képest (px)
    weight: int  # a bekapcsolt pixelek száma

    @property
    def width(self) -> int:
        return self.mask.shape[1]


def row_signature(core: np.ndarray, ink_left: int, ink_top: int, ink_right: int, ink_bottom: int,
                  center_x: float) -> Signature:
    """A sor jele a sáv betűmag-maszkjából (core: row_detector.Detection.core).

    A festett doboz (Row.ink_*) és a kép vízszintes közepe (center_x) a sáv koordinátáiban értendő.
    """
    mask = core[ink_top:ink_bottom, ink_left:ink_right].copy()  # a jel ne tartsa életben a sáv teljes maszkját
    mask.setflags(write=False)
    lines = cv2.reduce(mask, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S).ravel().astype(np.float64)
    total = float(lines.sum())
    centroid = float((lines * np.arange(lines.size)).sum() / total) if total else lines.size / 2.0
    return Signature(mask=mask, x0=ink_left - int(round(center_x)), y0=-int(round(centroid)),
                     weight=int(total // MASK_ON))


def similarity(a: Signature, b: Signature) -> float:
    """A két sor közös magpixelei a kisebb maszkhoz mérve (0–1), a legjobb függőleges eltolásnál."""
    smaller = min(a.weight, b.weight)
    if smaller == 0:
        return 0.0
    return _best_shift(a, b)[1] / float(smaller)


def compare(a: Signature, b: Signature, scale: float = 1.0) -> Verdict:
    """Ugyanaz a szöveg-e a két sor; scale: képszélesség / 1920.

    DIFFERENT: más szöveg. SAME: ugyanaz. VERIFY: ugyanannak vesszük, de a sort egyszer újra kell
    olvasni (egy betű megváltozhatott, vagy elmozdult mögötte a világos háttér).
    """
    if abs(a.width - b.width) > MAX_WIDTH_DIFFERENCE * scale:
        return DIFFERENT
    smaller = min(a.weight, b.weight)
    if smaller == 0:
        return DIFFERENT
    dy, overlap = _best_shift(a, b)
    if overlap < SAME_OVERLAP * smaller:
        return DIFFERENT
    window = max(MIN_LOCAL_WINDOW, int(round(LOCAL_WINDOW * scale)))
    return VERIFY if _local_change(a, b, dy, window) >= LOCAL_CHANGE * scale else SAME


def _overlap(a: Signature, b: Signature, dy: int) -> int:
    """A közös bekapcsolt pixelek száma, ha b-t dy sorral eltoljuk a közös horgonyhoz képest."""
    b_y0 = b.y0 + dy
    left, right = max(a.x0, b.x0), min(a.x0 + a.mask.shape[1], b.x0 + b.mask.shape[1])
    top, bottom = max(a.y0, b_y0), min(a.y0 + a.mask.shape[0], b_y0 + b.mask.shape[0])
    if right <= left or bottom <= top:
        return 0
    part_a = a.mask[top - a.y0:bottom - a.y0, left - a.x0:right - a.x0]
    part_b = b.mask[top - b_y0:bottom - b_y0, left - b.x0:right - b.x0]
    return cv2.countNonZero(cv2.bitwise_and(part_a, part_b))


def _best_shift(a: Signature, b: Signature) -> tuple[int, int]:
    """(b függőleges eltolása, amelynél a legnagyobb az átfedés; az átfedés pixelben)."""
    best_dy, best = 0, -1
    for dy in range(-MAX_SHIFT_Y, MAX_SHIFT_Y + 1):
        overlap = _overlap(a, b, dy)
        if overlap > best:
            best_dy, best = dy, overlap
    return best_dy, best


def _local_change(a: Signature, b: Signature, dy: int, window: int) -> int:
    """A legtöbb eltérő maszkpixel bármely `window` szomszédos oszlopon belül (b dy sorral eltolva)."""
    left, right = min(a.x0, b.x0), max(a.x0 + a.mask.shape[1], b.x0 + b.mask.shape[1])
    top, bottom = min(a.y0, b.y0 + dy), max(a.y0 + a.mask.shape[0], b.y0 + dy + b.mask.shape[0])
    shape = (bottom - top, right - left)
    changed = cv2.bitwise_xor(_placed(a, (a.y0 - top, a.x0 - left), shape),
                              _placed(b, (b.y0 + dy - top, b.x0 - left), shape))
    columns = cv2.reduce(changed, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S).ravel() // MASK_ON
    if columns.size <= window:
        return int(columns.sum())
    return int(np.convolve(columns, np.ones(window, dtype=np.int64), mode="valid").max())


def _placed(signature: Signature, corner: tuple[int, int], shape: tuple[int, int]) -> np.ndarray:
    """A jel maszkja egy üres vásznon; corner: a maszk bal felső sarkának (sor, oszlop) helye."""
    canvas = np.zeros(shape, np.uint8)
    height, width = signature.mask.shape
    canvas[corner[0]:corner[0] + height, corner[1]:corner[1] + width] = signature.mask
    return canvas
