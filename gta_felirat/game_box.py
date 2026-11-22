"""A játék sötét feliratdoboza a sor mögött (ha a felirathoz be van kapcsolva a háttér).

A betűtörlés után a doboz maradéka szakadozott sötét sávként marad a magyar szöveg körül (a betűk körvonala a
doboz szélébe is belemar, és a doboz vége nem a magyar szöveg végénél van). A tiszta kép ezért a doboz helyére
éles szélű, egyszínű dobozt rajzol, amely a dobozt és a magyar szöveget is körbefogja (clean_frame).

A doboz áttetsző sötét téglalap: a széleinél a kép hirtelen sötétebb lesz. Ahol a doboz mögött a kép is sötét,
ott a széle nem látszik: ilyenkor a szemközti szélt tükrözzük (a doboz a szövegre szimmetrikus). Ha egyik szél
sem látszik (sötét jelenet), nincs doboz (és a maradéka sem látszik).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ocr import Box

REFERENCE_WIDTH = 1920
STEP = 15  # a doboz szélén a kép legalább ennyivel sötétebb lesz (színcsatornák maximuma)
STEP_REACH_PX = 2  # a lépést ennyi képpontnyira mérjük a szél két oldalán (élsimítás, tömörítés)
EDGE_SHARE = 0.6  # a szél mentén a képpontok ekkora részén van ilyen lépés
MAX_BOX_LEVEL = 90  # a doboz (mediánja) legfeljebb ilyen világos
SEARCH_PX = 24  # 1920 széles képen: a sor szélein túl ennyit keresünk
INNER_PX = 20  # a vízszintes széleket a sor belső részén mérjük (a doboz vége a sor végénél is lehet)


@dataclass(frozen=True)
class GameBox:
    box: Box
    color: tuple[int, int, int]  # BGR


def _edge(profile: np.ndarray, start: int, end: int, darker_after: bool, reach: int) -> int | None:
    """A legerősebb szél [start, end) között: az első sötét index (darker_after) vagy az utolsó sötét utáni.

    profile: (hossz, keresztirány) világosság; a szél ott van, ahol a keresztirány nagy részén lépés van."""
    best, best_share = None, EDGE_SHARE
    for i in range(max(start, reach), min(end, len(profile) - reach)):
        before, after = profile[i - reach], profile[i + reach - 1]
        step = before - after if darker_after else after - before
        dark = after if darker_after else before
        share = float(np.mean((step >= STEP) & (dark <= MAX_BOX_LEVEL)))
        if share > best_share:
            best, best_share = i, share
    if best is None:
        return None
    return best if darker_after else best + reach - 1


def _sides(profile: np.ndarray, start: int, end: int, center: float, reach: int) -> tuple[int, int] | None:
    """A doboz két szemközti széle (eleje, vége), a hiányzót a center-re tükrözve; None, ha egyik sem látszik."""
    first = _edge(profile, start, int(center), True, reach)
    last = _edge(profile, int(center) + 1, end, False, reach)
    if first is None and last is None:
        return None
    if first is None:
        first = round(2 * center - last)
    if last is None:
        last = round(2 * center - first)
    return first, last


def find_game_box(image: np.ndarray, row: Box) -> GameBox | None:
    """A sor mögötti sötét feliratdoboz, ha van; row a sor (teljes szövegének) doboza."""
    height, width = image.shape[:2]
    scale = width / REFERENCE_WIDTH
    search, inner = round(SEARCH_PX * scale), round(INNER_PX * scale)
    reach = max(1, round(STEP_REACH_PX * scale))
    top, bottom = max(0, row.top - search), min(height, row.bottom + search)
    left, right = max(0, row.left - search), min(width, row.right + search)
    if row.width <= 2 * inner or bottom - top < 4 * reach:
        return None
    value = image[top:bottom, left:right].max(axis=2).astype(np.int16)
    across = value[:, row.left + inner - left:row.right - inner - left]
    rows = _sides(across, 0, bottom - top, (row.top + row.bottom) / 2 - top, reach)
    if rows is None or rows[1] - rows[0] < 4 * reach:
        return None
    band = value[rows[0]:rows[1]].T  # oszloponként
    columns = _sides(band, 0, right - left, width / 2 - left, reach)
    if columns is None:
        columns = (row.left - left, row.right - left)
    box = Box(max(0, left + columns[0]), top + rows[0], min(width, left + columns[1] + 1), top + rows[1] + 1)
    inside = image[box.top:box.bottom, box.left:box.right].reshape(-1, 3)
    if box.width <= 0 or not len(inside):
        return None
    color = np.median(inside, axis=0)
    if float(color.max()) > MAX_BOX_LEVEL:
        return None
    return GameBox(box, (int(color[0]), int(color[1]), int(color[2])))
