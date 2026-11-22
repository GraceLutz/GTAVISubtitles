"""A játék sötét feliratdoboza a felirat mögött (ha a felirathoz be van kapcsolva a háttér).

A betűtörlés után a doboz maradéka szakadozott sötét sávként marad a magyar szöveg körül (a betűk körvonala a
doboz szélébe is belemar, és a doboz vége nem a magyar szöveg végénél van). A tiszta kép ezért a doboz helyére
éles szélű, egyszínű dobozt rajzol, amely a dobozt és a magyar szöveget is körbefogja (clean_frame).

A játék egy feliratnak egyetlen dobozt rajzol: többsoros feliratnál a doboz az összes sort fogja, és olyan
széles, mint a leghosszabb sor. Ezért a szomszédos sorokat blokká vonjuk össze (blocks), és a blokk szélei
körül keressük a doboz széleit: ott a kép hirtelen sötétebb lesz. Ahol a doboz mögött a kép is sötét, ott a
széle nem látszik: ilyenkor a szemközti szélt tükrözzük (a doboz a feliratra szimmetrikus). Ha egyik szél sem
látszik (sötét jelenet), nincs doboz (és a maradéka sem látszik).
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
SEARCH_PX = 24  # 1920 széles képen: a blokk szélein kívül ennyit keresünk
NEAR_PX = 12  # és belül ennyit (a doboz szélének a blokk széle közelében kell lennie, nem a betűk között)
INNER_PX = 20  # a vízszintes széleket a blokk belső oszlopain mérjük
BLOCK_GAP_PX = 6  # az ennél közelebbi sorok egy felirat (egy doboz) sorai


@dataclass(frozen=True)
class GameBox:
    box: Box
    color: tuple[int, int, int]  # BGR


def blocks(rows: list[Box], frame_width: int) -> list[list[int]]:
    """A sorok indexei feliratblokkonként (egymás alatti, egymáshoz közeli sorok)."""
    gap = BLOCK_GAP_PX * frame_width / REFERENCE_WIDTH
    order = sorted(range(len(rows)), key=lambda i: rows[i].top)
    grouped: list[list[int]] = []
    for i in order:
        if grouped and rows[i].top <= max(rows[j].bottom for j in grouped[-1]) + gap:
            grouped[-1].append(i)
        else:
            grouped.append([i])
    return grouped


def _edge(profile: np.ndarray, window: tuple[int, int], darker_after: bool, reach: int) -> int | None:
    """A legerősebb szél a window-ban: az első sötét index (darker_after) vagy az utolsó sötét utáni.

    profile: (hossz, keresztirány) világosság; a szél ott van, ahol a keresztirány nagy részén lépés van."""
    best, best_share = None, EDGE_SHARE
    for i in range(max(window[0], reach), min(window[1], len(profile) - reach)):
        before, after = profile[i - reach], profile[i + reach - 1]
        step = before - after if darker_after else after - before
        dark = after if darker_after else before
        share = float(np.mean((step >= STEP) & (dark <= MAX_BOX_LEVEL)))
        if share > best_share:
            best, best_share = i, share
    if best is None:
        return None
    return best if darker_after else best + reach - 1


def _sides(profile: np.ndarray, ends: tuple[int, int], center: float, margins: tuple[int, int],
           reach: int, both: bool = False) -> tuple[int, int] | None:
    """A doboz két szemközti széle az ends (a blokk két széle) közelében; a hiányzót a center-re tükrözzük
    (both: mindkettőnek látszania kell)."""
    outside, inside = margins
    first = _edge(profile, (ends[0] - outside, ends[0] + inside), True, reach)
    last = _edge(profile, (ends[1] - inside, ends[1] + outside), False, reach)
    if (first is None and last is None) or (both and (first is None or last is None)):
        return None
    if first is None:
        first = round(2 * center - last)
    if last is None:
        last = round(2 * center - first)
    return first, last


def find_game_box(image: np.ndarray, block: Box, own: bool = False) -> GameBox | None:
    """A felirat mögötti sötét doboz, ha van; block a felirat sorainak (teljes szövegük) közös doboza.
    own: csak akkor, ha a doboz alsó és felső széle is látszik (egy sor saját doboza)."""
    height, width = image.shape[:2]
    scale = width / REFERENCE_WIDTH
    search, near, inner = round(SEARCH_PX * scale), round(NEAR_PX * scale), round(INNER_PX * scale)
    reach = max(1, round(STEP_REACH_PX * scale))
    top, bottom = max(0, block.top - search), min(height, block.bottom + search)
    left, right = max(0, block.left - search), min(width, block.right + search)
    if block.width <= 2 * inner or bottom - top < 4 * reach:
        return None
    value = image[top:bottom, left:right].max(axis=2).astype(np.int16)
    across = value[:, block.left + inner - left:block.right - inner - left]
    rows = _sides(across, (block.top - top, block.bottom - top), (block.top + block.bottom) / 2 - top,
                  (search, near), reach, own)
    if rows is None or rows[1] - rows[0] < block.height / 2:
        return None
    band = value[max(0, rows[0]):rows[1]].T  # oszloponként
    columns = _sides(band, (block.left - left, block.right - left), width / 2 - left, (search, near), reach)
    if columns is None:
        columns = (block.left - left, block.right - left)
    box = Box(max(0, left + columns[0]), max(0, top + rows[0]),
              min(width, left + columns[1] + 1), min(height, top + rows[1] + 1))
    inside = image[box.top:box.bottom, box.left:box.right].reshape(-1, 3)
    if box.width <= 0 or not len(inside):
        return None
    color = np.median(inside, axis=0)
    if float(color.max()) > MAX_BOX_LEVEL:
        return None
    return GameBox(box, (int(color[0]), int(color[1]), int(color[2])))


def game_boxes(image: np.ndarray, rows: list[Box]) -> list[GameBox | None]:
    """Soronként a mögötte álló játékdoboz. A külön beszélők sorainak saját doboza van (felső és alsó széle is
    látszik); a többi szomszédos sor egy felirat sorai, közös dobozzal."""
    found: list[GameBox | None] = [None] * len(rows)
    for block in blocks(rows, image.shape[1]):
        if len(block) > 1:
            for i in block:
                found[i] = find_game_box(image, rows[i], own=True)
        rest = [i for i in block if found[i] is None]
        for part in _contiguous(rest, block):
            shared = find_game_box(image, Box(min(rows[i].left for i in part), min(rows[i].top for i in part),
                                              max(rows[i].right for i in part), max(rows[i].bottom for i in part)))
            for i in part:
                found[i] = shared
    return found


def _contiguous(rest: list[int], block: list[int]) -> list[list[int]]:
    """A rest elemei a block sorrendjében egymás utáni szakaszokra bontva."""
    parts: list[list[int]] = []
    previous = None
    for position, i in enumerate(block):
        if i not in rest:
            continue
        if parts and previous == position - 1:
            parts[-1].append(i)
        else:
            parts.append([i])
        previous = position
    return parts
