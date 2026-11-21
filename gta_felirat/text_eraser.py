"""Az angol feliratsor kitörlése a képről: a betűk helyét a környező háttérből töltjük ki (a tiszta képhez).

Az élő réteg elmosott téglalappal takar (frosted_cover), ami mintás háttéren foltként látszik. A tiszta kép
a késleltetés miatt ráér: itt csak a betűk vonalait (és a fekete körvonalukat) cseréljük le, a háttér marad.

A sorkereső doboza a sor fehér magját fogja; a színes név és a sor vége kilóghat belőle. Ezért a sor szélességét
a teljes képszélességben mérjük, és a feliratsor középre igazított, így a kép közepére tükrözzük.

Csak a betűket és közvetlen körvonalukat töröljük (vékony, a környezetüknél világosabb vonalak és a mellettük lévő
eltérő képpontok): a sor alatti tárgyak (lámpa, lökhárító) élei így nagyrészt megmaradnak, és a kitöltés is
kevésbé maszatol. Ahol vízszintes él fut át a sor alatt, azt vízszintesen visszük tovább (edge_fill).
"""

from __future__ import annotations

import cv2
import numpy as np

from .edge_fill import follow_horizontal_edges
from .ocr import Box

PAD_Y = 5  # a sor fölött és alatt ennyi képpont is a sorhoz tartozik (körvonal, ékezet)
BACKGROUND_KERNEL = 31  # ekkora mediánszűrő a betűk nélküli hátteret becsli
TEXT_DIFF = 30  # a háttértől ennyivel eltérő képpont betű vagy körvonal (a maszkhoz)
STRONG_DIFF = 45  # a sor szélességének méréséhez csak a határozott eltérés számít
MAX_GAP_PX = 28  # 1920 széles képen: a szavak (és a név meg a szöveg) közti rés ennél kisebb
MIN_RUNS = 1.3  # a betűoszlopban átlagosan ennyi szakasz van (egy háttérél csak egyet ad)
RUN_WINDOW = 9  # ennyi oszlopra átlagolunk
LOPSIDED_PX = 40  # a két oldal ennél nagyobb eltérése már a háttér, nem a betű
EDGE_PX = 12  # a mért szélességen túl ennyi ráhagyás (a középre igazítás is pár képpontot téved)
CONTEXT_PX = 16  # a kitöltéshez ekkora környezet kell a sor körül
STROKE_KERNEL = 11  # 1920 széles képen: ennél keskenyebb világos vonal betű (a lámpa, rendszám ennél szélesebb)
STROKE_DIFF = 40  # a betűvonal ennyivel világosabb a környezeténél
OUTLINE_PX = 7  # a betűvonaltól ennyi képponton belül a sötétebb vagy eltérő képpont a körvonalhoz tartozik
OUTLINE_DIFF = 12  # a körvonal ennyivel sötétebb a környezeténél
MASK_GROW = 3  # a maszkot ennyivel bővítjük (élsimítás)
INPAINT_RADIUS = 3
REFERENCE_WIDTH = 1920
GRAIN_SIGMA = 3.0  # ennél finomabb részlet a háttér szemcséje (aszfalt, fű, csempe)
GRAIN_LIMIT = 8.0  # a pótolt szemcse legfeljebb ilyen erős
GRAIN_SHARE = 0.7  # a környezet szemcséjének ekkora részét pótoljuk (a teljes már pöttyösnek hat)
GRAIN_SIZE = 0.8  # a szemcse mérete (képpont); a játékkép szemcséje nem egyetlen képpontos zaj
_GRAIN_RNG = np.random.default_rng(1)  # rögzített mag: a teszt és a futás ugyanazt adja


def _difference(strip: np.ndarray) -> np.ndarray:
    background = cv2.medianBlur(strip, BACKGROUND_KERNEL)
    return np.abs(strip.astype(np.int16) - background.astype(np.int16)).max(axis=2)


def _columns(strong: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(betűoszlopok, tintás oszlopok): az előbbiben átlagosan több szakasz van, az utóbbiban legalább egy."""
    runs = (np.diff(strong.astype(np.int8), axis=0) == 1).sum(axis=0) + strong[0]
    return np.convolve(runs, np.ones(RUN_WINDOW) / RUN_WINDOW, mode="same") >= MIN_RUNS, runs > 0


def _reach(active: np.ndarray, inked: np.ndarray, start: int, step: int, max_gap: int) -> int:
    """Az utolsó betűoszlop start-tól step irányban, legfeljebb max_gap széles réseken át; a szó végén a még
    tintás oszlopok is (a szélső betű egyetlen szára az átlagolásban kiesik)."""
    last, gap, x = start, 0, start
    while 0 < x < len(active) - 1 and gap <= max_gap:
        x += step
        if active[x]:
            last, gap = x, 0
        else:
            gap += 1
    while 0 < last < len(inked) - 1 and inked[last + step]:
        last += step
    return last


def text_extent(difference: np.ndarray, box: Box) -> tuple[int, int]:
    """A sor teljes szövegének vízszintes határa (bal, jobb) a kép közepére szimmetrikusan; difference a sorsáv
    eltérése a háttértől (_difference)."""
    width = difference.shape[1]
    active, inked = _columns(difference > STRONG_DIFF)
    max_gap = round(MAX_GAP_PX * width / REFERENCE_WIDTH)
    left = _reach(active, inked, min(max(box.left, 1), width - 2), -1, max_gap)
    right = _reach(active, inked, min(max(box.right - 1, 1), width - 2), 1, max_gap)
    center = width / 2
    sides = (center - left, right + 1 - center)
    # A felirat középre igazított. Ha az egyik oldal jóval tovább ér, ott a háttér „folytatja” a sort: a rövidebb
    # számít; kis eltérésnél (a szélső betű halvány vége) a hosszabb
    is_lopsided = abs(sides[0] - sides[1]) > LOPSIDED_PX * width / REFERENCE_WIDTH
    half = max(min(sides) if is_lopsided else max(sides), box.width / 2) + EDGE_PX
    return max(0, int(center - half)), min(width, int(np.ceil(center + half)))


def erase_row(image: np.ndarray, box: Box) -> Box:
    """A sor betűit kitörli az image-ben (helyben); visszaadja a törölt sor dobozát (a teljes szöveggel)."""
    height, width = image.shape[:2]
    top, bottom = max(0, box.top - PAD_Y), min(height, box.bottom + PAD_Y)
    if bottom <= top:
        return box
    strip = image[top:bottom]
    difference = _difference(strip)
    left, right = text_extent(difference, box)
    mask = letter_mask(strip, difference, (left, right))
    outer = Box(max(0, left - CONTEXT_PX), max(0, top - CONTEXT_PX),
                min(width, right + CONTEXT_PX), min(height, bottom + CONTEXT_PX))
    crop = image[outer.top:outer.bottom, outer.left:outer.right]
    full = np.zeros(crop.shape[:2], np.uint8)
    full[top - outer.top:bottom - outer.top] = mask[:, outer.left:outer.right]
    _fill(crop, full, width)
    return Box(left, top, right, bottom)


def _ellipse(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def letter_mask(strip: np.ndarray, difference: np.ndarray, extent: tuple[int, int]) -> np.ndarray:
    """A betűk és körvonaluk maszkja a sorsávban, az extent (bal, jobb) oszlopai között."""
    left, right = extent
    width = strip.shape[1]
    kernel = _ellipse(max(3, round(STROKE_KERNEL * width / REFERENCE_WIDTH)) | 1)
    value = strip.max(axis=2)
    core = (cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel) > STROKE_DIFF).astype(np.uint8)
    core[:, :left] = 0
    core[:, right:] = 0
    near = cv2.dilate(core, _ellipse(OUTLINE_PX)) > 0
    outline = (cv2.morphologyEx(value, cv2.MORPH_BLACKHAT, kernel) > OUTLINE_DIFF) | (difference > TEXT_DIFF)
    mask = (core > 0) | (outline & near)
    return cv2.dilate(mask.astype(np.uint8), _ellipse(MASK_GROW))


def _grain(crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """A környező háttér szemcséivel azonos erősségű, szerkezet nélküli szemcse (a mintás háttéren különben sima
    folt maradna). Nem másolunk át mintát: a szomszéd sáv tárgyai szellemképként jelennének meg."""
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    detail = grey - cv2.GaussianBlur(grey, (0, 0), GRAIN_SIGMA)
    # A sor fölötti és alatti szegélyből mérünk; a medián eltérést a szórványos élek (más sor betűi) nem húzzák fel
    around = np.concatenate([detail[:CONTEXT_PX].ravel(), detail[-CONTEXT_PX:].ravel()])
    spread = 1.4826 * float(np.median(np.abs(around - np.median(around)))) if around.size else 0.0
    strength = min(GRAIN_SHARE * spread, GRAIN_LIMIT)
    noise = cv2.GaussianBlur(_GRAIN_RNG.standard_normal(grey.shape).astype(np.float32), (0, 0), GRAIN_SIZE)
    noise *= strength / max(float(noise.std()), 1e-6)
    return noise[..., None]


def _fill(crop: np.ndarray, mask: np.ndarray, frame_width: int) -> None:
    """A maszkolt képpontok kitöltése a környezetből, a vízszintes élek továbbvitelével, a háttér szemcséivel
    (különben sima folt maradna a mintás háttéren), puha széllel."""
    filled = cv2.inpaint(crop, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)
    filled = follow_horizontal_edges(crop, mask, filled, frame_width)
    filled += _grain(crop, mask)
    soft = cv2.GaussianBlur(mask.astype(np.float32), (3, 3), 0)[..., None]
    crop[:] = np.clip(crop * (1.0 - soft) + filled * soft, 0, 255).astype(np.uint8)
