"""A soralapú út sorkeresője: a klasszikus sorkereső (row_detector) és a képpont-aláírás (row_signature) illesztője.

A row_tracker csak ennyit tud a sorokról: hol állnak, feliratszínűek-e, biztos-e bennük a kereső, és
két rögzítés sora ugyanazokból a képpontokból áll-e. Ezt adja ez a modul, a teljes kép koordinátáiban.
A make_row_finder(whole_band=True) sorkeresője a sáv középen kívüli szövegét is visszaadja (sosem biztos sorként).
Mindkét függvény azon a szálon fut, amelyik a sávokat feldolgozza; a sávot nem módosítják.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial

import numpy as np

from .fragment_filter import DIALOGUE_BOTTOM_RATIO, MAX_GREY_SPREAD, MIN_WHITE_FRACTION, white_fraction
from .ocr import Box
from .row_detector import DEFAULT_PARAMS, REFERENCE_WIDTH, Detection, Params, detect
from .row_signature import SAME, Signature, compare, row_signature
from .row_tracker import DetectedRow, RowFinder

# A sorkereső küszöbei 1920 képpont széles képen mértek: 1600 alatt a soroknak már csak ~75%-át találja meg,
# ezért keskenyebb ablakban a teljes (neurális) olvasás marad
MIN_FRAME_WIDTH = 1600

# A küldetésutasítás sorát a sáv alsó csíkjában keressük: a párbeszédhatár fölött ennyivel kezdődik (a képmagasság
# arányában; a legalsó párbeszédsor alja ~0,915-nél, a küldetéssoré ~0,945-nél van), és legalább ilyen magas
OBJECTIVE_STRIP_ABOVE_RATIO = 0.01
OBJECTIVE_STRIP_MIN_RATIO = 0.04

_Signed = tuple[Signature, float]  # (a sor képpontjai, a kép szélessége a 1920-hoz mérve)


def is_wide_enough(frame_width: int) -> bool:
    """Elég széles-e a kép ahhoz, hogy a soralapú út megbízhatóan megtalálja a feliratsorokat."""
    return frame_width >= MIN_FRAME_WIDTH


def find_rows(band: np.ndarray, area: Box, frame_size: tuple[int, int],
              params: Params = DEFAULT_PARAMS) -> list[DetectedRow]:
    """A sáv sorai fentről lefelé, a teljes kép koordinátáiban: a párbeszédsorok, és alattuk a középre
    igazított küldetésutasítás sora (azt a kereső sosem adja biztos sorként: olvasás előtt nem takarjuk)."""
    frame_width, frame_height = frame_size
    dialogue_bottom = DIALOGUE_BOTTOM_RATIO * frame_height - area.top
    detection = detect(band, area.left, frame_width, params, dialogue_bottom=dialogue_bottom)
    rows = _detected_rows(detection, band, area, frame_width, 0)
    return rows + _objective_rows(band, area, frame_size, params, dialogue_bottom)


def _objective_rows(band: np.ndarray, area: Box, frame_size: tuple[int, int], params: Params,
                    dialogue_bottom: float) -> list[DetectedRow]:
    """A párbeszédsorok alatti, középre igazított sorok (küldetésutasítás) a sáv alsó csíkjából keresve."""
    frame_width, frame_height = frame_size
    top = max(0, int(dialogue_bottom - OBJECTIVE_STRIP_ABOVE_RATIO * frame_height))
    if band.shape[0] - top < OBJECTIVE_STRIP_MIN_RATIO * frame_height:
        return []  # a sáv nem ér le a küldetéssorig
    strip = np.ascontiguousarray(band[top:])
    detection = detect(strip, area.left, frame_width, replace(params, whole_band=False))
    below = replace(detection, rows=tuple(row for row in detection.rows
                                          if (row.top + row.bottom) / 2 + top > dialogue_bottom))
    return [replace(row, sure=False) for row in _detected_rows(below, strip, area, frame_width, top)]


def _detected_rows(detection: Detection, band: np.ndarray, area: Box, frame_width: int,
                   offset_y: int) -> list[DetectedRow]:
    """A kereső sorai a teljes kép koordinátáiban; band: amiben a kereső keresett, offset_y: annak teteje a sávban."""
    center_x = frame_width / 2 - area.left
    scale = frame_width / REFERENCE_WIDTH
    rows = []
    for row in detection.rows:
        box = Box(row.left + area.left, row.top + offset_y + area.top,
                  row.right + area.left, row.bottom + offset_y + area.top)
        pixels = row_signature(detection.core, row.ink_left, row.ink_top, row.ink_right, row.ink_bottom, center_x)
        whiteness = white_fraction(band, replace(box, top=box.top - offset_y, bottom=box.bottom - offset_y), area)
        if (row.dim or row.sure) and detection.core is not None:
            whiteness = max(whiteness, _letter_whiteness(band[row.top:row.bottom, row.left:row.right],
                                                         detection.core[row.top:row.bottom, row.left:row.right],
                                                         row.sure))
        rows.append(DetectedRow(box, (pixels, scale), whiteness, row.sure))
    return rows


def _letter_whiteness(patch: np.ndarray, core: np.ndarray, is_sure: bool) -> float:
    """A kereső betűmagjából mért feliratszín, ha a betűk színtelenek (fehérek vagy szürkék); különben 0.

    A szürke közbeszólás betűi sötétebbek a fehérség halvány küszöbénél, világos háttér előtt (és a
    tömörített képen) pedig a fehér betű is a hófehér szint alá eshet: ilyenkor a white_fraction keveset mér.
    A biztos sor (körvonalas, középre igazított betűk) színtelen betűkkel mindenképp feliratszínű."""
    letters = core > 0
    if not letters.any():
        return 0.0
    pixels = patch[letters].astype(np.int16)
    if np.median(pixels.max(axis=1) - pixels.min(axis=1)) > MAX_GREY_SPREAD:
        return 0.0  # színes szöveg (név, HUD): nem felirat
    share = float(letters.mean())
    return max(share, MIN_WHITE_FRACTION) if is_sure else share


def is_same_row(old: _Signed, new: _Signed) -> bool:
    """Ugyanazok a képpontok-e. A bizonytalan egyezés új sornak számít: az csak egy újraolvasásba kerül,
    a szöveg sosem lesz tőle hibás."""
    return compare(old[0], new[0], new[1]) == SAME


ROW_FINDER = RowFinder(find_rows, is_same_row)  # csak a kép közepére igazított sorok
WHOLE_BAND_PARAMS = replace(DEFAULT_PARAMS, whole_band=True)
# A sáv középen kívüli szövege is sor (sosem biztos sorként)
WHOLE_BAND_ROW_FINDER = RowFinder(partial(find_rows, params=WHOLE_BAND_PARAMS), is_same_row)


def band_edge_margin(frame_width: float) -> float:
    """A sáv szélétől ennyi képpontnál közelebb érő szabad sort a sáv széle vágta el: a kereső szabálya
    (row_geometry), a kitömött sordobozra, a képszélességhez igazítva. A pipeline is ezzel mér."""
    return WHOLE_BAND_PARAMS.edge_margin * frame_width / REFERENCE_WIDTH


def make_row_finder(whole_band: bool) -> RowFinder:
    """A soralapú olvasás sorkeresője; whole_band: config.whole_band_text (a sáv középen kívüli szövege is sor)."""
    return WHOLE_BAND_ROW_FINDER if whole_band else ROW_FINDER
