"""A soralapú út sorkeresője: a klasszikus sorkereső (row_detector) és a képpont-aláírás (row_signature) illesztője.

A row_tracker csak ennyit tud a sorokról: hol állnak, feliratszínűek-e, biztos-e bennük a kereső, és
két rögzítés sora ugyanazokból a képpontokból áll-e. Ezt adja ez a modul, a teljes kép koordinátáiban.
Mindkét függvény azon a szálon fut, amelyik a sávokat feldolgozza; a sávot nem módosítják.
"""

from __future__ import annotations

import numpy as np

from .fragment_filter import DIALOGUE_BOTTOM_RATIO, white_fraction
from .ocr import Box
from .row_detector import REFERENCE_WIDTH, detect
from .row_signature import SAME, Signature, compare, row_signature
from .row_tracker import DetectedRow, RowFinder

# A sorkereső küszöbei 1920 képpont széles képen mértek: 1600 alatt a soroknak már csak ~75%-át találja meg,
# ezért keskenyebb ablakban a teljes (neurális) olvasás marad
MIN_FRAME_WIDTH = 1600

_Signed = tuple[Signature, float]  # (a sor képpontjai, a kép szélessége a 1920-hoz mérve)


def is_wide_enough(frame_width: int) -> bool:
    """Elég széles-e a kép ahhoz, hogy a soralapú út megbízhatóan megtalálja a feliratsorokat."""
    return frame_width >= MIN_FRAME_WIDTH


def find_rows(band: np.ndarray, area: Box, frame_size: tuple[int, int]) -> list[DetectedRow]:
    """A sáv párbeszédsorai fentről lefelé, a teljes kép koordinátáiban (a küldetésutasítás sora nélkül)."""
    frame_width, frame_height = frame_size
    detection = detect(band, area.left, frame_width, dialogue_bottom=DIALOGUE_BOTTOM_RATIO * frame_height - area.top)
    center_x = frame_width / 2 - area.left
    scale = frame_width / REFERENCE_WIDTH
    rows = []
    for row in detection.rows:
        box = Box(row.left + area.left, row.top + area.top, row.right + area.left, row.bottom + area.top)
        pixels = row_signature(detection.core, row.ink_left, row.ink_top, row.ink_right, row.ink_bottom, center_x)
        rows.append(DetectedRow(box, (pixels, scale), white_fraction(band, box, area), row.sure))
    return rows


def is_same_row(old: _Signed, new: _Signed) -> bool:
    """Ugyanazok a képpontok-e. A bizonytalan egyezés új sornak számít: az csak egy újraolvasásba kerül,
    a szöveg sosem lesz tőle hibás."""
    return compare(old[0], new[0], new[1]) == SAME


ROW_FINDER = RowFinder(find_rows, is_same_row)
