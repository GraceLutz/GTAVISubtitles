"""A tiszta kép egy képkockájának előkészítése: a feliratsorok megkeresése és kitörlése (háttérszálon).

A rögzítéskor fut (analyse), mert a sorkeresés és a törlés soronként ~20 ms; a kép úgyis késleltetve kerül ki,
addigra kész. Megjelenítéskor csak azok a sorok kerülnek a képre kitörölve, amelyeket a választott állapot
fordít vagy takar (with_erased_rows); a többi (pl. rendszámtábla) érintetlen marad.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from .clean_history import FrameRow, is_covered, is_same_line, text_boxes
from .game_box import game_boxes
from .layer_plan import CoverJob, union
from .ocr import Box, region_to_box
from .row_finder import find_rows, is_wide_enough
from .text_eraser import erase_row

if TYPE_CHECKING:
    from .config import AppConfig
    from .layer_plan import LayerPlan, Shown

REFERENCE_WIDTH = 1920
NEAT_PAD_X_PX = 14  # 1920 széles képen: a magyar szöveg két oldalán ennyi a doboz ráhagyása
NEAT_MARGIN_PX = 2  # a játékdoboz élsimított széle is alá kerüljön


@dataclass(frozen=True)
class FrameAnalysis:
    rows: list[FrameRow] | None  # a képkockán talált sorok, a teljes szövegük szélességében; None: túl keskeny kép
    erased: np.ndarray | None = None  # a képkocka minden talált sor kitörlésével


def analyse(image: np.ndarray, config: AppConfig) -> FrameAnalysis:
    height, width = image.shape[:2]
    if not is_wide_enough(width):
        return FrameAnalysis(None)
    area = region_to_box(config.subtitle_region, width, height)
    found = find_rows(image[area.top:area.bottom, area.left:area.right], area, (width, height))
    if not found:
        return FrameAnalysis([])
    erased = image.copy()
    boxes = [erase_row(erased, row.box) for row in found]
    rows = [FrameRow(box, row.sure, game_box) for box, row, game_box in zip(boxes, found, game_boxes(image, boxes))]
    return FrameAnalysis(rows, erased)


def _sure_boxes(analyses: list[FrameAnalysis]) -> list[Box]:
    return [row.box for analysis in analyses for row in analysis.rows or [] if row.sure]


def steadied(analysis: FrameAnalysis, before: list[FrameAnalysis], after: list[FrameAnalysis]) -> FrameAnalysis:
    """A képkocka sorai a szomszédos képkockák (before: korábbiak időrendben, after: későbbiek) fényében.

    A sorkereső ugyanazt a sort képkockánként hol biztosnak, hol bizonytalannak látja, és néha meg sem találja
    (világos háttér). A bizonytalan sort csak a már ismert fordítás takarná, a meg nem talált sort semmi: az angol
    szöveg átlátszana. Ezért a szomszédjai szerint biztos sor itt is biztos; és ha itt nincs sor, de előtte és
    utána ugyanott biztos sor áll, az előző képkocka elemzése (sorok és törlés) érvényes."""
    if analysis.rows is None:
        return analysis
    if not analysis.rows:
        previous = next((other for other in reversed(before) if other.rows), None)
        later = _sure_boxes(after)
        if previous is not None and any(row.sure and any(is_same_line(row.box, box) for box in later)
                                        for row in previous.rows):
            return previous
        return analysis
    sure = _sure_boxes(before) + _sure_boxes(after)
    rows = [replace(row, sure=True) if not row.sure and any(is_same_line(row.box, box) for box in sure) else row
            for row in analysis.rows]
    return analysis if rows == analysis.rows else replace(analysis, rows=rows)


def _overlaps(first: Box, second: Box) -> bool:
    return is_covered(first, [second]) or is_covered(second, [first])


def with_erased_rows(image: np.ndarray, analysis: FrameAnalysis,
                     shown: Shown | None) -> tuple[np.ndarray, list[Box]]:
    """A kép azoknak a soroknak a kitörlésével, amelyeket a shown fordít vagy takar; és a kitörölt sorok."""
    if shown is None or analysis.erased is None or not analysis.rows:
        return image, []
    covered = text_boxes(shown) + list(shown.hidden)
    erased_rows = [row.box for row in analysis.rows if any(_overlaps(row.box, box) for box in covered)]
    if not erased_rows:
        return image, []
    output = image.copy()
    for box in erased_rows:
        output[box.top:box.bottom, box.left:box.right] = analysis.erased[box.top:box.bottom, box.left:box.right]
    return output, erased_rows


def _text_boxes(plan: LayerPlan) -> list[Box]:
    return [Box(job.x, job.y, job.x + job.rendered.width, job.y + job.rendered.height)
            for job in plan.texts]


def _overlap_y(first: Box, second: Box) -> bool:
    return min(first.bottom, second.bottom) > max(first.top, second.top)


def _merged(boxes: list[Box]) -> list[Box]:
    merged: list[Box] = []
    for box in boxes:
        touching = [other for other in merged if _overlap_y(box, other)
                    and min(box.right, other.right) > max(box.left, other.left)]
        merged = [other for other in merged if other not in touching] + [union([box, *touching])]
    return merged


def with_neat_game_boxes(image: np.ndarray, plan: LayerPlan | None, analysis: FrameAnalysis,
                         erased_rows: list[Box]) -> np.ndarray:
    """A kitörölt sorok mögötti játékdoboz helyén éles szélű, egyszínű doboz, amely a magyar szöveget is
    körbefogja (a törlés után a doboz maradéka szakadozott). Az image-et helyben módosítja (már másolat)."""
    if plan is None or not plan.texts or not analysis.rows or not erased_rows:
        return image
    height, width = image.shape[:2]
    pad_x = round(NEAT_PAD_X_PX * width / REFERENCE_WIDTH)
    texts = _text_boxes(plan)
    painted: list[tuple[Box, tuple[int, int, int]]] = []
    for row in analysis.rows:
        if row.game_box is None or row.box not in erased_rows:
            continue
        game = row.game_box.box
        on_it = [text for text in texts if _overlap_y(text, game)]
        if not on_it:
            continue  # nincs rajta magyar szöveg: üres doboz nem kell
        around = [Box(text.left - pad_x, text.top, text.right + pad_x, text.bottom) for text in on_it]
        painted.append((union([game, *around]), row.game_box.color))
    for box in _merged([box for box, _color in painted]):
        color = next(color for part, color in painted if is_covered(part, [box]))
        image[max(0, box.top - NEAT_MARGIN_PX):min(height, box.bottom + NEAT_MARGIN_PX),
              max(0, box.left - NEAT_MARGIN_PX):min(width, box.right + NEAT_MARGIN_PX)] = color
    return image


def without_covers_over(plan: LayerPlan, erased_rows: list[Box]) -> LayerPlan:
    """A kitörölt sorokra már nem kell elmosott takarás."""
    def is_kept(job) -> bool:
        box = job.area if isinstance(job, CoverJob) else job.box
        return not any(_overlaps(box, row) for row in erased_rows)

    return replace(plan, covers=tuple(job for job in plan.covers if is_kept(job)))
