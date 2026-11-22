"""A tiszta kép egy képkockájának előkészítése: a feliratsorok megkeresése és kitörlése (háttérszálon).

A rögzítéskor fut (analyse), mert a sorkeresés és a törlés soronként ~20 ms; a kép úgyis késleltetve kerül ki,
addigra kész. Megjelenítéskor csak azok a sorok kerülnek a képre kitörölve, amelyeket a választott állapot
fordít vagy takar (with_erased_rows); a többi (pl. rendszámtábla) érintetlen marad.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from .clean_history import FrameRow, is_covered, text_boxes
from .layer_plan import CoverJob
from .ocr import region_to_box
from .row_finder import find_rows, is_wide_enough
from .text_eraser import erase_row

if TYPE_CHECKING:
    import numpy as np

    from .config import AppConfig
    from .layer_plan import LayerPlan, Shown
    from .ocr import Box


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
    return FrameAnalysis([FrameRow(erase_row(erased, row.box), row.sure) for row in found], erased)


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


def without_covers_over(plan: LayerPlan, erased_rows: list[Box]) -> LayerPlan:
    """A kitörölt sorokra már nem kell elmosott takarás."""
    def is_kept(job) -> bool:
        box = job.area if isinstance(job, CoverJob) else job.box
        return not any(_overlaps(box, row) for row in erased_rows)

    return replace(plan, covers=tuple(job for job in plan.covers if is_kept(job)))
