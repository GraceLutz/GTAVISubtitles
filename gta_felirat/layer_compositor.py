"""A feliratréteg gyors kirajzolása az átlátszó ablak képpontjaiba (előszorzott BGRA), helyben.

A lebegőpontos út minden rétegnél lemásolta a teljes 1920×1080-as vásznat: egy rajzolás
25–45 ms volt. Itt minden réteg (takarás, panel, szöveg) egy kész, előszorzott 8 bites folt,
amelyet helyben teszünk a célra. A takarások a játékkép (Backdrop) cseréjéig, a szövegfoltok
a szöveg változásáig megmaradnak, így egy rajzolás 2–4 ms, változatlan játékképnél 1 ms alatt.
A kép legfeljebb 2/255-tel tér el a lebegőpontos úttól (kerekítés).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .frosted_cover import Backdrop, frosted_layer
from .layer_plan import BoxJob, CoverJob, LayerPlan, clip, union
from .ocr import Box
from .overlay import PANEL_COLOR, RenderedSubtitle

UINT8_SCALE = 1.0 / 255.0
OUTLINE_COLOR = (0, 200, 255, 255)  # BGRA: a húzott panel és a figyelt sáv kerete
OUTLINE_PX = 2
MAX_TEXT_PATCHES = 64


@dataclass(frozen=True)
class Patch:
    """Egy réteg, amely készen áll arra, hogy az előszorzott vászonra kerüljön."""
    pixels: np.ndarray  # H×W×4 uint8, előszorzott BGRA
    keep: np.ndarray  # H×W×4 uint8: 255 − alfa minden csatornában (ennyi marad az alatta lévőből)


def to_patch(color: np.ndarray, alpha: np.ndarray) -> Patch:
    """color: H×W×3 uint8 BGR (egyenes alfa), alpha: H×W uint8."""
    alpha3 = cv2.merge((alpha, alpha, alpha))
    premultiplied = cv2.multiply(color, alpha3, scale=UINT8_SCALE)
    pixels = cv2.merge((*cv2.split(premultiplied), alpha))
    inverse = cv2.bitwise_not(alpha)
    return Patch(pixels, cv2.merge((inverse, inverse, inverse, inverse)))


def put(target: np.ndarray, top: int, left: int, patch: Patch) -> None:
    """A folt a cél FÖLÉ kerül (mindkettő előszorzott), helyben: cél = folt + cél · (255 − alfa) / 255."""
    skip_y, skip_x = max(-top, 0), max(-left, 0)
    top, left = top + skip_y, left + skip_x
    height = min(patch.pixels.shape[0] - skip_y, target.shape[0] - top)
    width = min(patch.pixels.shape[1] - skip_x, target.shape[1] - left)
    if height <= 0 or width <= 0:
        return
    pixels = patch.pixels[skip_y:skip_y + height, skip_x:skip_x + width]
    keep = patch.keep[skip_y:skip_y + height, skip_x:skip_x + width]
    region = target[top:top + height, left:left + width]
    region[:] = cv2.add(cv2.multiply(region, keep, scale=UINT8_SCALE), pixels)


def box_patch(height: int, width: int, opacity: int) -> Patch:
    color = np.empty((height, width, 3), np.uint8)
    color[:] = PANEL_COLOR[::-1]
    return to_patch(color, np.full((height, width), opacity, np.uint8))


def text_patch(rendered: RenderedSubtitle) -> Patch:
    rgba = rendered.rgba
    return to_patch(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR), np.ascontiguousarray(rgba[:, :, 3]))


def cover_patch(source: np.ndarray, feather: int, strength: float) -> Patch | None:
    """A takarás foltja; None, ha teljesen átlátszó lenne (átlátszó panel, és nincs betű a területen)."""
    frosted, weight = frosted_layer(source, feather, strength)
    alpha = (weight[:, :, 0] * 255.0 + 0.5).astype(np.uint8)
    if cv2.countNonZero(alpha) == 0:
        return None
    return to_patch(np.clip(frosted + 0.5, 0, 255).astype(np.uint8), alpha)


def draw_outline(target: np.ndarray, box: Box) -> None:
    """Keret a doboz belső szélén (a célra vágva)."""
    area = target[max(box.top, 0):max(box.bottom, 0), max(box.left, 0):max(box.right, 0)]
    if area.size == 0:
        return
    area[:OUTLINE_PX] = OUTLINE_COLOR
    area[-OUTLINE_PX:] = OUTLINE_COLOR
    area[:, :OUTLINE_PX] = OUTLINE_COLOR
    area[:, -OUTLINE_PX:] = OUTLINE_COLOR


class LayerCompositor:
    """Egy LayerPlan kirajzolása az átlátszó ablak képpontjaiba; megjegyzi, mit rajzolt, és azt törli."""

    def __init__(self) -> None:
        self._backdrop: Backdrop | None = None  # ehhez a játékképhez készültek a tárolt takarások
        self._covers: dict[tuple[CoverJob, float], Patch | None] = {}
        self._texts: dict[int, tuple[RenderedSubtitle, Patch]] = {}
        self._drawn: Box | None = None  # a célon most kirajzolt terület

    def reset(self) -> None:
        """Új (üres) célfelület után: nincs mit törölni a régiről."""
        self._drawn = None

    def render(self, target: np.ndarray, plan: LayerPlan | None, backdrop: Backdrop | None,
               outlines: tuple[Box, ...] = ()) -> Box | None:
        """Helyben kirajzolja a tervet; a megváltozott területet adja vissza (None: semmi sem változott)."""
        height, width = target.shape[:2]
        old = self._drawn
        if old is not None:
            target[old.top:old.bottom, old.left:old.right] = 0
        boxes = self._draw(target, plan, backdrop) if plan is not None else []
        for box in outlines:
            draw_outline(target, box)
            boxes.append(box)
        new = clip(union(boxes), width, height) if boxes else None
        if new is not None and (new.width <= 0 or new.height <= 0):
            new = None
        self._drawn = new
        changed = [box for box in (old, new) if box is not None]
        return union(changed) if changed else None

    def _draw(self, target: np.ndarray, plan: LayerPlan, backdrop: Backdrop | None) -> list[Box]:
        if backdrop is not self._backdrop:
            self._covers, self._backdrop = {}, backdrop
        for job in plan.covers:
            if isinstance(job, CoverJob):
                patch = self._cover(job, plan.cover_strength, backdrop)
                if patch is not None:
                    put(target, job.area.top, job.area.left, patch)
            else:
                self._put_box(target, job)
        if plan.panel is not None and plan.panel.opacity > 0:  # az átlátszó panel semmit sem változtat
            self._put_box(target, plan.panel)
        for job in plan.texts:
            put(target, job.y, job.x, self._text(job.rendered))
        if plan.outline is not None:
            draw_outline(target, plan.outline)
        rect = plan.last_rect
        panel_box = Box(rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]) if rect is not None else None
        return [box for box in (plan.last_cover, panel_box) if box is not None]

    @staticmethod
    def _put_box(target: np.ndarray, job: BoxJob) -> None:
        if job.box.width > 0 and job.box.height > 0:
            put(target, job.box.top, job.box.left, box_patch(job.box.height, job.box.width, job.opacity))

    def _cover(self, job: CoverJob, strength: float, backdrop: Backdrop | None) -> Patch | None:
        key = (job, strength)
        if key in self._covers:
            return self._covers[key]
        patch = None
        if backdrop is not None:
            area = job.area
            offset_x, offset_y = area.left - backdrop.area.left, area.top - backdrop.area.top
            source = backdrop.image[offset_y:offset_y + area.height, offset_x:offset_x + area.width]
            # Ha a terv másik játékképhez készült (közben átméreteződött az ablak), nincs mit takarni
            if min(offset_x, offset_y) >= 0 and source.shape[:2] == (area.height, area.width):
                patch = cover_patch(source, job.feather, strength)
        self._covers[key] = patch
        return patch

    def _text(self, rendered: RenderedSubtitle) -> Patch:
        # A render_subtitle gyorsítótáraz: ugyanaz a szöveg ugyanazt az objektumot adja vissza
        entry = self._texts.get(id(rendered))
        if entry is None or entry[0] is not rendered:
            if len(self._texts) >= MAX_TEXT_PATCHES:
                self._texts.clear()
            entry = (rendered, text_patch(rendered))
            self._texts[id(rendered)] = entry
        return entry[1]
