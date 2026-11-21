"""A feliratpanel: kirajzolás, egérrel húzás, görgős méretezés.

- Bal gombbal húzva a panel oda kerül, ahová teszed (mentődik).
- Görgővel a panel fölött nő vagy csökken a betűméret.
- Jobb gombbal visszaugrik az eredeti angol felirat helyére.

Hogy mit és hová kell rajzolni, azt a layer_plan dönti el; itt a terv lebegőpontos kirajzolása
van (capture card képkockára), a gyors, helyben rajzoló változat a layer_compositor.
"""

from typing import Callable

import cv2
import numpy as np

from .compositing import paste
from .config import MAX_FONT_SIZE_RATIO, MIN_FONT_SIZE_RATIO, AppConfig
from .frosted_cover import Backdrop, frosted_layer
from .i18n import t
from .layer_plan import BoxJob, CoverJob, LayerPlan, LayerView, Shown, ShownLine, make_plan
from .ocr import Box
from .overlay import blend, cover_box

__all__ = ["Shown", "ShownLine", "SubtitleLayer", "draw_plan", "wheel_delta"]

FONT_STEP = 0.002
DRAG_OUTLINE_COLOR = (0, 200, 255, 255)  # BGR(A): az átlátszó rétegen is látszódjon


def wheel_delta(flags: int) -> int:
    """A görgő iránya az OpenCV flags felső 16 bitjében van, előjelesen."""
    delta = (flags >> 16) & 0xFFFF
    return delta - 0x10000 if delta >= 0x8000 else delta


def draw_plan(frame: np.ndarray, plan: LayerPlan, backdrop: Backdrop | None) -> np.ndarray:
    """A terv kirajzolása lebegőpontos keveréssel; új képet ad vissza, a frame-et nem módosítja."""
    output = frame
    for job in plan.covers:
        output = _draw_cover(output, job, plan.cover_strength, backdrop)
    if plan.panel is not None and plan.panel.opacity > 0:  # az átlátszó panel semmit sem változtat
        output = cover_box(output, plan.panel.box, plan.panel.opacity)
    for job in plan.texts:
        output = blend(output, job.rendered, job.x, job.y)
    outline = plan.outline
    if outline is not None:
        output = output.copy() if output is frame else output
        cv2.rectangle(output, (outline.left, outline.top), (outline.right - 1, outline.bottom - 1),
                      DRAG_OUTLINE_COLOR, 2)
    return output


def _draw_cover(frame: np.ndarray, job: CoverJob | BoxJob, strength: float,
                backdrop: Backdrop | None) -> np.ndarray:
    if isinstance(job, BoxJob):
        return cover_box(frame, job.box, job.opacity)
    if backdrop is None:
        return frame  # a terv játékképpel készült, de az közben elveszett: nincs mivel takarni
    area = job.area
    offset_x, offset_y = area.left - backdrop.area.left, area.top - backdrop.area.top
    source = backdrop.image[offset_y:offset_y + area.height, offset_x:offset_x + area.width]
    # Átlátszó panelnél a takarás is átlátszó: csak az angol betűk tűnnek el
    frosted, weight = frosted_layer(source, job.feather, strength)
    # Képkockán a széle az éles képbe, átlátszó rétegen az élő játékképbe olvad
    return paste(frame, area.top, area.left, frosted, weight)


class SubtitleLayer:
    def __init__(self, config: AppConfig, on_change: Callable[[dict], None]) -> None:
        self._config = config
        self._on_change = on_change
        self._last_rect: tuple[int, int, int, int] | None = None
        self._last_cover: Box | None = None
        self._frame_size = (0, 0)
        self._grab_offset: tuple[int, int] | None = None
        self._drag_center: tuple[int, int] | None = None

    def update_config(self, config: AppConfig) -> None:
        self._config = config

    @property
    def last_rect(self) -> tuple[int, int, int, int] | None:
        """A legutóbb kirajzolt panel (x, y, szélesség, magasság)."""
        return self._last_rect

    @property
    def last_cover(self) -> Box | None:
        """A legutóbb eltakart terület (az angol felirat helye margóval), vagy None."""
        return self._last_cover

    @property
    def is_dragging(self) -> bool:
        return self._grab_offset is not None

    def plan(self, frame_size: tuple[int, int], shown: Shown | None, hide_original: bool,
             preview_anchor: Box | None = None, backdrop: Backdrop | None = None) -> LayerPlan | None:
        """Mit kell kirajzolni és hová; megjegyzi a panel és a takarás helyét (egér, saját szöveg).

        preview_anchor: ha nincs felirat, mintaszöveg mutatja a panel helyét.
        backdrop: a takarás nélküli játékkép; ha megvan, az angol betűket ennek elmosott mása takarja.
        """
        width, height = frame_size
        self._frame_size = frame_size
        is_preview = shown is None and preview_anchor is not None
        if is_preview:
            shown = Shown(t("layer.preview_text"), preview_anchor)
        view = LayerView(width, height, hide_original, backdrop.area if backdrop is not None else None,
                         self._custom_center(width, height), self.is_dragging, is_preview)
        plan = make_plan(self._config, view, shown)
        self._last_rect = plan.last_rect if plan is not None else None
        self._last_cover = plan.last_cover if plan is not None else None
        return plan

    def draw(self, frame: np.ndarray, shown: Shown | None, hide_original: bool,
             preview_anchor: Box | None = None, backdrop: Backdrop | None = None) -> np.ndarray:
        """A réteg egy képkockán (capture card mód): új képet ad vissza; ha nincs mit rajzolni, a frame-et."""
        plan = self.plan((frame.shape[1], frame.shape[0]), shown, hide_original, preview_anchor, backdrop)
        return frame if plan is None else draw_plan(frame, plan, backdrop)

    def _custom_center(self, width: int, height: int) -> tuple[int, int] | None:
        if self._drag_center is not None:
            return self._drag_center
        if self._config.subtitle_x is None:
            return None
        return int(self._config.subtitle_x * width), int(self._config.subtitle_y * height)

    def handle_mouse(self, event: int, x: int, y: int, flags: int) -> bool:
        """True, ha az esemény a feliratpanelé volt."""
        if event == cv2.EVENT_MOUSEMOVE and self._grab_offset is not None:
            self._drag_center = (x - self._grab_offset[0], y - self._grab_offset[1])
            return True
        if event == cv2.EVENT_LBUTTONUP and self._grab_offset is not None:
            self._finish_drag()
            return True
        if not self._contains(x, y):
            return False
        if event == cv2.EVENT_LBUTTONDOWN:
            left, top, width, height = self._last_rect
            center = (left + width // 2, top + height // 2)
            self._grab_offset = (x - center[0], y - center[1])
            self._drag_center = center
        elif event == cv2.EVENT_MOUSEWHEEL:
            step = FONT_STEP if wheel_delta(flags) > 0 else -FONT_STEP
            ratio = min(max(self._config.font_size_ratio + step, MIN_FONT_SIZE_RATIO), MAX_FONT_SIZE_RATIO)
            self._on_change({"font_size_ratio": round(ratio, 4)})
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._on_change({"subtitle_x": None, "subtitle_y": None})
        else:
            return False
        return True

    def _finish_drag(self) -> None:
        width, height = self._frame_size
        center_x, center_y = self._drag_center
        self._grab_offset = None
        self._drag_center = None
        self._on_change({"subtitle_x": min(max(center_x / width, 0.0), 1.0),
                         "subtitle_y": min(max(center_y / height, 0.0), 1.0)})

    def _contains(self, x: int, y: int) -> bool:
        if self._last_rect is None:
            return False
        left, top, width, height = self._last_rect
        return left <= x < left + width and top <= y < top + height
