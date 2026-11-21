"""A feliratpanel: kirajzolás, egérrel húzás, görgős méretezés.

- Bal gombbal húzva a panel oda kerül, ahová teszed (mentődik).
- Görgővel a panel fölött nő vagy csökken a betűméret.
- A Minta módban a panel bal vagy jobb szélét húzva a fordítás szélessége, a tetejét vagy alját húzva
  a panel magassága állítható (mentődik).
- Jobb gombbal visszaugrik az eredeti angol felirat helyére.

Hogy mit és hová kell rajzolni, azt a layer_plan dönti el; itt a terv lebegőpontos kirajzolása
van (capture card képkockára), a gyors, helyben rajzoló változat a layer_compositor.
"""

from dataclasses import replace
from typing import Callable

import cv2
import numpy as np

from .compositing import paste
from .config import (MAX_FONT_SIZE_RATIO, MAX_PANEL_HEIGHT_RATIO, MAX_PANEL_WIDTH_RATIO, MIN_FONT_SIZE_RATIO,
                     MIN_PANEL_HEIGHT_RATIO, MIN_PANEL_WIDTH_RATIO, AppConfig)
from .frosted_cover import Backdrop, frosted_layer
from .i18n import t
from .layer_compositor import draw_outline
from .layer_plan import BoxJob, CoverJob, LayerPlan, LayerView, Shown, ShownLine, make_plan
from .ocr import Box
from .overlay import blend, cover_box

__all__ = ["Shown", "ShownLine", "SubtitleLayer", "draw_plan", "wheel_delta"]

FONT_STEP = 0.002
EDGE_GRAB_PX = 12  # a panel szélétől ennyire még a szélét fogod meg (szélesítés), beljebb az egészet
WIDTH_DECIMALS = 4
MOVE_CURSOR = "fleur"
WIDEN_CURSOR = "sb_h_double_arrow"
HEIGHTEN_CURSOR = "sb_v_double_arrow"


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
        draw_outline(output, outline)
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
        self._drag_start: tuple[int, int] | None = None  # a panel közepe a megfogáskor
        self._widen_center_x: int | None = None  # a panel közepe, amíg a szélét húzzuk
        self._drag_width_ratio: float | None = None  # a húzott szélesség, elengedésig
        # A teteje vagy alja húzásakor: a helyben maradó (másik) él, és hogy a tetejét fogtuk-e meg
        self._fixed_edge_y: int | None = None
        self._is_top_grabbed = False
        self._drag_height_ratio: float | None = None  # a húzott magasság, elengedésig
        self._start_height_px = 0  # a panel magassága a megfogáskor
        self._drawn_height_px = 0  # a húzás közben ténylegesen kirajzolt magasság (a szöveg is megszabja)
        self._is_preview = False

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
    def is_preview(self) -> bool:
        """A legutóbbi rajzolás a mintapanel volt (nem valódi fordítás)."""
        return self._is_preview

    @property
    def is_dragging(self) -> bool:
        return (self._grab_offset is not None or self._widen_center_x is not None
                or self._fixed_edge_y is not None)

    def plan(self, frame_size: tuple[int, int], shown: Shown | None, hide_original: bool,
             preview_anchor: Box | None = None, backdrop: Backdrop | None = None) -> LayerPlan | None:
        """Mit kell kirajzolni és hová; megjegyzi a panel és a takarás helyét (egér, saját szöveg).

        preview_anchor: ha nincs felirat, mintaszöveg mutatja a panel helyét.
        backdrop: a takarás nélküli játékkép; ha megvan, az angol betűket ennek elmosott mása takarja.
        """
        width, height = frame_size
        self._frame_size = frame_size
        is_preview = shown is None and preview_anchor is not None
        self._is_preview = is_preview
        if is_preview:
            shown = Shown(t("layer.preview_text"), preview_anchor)
        view = LayerView(width, height, hide_original, backdrop.area if backdrop is not None else None,
                         self._custom_center(width, height), self.is_dragging, is_preview)
        config = self._config
        if self._drag_width_ratio is not None:
            config = replace(config, panel_width_ratio=self._drag_width_ratio)
        if self._drag_height_ratio is not None:
            config = replace(config, panel_height_ratio=self._drag_height_ratio)
        plan = make_plan(config, view, shown)
        if plan is not None and self._drag_height_ratio is not None:
            plan = self._keep_fixed_edge(plan, config, view, shown)
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

    def cursor_at(self, x: int, y: int) -> str:
        """Tk-kurzor neve az egér helyén ("": nem a panel fölött)."""
        if not self._contains(x, y, EDGE_GRAB_PX if self._is_preview else 0):
            return ""
        if self._is_at_side(x):
            return WIDEN_CURSOR
        return HEIGHTEN_CURSOR if self._is_at_top_or_bottom(y) else MOVE_CURSOR

    def handle_mouse(self, event: int, x: int, y: int, flags: int) -> bool:
        """True, ha az esemény a feliratpanelé volt."""
        if self._widen_center_x is not None:
            return self._handle_widening(event, x)
        if self._fixed_edge_y is not None:
            return self._handle_heightening(event, y)
        if event == cv2.EVENT_MOUSEMOVE and self._grab_offset is not None:
            self._drag_center = (x - self._grab_offset[0], y - self._grab_offset[1])
            return True
        if event == cv2.EVENT_LBUTTONUP and self._grab_offset is not None:
            self._finish_drag()
            return True
        if not self._contains(x, y, EDGE_GRAB_PX if self._is_preview else 0):
            return False
        if event == cv2.EVENT_LBUTTONDOWN and self._is_at_side(x):
            left, _top, width, _height = self._last_rect
            self._widen_center_x = left + width // 2
            self._drag_width_ratio = self._config.panel_width_ratio
        elif event == cv2.EVENT_LBUTTONDOWN and self._is_at_top_or_bottom(y):
            self._start_heightening(y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            left, top, width, height = self._last_rect
            center = (left + width // 2, top + height // 2)
            self._grab_offset = (x - center[0], y - center[1])
            self._drag_center = center
            self._drag_start = center
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
        has_moved = self._drag_center != self._drag_start
        self._grab_offset = None
        self._drag_center = None
        self._drag_start = None
        if not has_moved:  # egy kattintás még nem rögzíti a panelt a mostani helyén
            return
        self._on_change({"subtitle_x": min(max(center_x / width, 0.0), 1.0),
                         "subtitle_y": min(max(center_y / height, 0.0), 1.0)})

    def _handle_widening(self, event: int, x: int) -> bool:
        """A panel szélének húzása: a közepe marad, a szélessége követi az egeret."""
        frame_width = self._frame_size[0]
        if event == cv2.EVENT_MOUSEMOVE and frame_width > 0:
            ratio = 2 * abs(x - self._widen_center_x) / frame_width
            self._drag_width_ratio = round(min(max(ratio, MIN_PANEL_WIDTH_RATIO), MAX_PANEL_WIDTH_RATIO),
                                           WIDTH_DECIMALS)
        elif event == cv2.EVENT_LBUTTONUP:
            ratio, self._widen_center_x, self._drag_width_ratio = self._drag_width_ratio, None, None
            if ratio is not None and ratio != self._config.panel_width_ratio:
                self._on_change({"panel_width_ratio": ratio})
        return True

    def _keep_fixed_edge(self, plan: LayerPlan, config: AppConfig, view: LayerView, shown: Shown) -> LayerPlan:
        """Magasítás közben a helyben maradó él a valóban kirajzolt magassághoz igazodik.

        A panel a szövegnél nem lehet alacsonyabb: ilyenkor magasabb a húzottnál, és a közepét ehhez kell tenni.
        """
        if plan.last_rect is None:
            return plan  # nincs panel (pl. csak az angol sort takarjuk): nincs mihez igazodni
        drawn = plan.last_rect[3]
        self._drawn_height_px = drawn
        center_y = self._fixed_edge_y - drawn // 2 if self._is_top_grabbed else self._fixed_edge_y + drawn // 2
        if center_y == self._drag_center[1]:
            return plan
        self._drag_center = (self._drag_center[0], center_y)
        return make_plan(config, replace(view, custom_center=self._drag_center), shown)

    def _start_heightening(self, y: int) -> None:
        left, top, width, height = self._last_rect
        self._is_top_grabbed = abs(y - top) <= abs(y - (top + height))
        self._fixed_edge_y = top + height if self._is_top_grabbed else top
        self._drag_height_ratio = None  # mozdulásig a panel marad, ahogy volt
        self._start_height_px = height
        self._drawn_height_px = height
        self._drag_center = (left + width // 2, top + height // 2)

    def _handle_heightening(self, event: int, y: int) -> bool:
        """A panel tetejének vagy aljának húzása: a másik éle helyben marad, a megfogott követi az egeret."""
        frame_height = self._frame_size[1]
        if event == cv2.EVENT_MOUSEMOVE and frame_height > 0:
            pulled = self._fixed_edge_y - y if self._is_top_grabbed else y - self._fixed_edge_y
            ratio = round(min(max(pulled / frame_height, MIN_PANEL_HEIGHT_RATIO), MAX_PANEL_HEIGHT_RATIO),
                          WIDTH_DECIMALS)
            half = int(ratio * frame_height) // 2
            center_y = self._fixed_edge_y - half if self._is_top_grabbed else self._fixed_edge_y + half
            self._drag_height_ratio = ratio
            self._drawn_height_px = int(ratio * frame_height)  # a következő rajzolás pontosítja (szöveg)
            self._drag_center = (self._drag_center[0], center_y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._finish_heightening()
        return True

    def _finish_heightening(self) -> None:
        """Elengedéskor a panel ott marad, ahol húzás közben látszott: a magassága és a helye is mentődik."""
        ratio, center, drawn = self._drag_height_ratio, self._drag_center, self._drawn_height_px
        self._fixed_edge_y, self._drag_height_ratio, self._drag_center = None, None, None
        width, frame_height = self._frame_size
        if ratio is None or drawn == self._start_height_px:  # nem változott a magassága
            return
        # A szövegnél alacsonyabbra húzott panel a szöveg magasságú marad: ezt mentjük, nem a húzottat
        saved = round(min(max(ratio, drawn / frame_height), MAX_PANEL_HEIGHT_RATIO), WIDTH_DECIMALS)
        self._on_change({"panel_height_ratio": saved,
                         "subtitle_x": min(max(center[0] / width, 0.0), 1.0),
                         "subtitle_y": min(max(center[1] / frame_height, 0.0), 1.0)})

    def _is_at_top_or_bottom(self, y: int) -> bool:
        """A panel tetején vagy alján áll-e az egér (csak a Minta módban magasítható)."""
        if not self._is_preview or self._last_rect is None:
            return False
        _left, top, _width, height = self._last_rect
        return abs(y - top) <= EDGE_GRAB_PX or abs(y - (top + height - 1)) <= EDGE_GRAB_PX

    def _is_at_side(self, x: int) -> bool:
        """A panel bal vagy jobb szélén áll-e az egér (csak a Minta módban szélesíthető)."""
        if not self._is_preview or self._last_rect is None:
            return False
        left, _top, width, _height = self._last_rect
        return abs(x - left) <= EDGE_GRAB_PX or abs(x - (left + width)) <= EDGE_GRAB_PX

    def _contains(self, x: int, y: int, margin: int = 0) -> bool:
        if self._last_rect is None:
            return False
        left, top, width, height = self._last_rect
        return left - margin <= x < left + width + margin and top - margin <= y < top + height + margin
