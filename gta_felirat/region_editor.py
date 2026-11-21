"""A figyelt sáv (subtitle_region) egérrel: belül megfogva áthelyezhető, a szélénél és a sarkánál méretezhető.

Csak beállításkor él (a „Minta” kapcsoló / az r gomb): ilyenkor a sáv látszik és megfogható. Húzás közben az
új sávot mutatjuk (region), elengedéskor egyszer mentjük (on_change). Az egéresemények az OpenCV kódjait
használják (a Tk-ablak is ezekre fordítja le a sajátjait), a koordináták a kép képpontjai.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cv2

from .config import Region
from .ocr import Box, region_to_box

EDGE_GRAB_PX = 12  # ennyire a szélétől még a szélét fogod meg (méretezés), beljebb az egészet (áthelyezés)
MIN_WIDTH_RATIO = 0.1  # ennél keskenyebb vagy alacsonyabb sávban egy feliratsor sem férne el
MIN_HEIGHT_RATIO = 0.05
REGION_DECIMALS = 4

# A megfogott rész: melyik élek mozognak (bal, felső, jobb, alsó)
MOVE = (True, True, True, True)
CURSORS = {  # Tk-kurzor a megfogható részek fölött
    (True, False, False, False): "sb_h_double_arrow", (False, False, True, False): "sb_h_double_arrow",
    (False, True, False, False): "sb_v_double_arrow", (False, False, False, True): "sb_v_double_arrow",
    (True, True, False, False): "size_nw_se", (False, False, True, True): "size_nw_se",
    (False, True, True, False): "size_ne_sw", (True, False, False, True): "size_ne_sw",
}
MOVE_CURSOR = "fleur"


@dataclass(frozen=True)
class _Drag:
    edges: tuple[bool, bool, bool, bool]  # bal, felső, jobb, alsó: melyik mozog
    start: tuple[int, int]
    region: Region  # a sáv a megfogáskor


class RegionEditor:
    """A figyelt sáv egérrel; csak akkor fogad eseményt, ha be van kapcsolva (enabled)."""

    def __init__(self, region: Region, on_change: Callable[[dict], None]) -> None:
        self._region = region
        self._on_change = on_change
        self._frame_size = (0, 0)
        self._drag: _Drag | None = None
        self._dragged: Region | None = None
        self.enabled = False

    @property
    def region(self) -> Region:
        """A mostani sáv: húzás közben az új, egyébként a beállított."""
        return self._dragged or self._region

    @property
    def is_dragging(self) -> bool:
        return self._drag is not None

    def update_region(self, region: Region) -> None:
        """A beállítás megváltozott (pl. mentés után, vagy a config.json-ból)."""
        self._region = region

    def box(self, frame_width: int, frame_height: int) -> Box:
        self._frame_size = (frame_width, frame_height)
        return region_to_box(self.region, frame_width, frame_height)

    def grabs_edge(self, x: int, y: int) -> bool:
        """A sáv szélén vagy sarkán áll-e az egér (ez elsőbbséget kap a rajta fekvő panellel szemben)."""
        edges = self._grabbed_edges(x, y) if self.enabled else None
        return edges is not None and edges != MOVE

    def cursor_at(self, x: int, y: int) -> str:
        """Tk-kurzor neve az egér helyén ("" : a sávon kívül, nem a miénk)."""
        if not self.enabled:
            return ""
        edges = self._grabbed_edges(x, y)
        if edges is None:
            return ""
        return MOVE_CURSOR if edges == MOVE else CURSORS[edges]

    def handle_mouse(self, event: int, x: int, y: int) -> bool:
        """True, ha az esemény a sávé volt."""
        if self._drag is not None:
            if event == cv2.EVENT_MOUSEMOVE:
                self._dragged = self._moved(self._drag, x, y)
                return True
            if event == cv2.EVENT_LBUTTONUP:
                self._finish()
                return True
        if not self.enabled or event != cv2.EVENT_LBUTTONDOWN:
            return False
        edges = self._grabbed_edges(x, y)
        if edges is None:
            return False
        self._drag = _Drag(edges, (x, y), self._region)
        self._dragged = self._region
        return True

    def _grabbed_edges(self, x: int, y: int) -> tuple[bool, bool, bool, bool] | None:
        width, height = self._frame_size
        if width <= 0 or height <= 0:
            return None
        box = region_to_box(self.region, width, height)
        if not (box.left - EDGE_GRAB_PX <= x <= box.right + EDGE_GRAB_PX
                and box.top - EDGE_GRAB_PX <= y <= box.bottom + EDGE_GRAB_PX):
            return None
        edges = (abs(x - box.left) <= EDGE_GRAB_PX, abs(y - box.top) <= EDGE_GRAB_PX,
                 abs(x - box.right) <= EDGE_GRAB_PX, abs(y - box.bottom) <= EDGE_GRAB_PX)
        if not any(edges):
            return MOVE if box.left < x < box.right and box.top < y < box.bottom else None
        return edges

    def _moved(self, drag: _Drag, x: int, y: int) -> Region:
        width, height = self._frame_size
        dx, dy = (x - drag.start[0]) / width, (y - drag.start[1]) / height
        old = drag.region
        if drag.edges == MOVE:
            dx = min(max(dx, -old.left), 1.0 - old.right)
            dy = min(max(dy, -old.top), 1.0 - old.bottom)
            return _rounded(Region(old.left + dx, old.top + dy, old.right + dx, old.bottom + dy))
        move_left, move_top, move_right, move_bottom = drag.edges
        left = min(max(old.left + dx, 0.0), old.right - MIN_WIDTH_RATIO) if move_left else old.left
        right = max(min(old.right + dx, 1.0), old.left + MIN_WIDTH_RATIO) if move_right else old.right
        top = min(max(old.top + dy, 0.0), old.bottom - MIN_HEIGHT_RATIO) if move_top else old.top
        bottom = max(min(old.bottom + dy, 1.0), old.top + MIN_HEIGHT_RATIO) if move_bottom else old.bottom
        return _rounded(Region(left, top, right, bottom))

    def _finish(self) -> None:
        region, self._drag, self._dragged = self._dragged, None, None
        if region is not None and region != self._region:
            self._region = region
            self._on_change({"subtitle_region": region})


def route_mouse(editor: RegionEditor, layer, event: int, x: int, y: int, flags: int) -> bool:
    """Az egér a sávé vagy a feliratpanelé (layer: SubtitleLayer, vagy None, ha most nem kaphatja meg).

    Amelyiket épp húzzuk, az kapja a mozgást; a sáv szélei elsőbbek a rajtuk fekvő panelnél (különben a
    panel alatt nem lehetne méretezni a sávot); egyébként előbb a panel, aztán a sáv belseje.
    """
    if editor.is_dragging:
        return editor.handle_mouse(event, x, y)
    if layer is not None and layer.is_dragging:
        return layer.handle_mouse(event, x, y, flags)
    if editor.grabs_edge(x, y) and editor.handle_mouse(event, x, y):
        return True
    if layer is not None and layer.handle_mouse(event, x, y, flags):
        return True
    return editor.handle_mouse(event, x, y)


def edge_cursor(editor: RegionEditor, layer, x: int, y: int) -> str:
    """A kurzor ugyanabban a sorrendben, ahogy a route_mouse dönt."""
    if editor.grabs_edge(x, y):
        return editor.cursor_at(x, y)
    return (layer.cursor_at(x, y) if layer is not None else "") or editor.cursor_at(x, y)


def _rounded(region: Region) -> Region:
    return Region(*(min(max(round(value, REGION_DECIMALS), 0.0), 1.0)
                    for value in (region.left, region.top, region.right, region.bottom)))
