"""A webkamerás kép egérrel húzható és görgővel méretezhető."""

from typing import Callable

import cv2

from .facecam import FacecamStyle, facecam_rect, moved_to, resized_by
from .subtitle_layer import wheel_delta

WHEEL_RESIZE_STEP = 0.02


class FacecamDragger:
    """Az OpenCV egéreseményeiből számolja a kép új helyét; elengedéskor ment."""

    def __init__(self, style: FacecamStyle, on_save: Callable[[FacecamStyle], None]) -> None:
        self._style = style
        self._on_save = on_save
        self._frame_size = (0, 0)
        self._cam_size = (16, 9)
        self._grab_offset: tuple[int, int] | None = None

    @property
    def style(self) -> FacecamStyle:
        return self._style

    @property
    def is_dragging(self) -> bool:
        return self._grab_offset is not None

    def set_geometry(self, frame_width: int, frame_height: int, cam_width: int, cam_height: int) -> None:
        """A legutóbb kirajzolt méretek; ezekből tudjuk, hova kattintott a felhasználó."""
        self._frame_size = (frame_width, frame_height)
        self._cam_size = (cam_width, cam_height)

    def handle_mouse(self, event: int, x: int, y: int, flags: int) -> None:
        if self._frame_size[0] == 0:
            return  # még nem rajzoltunk ki semmit
        if event == cv2.EVENT_LBUTTONDOWN and self._contains(x, y):
            left, top, _, _ = self._rect()
            self._grab_offset = (x - left, y - top)
        elif event == cv2.EVENT_MOUSEMOVE and self._grab_offset is not None:
            self._style = moved_to(self._style, x - self._grab_offset[0], y - self._grab_offset[1],
                                   *self._frame_size)
        elif event == cv2.EVENT_LBUTTONUP and self._grab_offset is not None:
            self._grab_offset = None
            self._style = self._clamped()
            self._on_save(self._style)
        elif event == cv2.EVENT_MOUSEWHEEL and self._contains(x, y):
            step = WHEEL_RESIZE_STEP if wheel_delta(flags) > 0 else -WHEEL_RESIZE_STEP
            self._style = resized_by(self._style, step)
            self._on_save(self._style)

    def _rect(self) -> tuple[int, int, int, int]:
        return facecam_rect(*self._frame_size, *self._cam_size, self._style)

    def _contains(self, x: int, y: int) -> bool:
        left, top, width, height = self._rect()
        return left <= x < left + width and top <= y < top + height

    def _clamped(self) -> FacecamStyle:
        """A képen kívülre húzott pozíciót a ténylegesen kirajzolt helyre igazítja."""
        left, top, _, _ = self._rect()
        return moved_to(self._style, left, top, *self._frame_size)
