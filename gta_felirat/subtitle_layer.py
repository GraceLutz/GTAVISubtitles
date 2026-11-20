"""A feliratpanel: kirajzolás, egérrel húzás, görgős méretezés.

- Bal gombbal húzva a panel oda kerül, ahová teszed (mentődik).
- Görgővel a panel fölött nő vagy csökken a betűméret.
- Jobb gombbal visszaugrik az eredeti angol felirat helyére.
"""

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .config import MAX_FONT_SIZE_RATIO, MIN_FONT_SIZE_RATIO, AppConfig
from .ocr import Box
from .overlay import (MAX_TEXT_WIDTH_RATIO, beside_original_position, blend, centered_position, cover_box, render_subtitle,
                      subtitle_position)

FONT_STEP = 0.002
MIN_FONT_PX = 12
PREVIEW_TEXT = "Feliratpanel – húzd egérrel, görgővel méretezd, jobb klikk: eredeti hely"
DRAG_OUTLINE_COLOR = (0, 200, 255)
ORIGINAL_COVER_OPACITY = 255


def wheel_delta(flags: int) -> int:
    """A görgő iránya az OpenCV flags felső 16 bitjében van, előjelesen."""
    delta = (flags >> 16) & 0xFFFF
    return delta - 0x10000 if delta >= 0x8000 else delta


@dataclass(frozen=True)
class Shown:
    text: str
    anchor: Box  # az eredeti angol felirat helye


class SubtitleLayer:
    def __init__(self, config: AppConfig, on_change: Callable[[dict], None]) -> None:
        self._config = config
        self._on_change = on_change
        self._last_rect: tuple[int, int, int, int] | None = None
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
    def is_dragging(self) -> bool:
        return self._grab_offset is not None

    def draw(self, frame: np.ndarray, shown: Shown | None, hide_original: bool,
             preview_anchor: Box | None = None) -> np.ndarray:
        """preview_anchor: ha nincs felirat, mintaszöveg mutatja a panel helyét."""
        height, width = frame.shape[:2]
        self._frame_size = (width, height)
        if shown is None and preview_anchor is not None:
            shown = Shown(PREVIEW_TEXT, preview_anchor)
        if shown is None:
            self._last_rect = None
            return frame
        font_size = max(MIN_FONT_PX, int(height * self._config.font_size_ratio))
        center = self._custom_center(width, height)
        if center is None:
            return self._draw_at_original(frame, shown, hide_original, font_size)
        # Az angol szöveget teljesen takarjuk, akármilyen átlátszó a magyar panel
        output = cover_box(frame, shown.anchor, ORIGINAL_COVER_OPACITY) if hide_original else frame
        rendered = render_subtitle(shown.text, self._config.font_path, font_size,
                                   int(width * MAX_TEXT_WIDTH_RATIO), (0, 0), self._config.panel_opacity)
        x, y = centered_position(rendered, center[0], center[1], width, height)
        return self._finish(output, rendered, x, y)

    def _draw_at_original(self, frame: np.ndarray, shown: Shown, hide_original: bool,
                          font_size: int) -> np.ndarray:
        height, width = frame.shape[:2]
        anchor = shown.anchor
        if hide_original:
            min_panel = (anchor.width + font_size, anchor.height + font_size // 2)
        else:
            min_panel = (0, 0)
        rendered = render_subtitle(shown.text, self._config.font_path, font_size,
                                   int(width * MAX_TEXT_WIDTH_RATIO), min_panel, self._config.panel_opacity)
        if hide_original:
            x, y = subtitle_position(rendered, anchor, width, height)
        else:
            x, y = beside_original_position(rendered, anchor, width, height, gap=font_size // 2)
        return self._finish(frame, rendered, x, y)

    def _finish(self, frame: np.ndarray, rendered, x: int, y: int) -> np.ndarray:
        self._last_rect = (x, y, rendered.width, rendered.height)
        output = blend(frame, rendered, x, y)
        if self.is_dragging:
            cv2.rectangle(output, (x, y), (x + rendered.width - 1, y + rendered.height - 1),
                          DRAG_OUTLINE_COLOR, 2)
        return output

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
