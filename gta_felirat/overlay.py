"""A magyar felirat kirajzolása a képre.

Az OpenCV putText nem tud ékezetet (ő, ű), ezért a szöveget Pillow-val
rajzoljuk egy átlátszó képre, egyszer szövegenként, és csak azt keverjük rá
minden képkockára.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .ocr import Box

MAX_TEXT_WIDTH_RATIO = 0.62
PANEL_PADDING_RATIO = 0.6
PANEL_ALPHA = 245
LINE_SPACING_RATIO = 0.25
OUTLINE_RATIO = 0.08
TEXT_COLOR = (255, 255, 255, 255)
OUTLINE_COLOR = (0, 0, 0, 255)
PANEL_COLOR = (10, 10, 10)


@dataclass(frozen=True)
class RenderedSubtitle:
    rgba: np.ndarray  # magasság × szélesség × 4, RGBA

    @property
    def width(self) -> int:
        return self.rgba.shape[1]

    @property
    def height(self) -> int:
        return self.rgba.shape[0]


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and font.getlength(candidate) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


@lru_cache(maxsize=8)
def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default(size)


@lru_cache(maxsize=64)
def render_subtitle(text: str, font_path: str, font_size: int, max_width: int,
                    min_panel: tuple[int, int] = (0, 0), opacity: int = PANEL_ALPHA) -> RenderedSubtitle:
    """Kirajzolja a szöveget egy sötét panelre; min_panel = (szél, mag) eltakarandó terület."""
    font = _load_font(font_path, font_size)
    lines = wrap_text(text, font, max_width) or [""]
    line_height = int(font_size * (1 + LINE_SPACING_RATIO))
    text_width = int(max(font.getlength(line) for line in lines))
    padding = int(font_size * PANEL_PADDING_RATIO)
    width = max(text_width + 2 * padding, min_panel[0])
    height = max(line_height * len(lines) + padding, min_panel[1])
    image = Image.new("RGBA", (width, height), PANEL_COLOR + (opacity,))
    draw = ImageDraw.Draw(image)
    outline = max(1, int(font_size * OUTLINE_RATIO))
    top = (height - line_height * len(lines)) // 2
    for index, line in enumerate(lines):
        x = (width - font.getlength(line)) / 2
        y = top + index * line_height
        draw.text((x, y), line, font=font, fill=TEXT_COLOR,
                  stroke_width=outline, stroke_fill=OUTLINE_COLOR)
    return RenderedSubtitle(rgba=np.asarray(image))


def centered_position(rendered: RenderedSubtitle, center_x: int, center_y: int,
                      frame_width: int, frame_height: int) -> tuple[int, int]:
    x = min(max(center_x - rendered.width // 2, 0), max(frame_width - rendered.width, 0))
    y = min(max(center_y - rendered.height // 2, 0), max(frame_height - rendered.height, 0))
    return x, y


def cover_box(frame: np.ndarray, box: Box, opacity: int) -> np.ndarray:
    """Új képet ad vissza, amelyen a doboz sötétítve van (az eredeti felirat eltakarása)."""
    result = frame.copy()
    top, bottom = max(box.top, 0), min(box.bottom, frame.shape[0])
    left, right = max(box.left, 0), min(box.right, frame.shape[1])
    if bottom <= top or right <= left:
        return result
    alpha = opacity / 255.0
    area = result[top:bottom, left:right].astype(np.float32)
    dark = np.array(PANEL_COLOR[::-1], np.float32)
    result[top:bottom, left:right] = (dark * alpha + area * (1 - alpha)).astype(np.uint8)
    return result


def subtitle_position(rendered: RenderedSubtitle, anchor: Box,
                      frame_width: int, frame_height: int) -> tuple[int, int]:
    """Az eredeti felirat közepére igazítja a panelt, a képen belül tartva."""
    center_x = (anchor.left + anchor.right) // 2
    center_y = (anchor.top + anchor.bottom) // 2
    return centered_position(rendered, center_x, center_y, frame_width, frame_height)


def blend(frame: np.ndarray, rendered: RenderedSubtitle, x: int, y: int) -> np.ndarray:
    """Új képet ad vissza, amelyre a feliratpanel rá van keverve (a frame BGR)."""
    result = frame.copy()
    height = min(rendered.height, frame.shape[0] - y)
    width = min(rendered.width, frame.shape[1] - x)
    if height <= 0 or width <= 0:
        return result
    overlay = rendered.rgba[:height, :width]
    alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
    overlay_bgr = overlay[:, :, 2::-1].astype(np.float32)
    target = result[y:y + height, x:x + width].astype(np.float32)
    result[y:y + height, x:x + width] = (overlay_bgr * alpha + target * (1 - alpha)).astype(np.uint8)
    return result
