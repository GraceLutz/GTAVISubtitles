"""A magyar felirat kirajzolása a képre.

Az OpenCV putText nem tud ékezetet (ő, ű), ezért a szöveget Pillow-val
rajzoljuk egy átlátszó képre, egyszer szövegenként, és csak azt keverjük rá
minden képkockára.
"""

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .compositing import paste
from .ocr import Box

MAX_TEXT_WIDTH_RATIO = 0.62
PANEL_PADDING_RATIO = 0.6
PANEL_ALPHA = 245
LINE_SPACING_RATIO = 0.25
# GTA-stílus: vékony körvonal és halvány, kissé eltolt árnyék a vastag fekete keret helyett
OUTLINE_RATIO = 0.04
SHADOW_OFFSET_RATIO = 0.06
TEXT_COLOR = (255, 255, 255, 255)
OUTLINE_COLOR = (0, 0, 0, 255)
SHADOW_COLOR = (0, 0, 0, 170)
PANEL_COLOR = (10, 10, 10)
MAX_SPEAKER_WORDS = 3
# Sor eleji beszélőnév: „Locals:” → a név színes, a kettősponttól fehér
_SPEAKER_NAME = re.compile(r"^([^\W\d_](?:[^\W\d_]|[ .'-]){0,30}):\s")


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
    """Tördelés; a sortörés (beszélőnként külön sor) mindig új sort kezd."""
    lines: list[str] = []
    for paragraph in text.split("\n"):
        lines.extend(_wrap_paragraph(paragraph, font, max_width))
    return lines


def _wrap_paragraph(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
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


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    """„#7ECFD6” → (126, 207, 214)."""
    value = color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _speaker_split(line: str) -> tuple[str, str]:
    """(név, maradék) a sor elejéről; ('', sor), ha nem beszélővel kezdődik."""
    match = _SPEAKER_NAME.match(line)
    if not match or not line[0].isupper() or len(match.group(1).split()) > MAX_SPEAKER_WORDS:
        return "", line
    return match.group(1), line[len(match.group(1)):]


def _draw_line(draw: ImageDraw.ImageDraw, position: tuple[float, int], line: str,
               font: ImageFont.FreeTypeFont, speaker_color: tuple[int, int, int] | None) -> None:
    """Árnyék, majd a szöveg; a beszélő neve speaker_color színű, a többi fehér."""
    x, y = position
    size = font.size
    outline = max(1, int(size * OUTLINE_RATIO))
    shadow = max(1, int(size * SHADOW_OFFSET_RATIO))
    draw.text((x + shadow, y + shadow), line, font=font, fill=SHADOW_COLOR)
    name, rest = _speaker_split(line) if speaker_color else ("", line)
    if name:
        draw.text((x, y), name, font=font, fill=speaker_color + (255,),
                  stroke_width=outline, stroke_fill=OUTLINE_COLOR)
        x += font.getlength(name)
    draw.text((x, y), rest, font=font, fill=TEXT_COLOR, stroke_width=outline, stroke_fill=OUTLINE_COLOR)


@lru_cache(maxsize=64)
def render_subtitle(text: str, font_path: str, font_size: int, max_width: int,
                    min_panel: tuple[int, int] = (0, 0), opacity: int = PANEL_ALPHA,
                    speaker_color: tuple[int, int, int] | None = None) -> RenderedSubtitle:
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
    top = (height - line_height * len(lines)) // 2
    for index, line in enumerate(lines):
        x = (width - font.getlength(line)) / 2
        _draw_line(draw, (x, top + index * line_height), line, font, speaker_color)
    return RenderedSubtitle(rgba=np.asarray(image))


def centered_position(rendered: RenderedSubtitle, center_x: int, center_y: int,
                      frame_width: int, frame_height: int) -> tuple[int, int]:
    x = min(max(center_x - rendered.width // 2, 0), max(frame_width - rendered.width, 0))
    y = min(max(center_y - rendered.height // 2, 0), max(frame_height - rendered.height, 0))
    return x, y


def cover_box(frame: np.ndarray, box: Box, opacity: int) -> np.ndarray:
    """Új képet ad vissza, amelyen a doboz sötétítve van (az eredeti felirat eltakarása)."""
    top, bottom = max(box.top, 0), min(box.bottom, frame.shape[0])
    left, right = max(box.left, 0), min(box.right, frame.shape[1])
    if bottom <= top or right <= left:
        return frame.copy()
    alpha = np.full((bottom - top, right - left, 1), opacity / 255.0, np.float32)
    return paste(frame, top, left, np.array(PANEL_COLOR[::-1], np.float32), alpha)


def subtitle_position(rendered: RenderedSubtitle, anchor: Box,
                      frame_width: int, frame_height: int) -> tuple[int, int]:
    """Az eredeti felirat közepére igazítja a panelt, a képen belül tartva."""
    center_x = (anchor.left + anchor.right) // 2
    center_y = (anchor.top + anchor.bottom) // 2
    return centered_position(rendered, center_x, center_y, frame_width, frame_height)


def beside_original_position(rendered: RenderedSubtitle, anchor: Box, frame_width: int,
                             frame_height: int, gap: int) -> tuple[int, int]:
    """Az angol felirat alá teszi a fordítást; ha ott nincs hely, fölé, hogy egyik se takarja a másikat."""
    center_x = (anchor.left + anchor.right) // 2
    x = min(max(center_x - rendered.width // 2, 0), max(frame_width - rendered.width, 0))
    below = anchor.bottom + gap
    if below + rendered.height <= frame_height:
        return x, below
    return x, max(anchor.top - gap - rendered.height, 0)


def blend(frame: np.ndarray, rendered: RenderedSubtitle, x: int, y: int) -> np.ndarray:
    """Új képet ad vissza, amelyre a feliratpanel rá van keverve (a frame BGR vagy átlátszó BGRA)."""
    height = min(rendered.height, frame.shape[0] - y)
    width = min(rendered.width, frame.shape[1] - x)
    if height <= 0 or width <= 0:
        return frame.copy()
    overlay = rendered.rgba[:height, :width]
    alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
    return paste(frame, y, x, overlay[:, :, 2::-1], alpha)
