"""Webkamerás kép (facecam) oda, ahová a felhasználó egérrel húzta."""

from dataclasses import dataclass, replace

import cv2
import numpy as np

from .config import MAX_WEBCAM_WIDTH_RATIO, MIN_WEBCAM_WIDTH_RATIO

BORDER_PX = 3
BORDER_COLOR = (255, 255, 255)
DRAG_BORDER_COLOR = (0, 200, 255)


@dataclass(frozen=True)
class FacecamStyle:
    x: float  # a kép bal felső sarka a játékkép arányában (0.0–1.0)
    y: float
    width_ratio: float
    mirror: bool


def facecam_rect(frame_width: int, frame_height: int, cam_width: int, cam_height: int,
                 style: FacecamStyle) -> tuple[int, int, int, int]:
    """(x, y, szélesség, magasság) a kamera képarányával, mindig a képen belül."""
    width = min(max(1, int(frame_width * style.width_ratio)), frame_width)
    height = min(max(1, int(width * cam_height / cam_width)), frame_height)
    x = min(max(int(style.x * frame_width), 0), frame_width - width)
    y = min(max(int(style.y * frame_height), 0), frame_height - height)
    return x, y, width, height


def moved_to(style: FacecamStyle, left_px: int, top_px: int,
             frame_width: int, frame_height: int) -> FacecamStyle:
    return replace(style, x=min(max(left_px / frame_width, 0.0), 1.0),
                   y=min(max(top_px / frame_height, 0.0), 1.0))


def resized_by(style: FacecamStyle, step: float) -> FacecamStyle:
    ratio = min(max(style.width_ratio + step, MIN_WEBCAM_WIDTH_RATIO), MAX_WEBCAM_WIDTH_RATIO)
    return replace(style, width_ratio=ratio)


def place_facecam(frame: np.ndarray, cam_frame: np.ndarray, style: FacecamStyle,
                  highlighted: bool = False) -> np.ndarray:
    """Új képet ad vissza, rajta a keretezett webkamerás képpel."""
    frame_height, frame_width = frame.shape[:2]
    cam_height, cam_width = cam_frame.shape[:2]
    x, y, width, height = facecam_rect(frame_width, frame_height, cam_width, cam_height, style)
    resized = cv2.resize(cam_frame, (width, height), interpolation=cv2.INTER_AREA)
    if style.mirror:
        resized = cv2.flip(resized, 1)
    result = frame.copy()
    result[y:y + height, x:x + width] = resized
    color = DRAG_BORDER_COLOR if highlighted else BORDER_COLOR
    cv2.rectangle(result, (x, y), (x + width - 1, y + height - 1), color, BORDER_PX)
    return result
