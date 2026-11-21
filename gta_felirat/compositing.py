"""Képrészlet rákeverése egy célképre: BGR képkockára vagy átlátszó BGRA vászonra.

A capture card módban a játék képkockájára (BGR) rajzolunk. Remote Play módban egy
átlátszó rétegre (BGRA, egyenes alfa): ott a rákevert rész átlátszósága is megmarad, így
a takarás széle valóban beleolvad az alatta futó élő játékképbe.
"""

import numpy as np

MAX_CHANNEL = 255.0


def composite(target: np.ndarray, color: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Új képrészlet: color (BGR, float vagy uint8) alpha (0–1, H×W×1) erősséggel a target fölé.

    A target lehet BGR vagy BGRA; a visszaadott kép ugyanolyan alakú és uint8.
    """
    alpha = alpha.astype(np.float32)
    color = np.asarray(color, np.float32)
    if target.shape[2] == 3:
        mixed = color * alpha + target.astype(np.float32) * (1.0 - alpha)
        return np.clip(mixed + 0.5, 0, MAX_CHANNEL).astype(np.uint8)
    below_alpha = target[:, :, 3:4].astype(np.float32) / MAX_CHANNEL
    below_weight = below_alpha * (1.0 - alpha)
    out_alpha = alpha + below_weight
    color_sum = color * alpha + target[:, :, :3].astype(np.float32) * below_weight
    out_color = np.divide(color_sum, out_alpha, out=np.zeros_like(color_sum), where=out_alpha > 0)
    result = np.empty(target.shape, np.uint8)
    result[:, :, :3] = np.clip(out_color + 0.5, 0, MAX_CHANNEL)
    result[:, :, 3:4] = np.clip(out_alpha * MAX_CHANNEL + 0.5, 0, MAX_CHANNEL)
    return result


def paste(frame: np.ndarray, top: int, left: int, color: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Új kép, amelyen a (top, left) sarkú téglalapra color van keverve; a frame-et nem módosítja."""
    height, width = alpha.shape[:2]
    result = frame.copy()
    region = result[top:top + height, left:left + width]
    result[top:top + height, left:left + width] = composite(region, color, alpha)
    return result
