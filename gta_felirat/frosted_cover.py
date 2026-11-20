"""Az angol felirat igényes eltakarása: elmosott, kissé sötétített „matt üveg” a játékképből.

A takarás közepén a játékkép elmosva látszik (a fehér betűket az elmosásból kihagyjuk, így
nem marad utánuk világos folt), a széle felé pedig fokozatosan visszaáll az éles képre,
így nincs kemény kerete. A forráskép a takarás nélküli játékkép (Remote Play módban az
ablak saját tartalma, capture card módban maga a képkocka).
"""

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from .ocr import Box

TEXT_MASK_LEVEL = 170  # ennél világosabb pixel a felirat betűje (a GTA-felirat fehér)
TEXT_MASK_GROW_PX = 3  # a betűk fekete körvonala is maradjon ki az elmosásból
MIN_BLUR_SIGMA = 6.0
BLUR_SIGMA_RATIO = 0.35  # a takarás magasságához képest
BLUR_DOWNSCALE = 4  # ennyiszer kisebb képen mosunk el
DARKEN = 0.72  # a sötétebb háttéren jobban olvasható a fordítás
MIN_TEXT_FREE_WEIGHT = 0.05  # ennél kevesebb betűmentes környezetnél sima elmosás


@dataclass(frozen=True)
class Backdrop:
    image: np.ndarray  # a játékkép takarás nélkül (BGR)
    area: Box  # hol van ez a kép a rajzolt képen


def cover_area(anchor: Box, margin: int, limits: Box) -> Box:
    """Az angol szöveg doboza margóval kibővítve, a forráskép határain belül."""
    return Box(left=max(anchor.left - margin, limits.left), top=max(anchor.top - margin, limits.top),
               right=min(anchor.right + margin, limits.right), bottom=min(anchor.bottom + margin, limits.bottom))


def _blur_without_text(source: np.ndarray, sigma: float) -> np.ndarray:
    # Normált elmosás: a betűpixelek nem számítanak bele, így a háttérszín folytatódik alattuk
    brightest = cv2.max(cv2.max(source[:, :, 0], source[:, :, 1]), source[:, :, 2])  # ~20× gyorsabb a numpy-nál
    text = (brightest >= TEXT_MASK_LEVEL).astype(np.uint8)
    kernel = np.ones((2 * TEXT_MASK_GROW_PX + 1, 2 * TEXT_MASK_GROW_PX + 1), np.uint8)
    keep = 1.0 - cv2.dilate(text, kernel).astype(np.float32)
    # Az erős elmosás kicsinyített képen ugyanígy néz ki, csak töredék idő (~10× gyorsabb)
    height, width = source.shape[:2]
    small_size = (max(1, width // BLUR_DOWNSCALE), max(1, height // BLUR_DOWNSCALE))
    small_sigma = max(sigma / BLUR_DOWNSCALE, 1.0)
    image = cv2.resize(source, small_size, interpolation=cv2.INTER_AREA).astype(np.float32)
    small_keep = cv2.resize(keep, small_size, interpolation=cv2.INTER_AREA)
    weighted = cv2.GaussianBlur(image * small_keep[:, :, None], (0, 0), small_sigma)
    weight = cv2.GaussianBlur(small_keep, (0, 0), small_sigma)[:, :, None]
    plain = cv2.GaussianBlur(image, (0, 0), small_sigma)
    blurred = np.where(weight > MIN_TEXT_FREE_WEIGHT, weighted / np.maximum(weight, MIN_TEXT_FREE_WEIGHT), plain)
    darkened = (blurred * DARKEN).astype(np.float32)  # kicsiben sötétítünk: kevesebb pixel
    return cv2.resize(darkened, (width, height), interpolation=cv2.INTER_LINEAR)


@lru_cache(maxsize=16)
def _edge_weight(height: int, width: int, feather: int) -> np.ndarray:
    """0 a széleken, 1 belül; a kettő között lágy (smoothstep) átmenet (csak olvasható, gyorsítótárazott)."""
    ys = np.arange(height, dtype=np.float32)
    xs = np.arange(width, dtype=np.float32)
    to_edge = np.minimum(np.minimum(ys, height - 1 - ys)[:, None], np.minimum(xs, width - 1 - xs)[None, :])
    linear = np.clip(to_edge / max(feather, 1), 0.0, 1.0)
    weight = linear * linear * (3.0 - 2.0 * linear)
    weight.setflags(write=False)
    return weight


def frosted_patch(source: np.ndarray, feather: int) -> np.ndarray:
    """A forrásképpel azonos méretű takarás: belül elmosott és sötétített, a szélén az éles kép."""
    height, width = source.shape[:2]
    sigma = max(MIN_BLUR_SIGMA, height * BLUR_SIGMA_RATIO)
    frosted = _blur_without_text(source, sigma)
    weight = _edge_weight(height, width, feather)
    mixed = cv2.blendLinear(frosted, source.astype(np.float32), weight, 1.0 - weight)
    return cv2.convertScaleAbs(mixed)  # kerekít és 0–255 közé szorít
