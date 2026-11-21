"""Vízszintes élek továbbvitele a kitörölt feliratsor alatt.

A sima kitöltés (TELEA) minden irányból egyformán mos össze: ha a sor alatt egy tárgy vízszintes éle fut át
(lökhárító, lámpa alja), az él fölötti és alatti szín összefolyik, maszat marad. Ahol a törölt képpont fölötti
és alatti vége nagyon eltér, a bal és jobb oldali viszont egyezik, ott él fut át vízszintesen: ott a két oldal
közötti vízszintes interpoláció a helyes kitöltés.
"""

from __future__ import annotations

import cv2
import numpy as np

EDGE_MARGIN = 30.0  # a függőleges eltérés ennyivel legyen nagyobb a vízszintesnél (a háttér szemcséje ennyi)
EDGE_RANGE = 60.0  # e fölött már teljesen a vízszintes interpoláció számít
MAX_SPAN_PX = 50  # 1920 széles képen: ennél hosszabb törölt szakaszon át nem húzunk vízszintes élt (más tárgyra érne)
REFERENCE_WIDTH = 1920
SMOOTH_SIGMA = 1.5  # a végpontok szemcséje ne csíkozza a kitöltést
EVIDENCE_SIGMA = 3.0  # az él-döntés ne ugráljon képpontonként
END_MARGIN_PX = 2


def _ends(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Soronként minden képponthoz a legközelebbi bal és jobb oldali nem törölt képpont: (bal, jobb, bal érték,
    jobb érték). Ahol az egyik oldalon nincs ilyen, a bal -1, a jobb a szélesség."""
    height, width = mask.shape
    columns = np.broadcast_to(np.arange(width), (height, width))
    left = np.maximum.accumulate(np.where(mask, -1, columns), axis=1)
    right = np.minimum.accumulate(np.where(mask, width, columns)[:, ::-1], axis=1)[:, ::-1]
    rows = np.broadcast_to(np.arange(height)[:, None], (height, width))
    left_value = image[rows, np.clip(left, 0, width - 1)]
    right_value = image[rows, np.clip(right, 0, width - 1)]
    return left, right, left_value, right_value


def _mismatch(ends, width: int, max_span: float) -> np.ndarray:
    """A két vég színeltérése soronként; ahol nincs mindkét vég, 0, ahol túl messze vannak, végtelen."""
    left, right, left_value, right_value = ends
    both = (left >= 0) & (right < width)
    difference = np.abs(left_value - right_value).sum(axis=2)
    return np.where(both & (right - left <= max_span), difference, np.where(both, np.inf, 0.0))


def _across(ends, width: int) -> np.ndarray:
    """Vízszintes lineáris interpoláció a bal és jobb vég között (egyoldali végnél annak színe)."""
    left, right, left_value, right_value = ends
    has_left, has_right = left >= 0, right < width
    left_value = np.where(has_left[..., None], left_value, right_value)
    right_value = np.where(has_right[..., None], right_value, left_value)
    span = np.maximum(right - left, 1)
    share = np.where(has_left & has_right, (np.arange(width) - left) / span, 0.5)[..., None]
    return left_value * (1 - share) + right_value * share


def follow_horizontal_edges(image: np.ndarray, mask: np.ndarray, smooth: np.ndarray,
                            frame_width: int) -> np.ndarray:
    """A smooth kitöltés (float32), de ahol a törölt sor alatt vízszintes él fut át, ott vízszintes interpoláció."""
    # A végpontokat a maszkon kicsit túlról vesszük: a betűk élsimított széle ne húzzon világos csíkot
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, END_MARGIN_PX * 2 + 1), np.uint8)) > 0
    soft = cv2.GaussianBlur(image, (0, 0), SMOOTH_SIGMA).astype(np.float32)
    max_span = MAX_SPAN_PX * frame_width / REFERENCE_WIDTH
    across = _ends(soft, mask)
    across_mismatch = _mismatch(across, mask.shape[1], max_span)
    down_mismatch = _mismatch(_ends(soft.transpose(1, 0, 2), mask.T), mask.shape[0], np.inf).T
    with np.errstate(invalid="ignore"):
        evidence = np.nan_to_num((down_mismatch - across_mismatch - EDGE_MARGIN) / EDGE_RANGE, nan=0.0,
                                 posinf=1.0, neginf=0.0)
    evidence = cv2.GaussianBlur(np.clip(evidence, 0, 1).astype(np.float32), (0, 0), EVIDENCE_SIGMA)[..., None]
    return smooth.astype(np.float32) * (1 - evidence) + _across(across, mask.shape[1]) * evidence
