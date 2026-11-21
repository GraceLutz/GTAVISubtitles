"""Mesterséges feliratsáv a tesztekhez: fehér, fekete körvonalú, középre igazított sorok sötét jeleneten."""

import cv2
import numpy as np

from gta_felirat.config import Region
from gta_felirat.ocr import region_to_box

FRAME = (1920, 1080)
AREA = region_to_box(Region(), *FRAME)
CENTER_X = FRAME[0] // 2 - AREA.left  # a kép közepe a sáv koordinátáiban
DIALOGUE_BASELINE = 140  # a legalsó párbeszédsor alapvonala a sávban
UPPER_BASELINE = 97  # a fölötte álló párbeszédsoré
OBJECTIVE_BASELINE = 200  # a küldetésutasítás sora a párbeszédsorok alatt
FONT, FONT_SCALE, OUTLINE_PX = cv2.FONT_HERSHEY_SIMPLEX, 0.76, 2  # ekkora cv2-betű hasonlít a játék feliratára


def band_with(*lines, seed=1):
    """Sötét, enyhén zajos jelenet a megadott (szöveg, alapvonal) sorokkal."""
    noise = np.random.default_rng(seed).integers(-12, 13, (AREA.height, AREA.width, 3))
    band = np.clip(40 + noise, 0, 255).astype(np.uint8)
    for text, baseline in lines:
        (width, _), _ = cv2.getTextSize(text, FONT, FONT_SCALE, 1)
        letters = np.zeros(band.shape[:2], np.uint8)
        cv2.putText(letters, text, (round(CENTER_X - width / 2), baseline), FONT, FONT_SCALE, 255, 1, cv2.LINE_AA)
        side = 2 * OUTLINE_PX + 1
        band[cv2.dilate(letters, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))) > 0] = 0
        alpha = letters[:, :, None] / 255.0
        band[:] = (band * (1.0 - alpha) + 255 * alpha + 0.5).astype(np.uint8)
    return band
