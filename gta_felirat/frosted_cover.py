"""Az angol felirat igényes eltakarása: elmosott, kissé sötétített „matt üveg” a játékképből.

A takarás közepén a játékkép elmosva látszik (a fehér betűket az elmosásból kihagyjuk, így
nem marad utánuk világos folt), a széle felé pedig fokozatosan visszaáll az éles képre,
így nincs kemény kerete. A forráskép a takarás nélküli játékkép (Remote Play módban az
ablak saját tartalma, capture card módban maga a képkocka).

A takarás erőssége a feliratpanel átlátszóságát követi: átlátszó panelnél (strength 0) csak
az angol betűk helyére kerül a háttér színe, sötétítés nélkül, a betűk között az élő játékkép
látszik. Így a fordítás pontosan az angol szöveg helyén állhat, sötét doboz nélkül.
"""

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from .compositing import composite
from .ocr import Box

TEXT_LEVEL = 200  # mindhárom csatorna legalább ennyi: a betű hófehér magja (a napos út nem az)
TEXT_GROW_PX = 5  # a mag körül ennyi tartozik még a betűhöz: élsimított szél és fekete körvonal
# A betűt fekete körvonal veszi körül, ezért a fehér magja közelében mindig van sötét pixel.
# A fehér póló vagy a napos talaj belsejében nincs: így azok nem számítanak betűnek.
OUTLINE_REACH_PX = 4
OUTLINE_LEVEL = 150  # ennél sötétebb pixel a közelben: körvonal vagy sötét háttér
# A tévéműsor felirata halványabb (~183): a küszöb a kép legvilágosabb pixeleihez igazodik
DIM_TEXT_LEVEL = 150
TEXT_TOP_PERCENTILE = 99.5  # a betűk legalább a takart terület ennyi „tetején” vannak
TEXT_LEVEL_MARGIN = 25  # ennyivel a legvilágosabb pixelek alatt még a betű magja
MAX_GREY_SPREAD = 40  # a halvány betű szürkésfehér: a színcsatornái közel egyformák
MIN_BLUR_SIGMA = 6.0
BLUR_SIGMA_RATIO = 0.35  # a takarás magasságához képest
BLUR_DOWNSCALE = 4  # ennyiszer kisebb képen mosunk el
DARKEN = 0.72  # teljes erősségnél: a sötétebb háttéren jobban olvasható a fordítás
ERASE_SOFTEN_SIGMA = 1.5  # az eltüntetett betűk széle lágyan olvad a játékképbe
MIN_TEXT_FREE_WEIGHT = 0.05  # ennél kevesebb betűmentes környezetnél sima elmosás
SIDE_FEATHER_RATIO = 2  # oldalt szélesebb az átmenet: a felirat hosszú és lapos


@dataclass(frozen=True)
class Backdrop:
    image: np.ndarray  # a játékkép takarás nélkül (BGR)
    area: Box  # hol van ez a kép a rajzolt képen


def cover_area(anchor: Box, margin: int, limits: Box, side_margin: int | None = None) -> Box:
    """Az angol szöveg doboza margóval (oldalt side_margin-nal) kibővítve, a forráskép határain belül."""
    side = margin if side_margin is None else side_margin
    return Box(left=max(anchor.left - side, limits.left), top=max(anchor.top - margin, limits.top),
               right=min(anchor.right + side, limits.right), bottom=min(anchor.bottom + margin, limits.bottom))


def _disc(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


_TEXT_GROW_KERNEL = _disc(TEXT_GROW_PX)
_OUTLINE_REACH_KERNEL = _disc(OUTLINE_REACH_PX)


def _text_mask(source: np.ndarray) -> np.ndarray:
    """1 a felirat betűin és a körvonalukon, 0 máshol (uint8)."""
    darkest = cv2.min(cv2.min(source[:, :, 0], source[:, :, 1]), source[:, :, 2])  # ~20× gyorsabb a numpy-nál
    brightest = cv2.max(cv2.max(source[:, :, 0], source[:, :, 1]), source[:, :, 2])
    has_dark_nearby = cv2.erode(brightest, _OUTLINE_REACH_KERNEL) <= OUTLINE_LEVEL
    level = _text_level(darkest)
    letters = (darkest >= level) & has_dark_nearby
    if level < TEXT_LEVEL:
        letters &= cv2.subtract(brightest, darkest) <= MAX_GREY_SPREAD
    return cv2.dilate(letters.astype(np.uint8), _TEXT_GROW_KERNEL)


def _text_level(darkest: np.ndarray) -> float:
    """A betű magjának küszöbe: hófehér feliratnál TEXT_LEVEL, halványnál a legvilágosabb pixelek alatt."""
    top = float(np.percentile(darkest, TEXT_TOP_PERCENTILE))
    return min(TEXT_LEVEL, max(DIM_TEXT_LEVEL, top - TEXT_LEVEL_MARGIN))


def _blur_without_text(source: np.ndarray, text: np.ndarray, sigma: float, darken: float) -> np.ndarray:
    # Normált elmosás: a betűpixelek nem számítanak bele, így a háttérszín folytatódik alattuk
    keep = 1.0 - text.astype(np.float32)
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
    darkened = (blurred * darken).astype(np.float32)  # kicsiben sötétítünk: kevesebb pixel
    return cv2.resize(darkened, (width, height), interpolation=cv2.INTER_LINEAR)


def _smoothstep_ramp(length: int, feather: int) -> np.ndarray:
    positions = np.arange(length, dtype=np.float32)
    linear = np.clip(np.minimum(positions, length - 1 - positions) / max(feather, 1), 0.0, 1.0)
    return linear * linear * (3.0 - 2.0 * linear)


@lru_cache(maxsize=16)
def _edge_weight(height: int, width: int, feather: int) -> np.ndarray:
    """0 a széleken, 1 belül, lágy (smoothstep) átmenettel; oldalt szélesebb (csak olvasható, gyorsítótárazott).

    A két irány szorzata lekerekített sarkot ad, nem látszik átlós „él” a sarkokban.
    """
    weight = _smoothstep_ramp(height, feather)[:, None] * _smoothstep_ramp(width, feather * SIDE_FEATHER_RATIO)[None, :]
    weight.setflags(write=False)
    return weight


def _letter_coverage(text: np.ndarray) -> np.ndarray:
    """1 a betűkön, lágy széllel 0-ra fut ki: csak maguk a betűk tűnnek el."""
    return cv2.GaussianBlur(text.astype(np.float32), (0, 0), ERASE_SOFTEN_SIGMA)


def frosted_layer(source: np.ndarray, feather: int, strength: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """A takarás színe (elmosott; float BGR) és fedése (0 a szélén, 1 belül; H×W×1).

    strength: 1 = teljes matt üveg, sötétítve; 0 = csak a betűk helyén, sötétítés nélkül.
    Átlátszó rétegen a lágy szél az alatta futó élő játékképbe olvad, így mozgás közben sem
    látszik, hogy a forráskép egy pillanattal korábbi.
    """
    height, width = source.shape[:2]
    sigma = max(MIN_BLUR_SIGMA, height * BLUR_SIGMA_RATIO)
    text = _text_mask(source)
    frosted = _blur_without_text(source, text, sigma, 1.0 - (1.0 - DARKEN) * strength)
    weight = _edge_weight(height, width, feather)
    if strength < 1.0:
        letters = _letter_coverage(text)
        weight = weight * (letters + (1.0 - letters) * strength)
    return frosted, weight[:, :, None]


def frosted_patch(source: np.ndarray, feather: int, strength: float = 1.0) -> np.ndarray:
    """A forrásképpel azonos méretű takarás: belül elmosott, a szélén az éles kép."""
    frosted, weight = frosted_layer(source, feather, strength)
    return composite(source, frosted, weight)
