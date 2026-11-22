"""A felirat alatti valódi háttér megjegyzése, hogy a betűk eltüntetése után ne maradjon elkent folt.

A betűk helyét eddig a környék elmosásával töltöttük ki: sötét, egyszínű háttéren ez nem látszik, de mintás,
világos háttéren (téglafal, ruha) a szöveg helyén elkent sáv („satír”) maradt. Párbeszéd közben a kamera
többnyire áll, így a felirat megjelenése előtti képen pontosan látszott, mi van a betűk alatt. Ezt jegyezzük
meg képpontonként, és a betűk helyére ezt tesszük, ha a környék azóta nem változott (ugyanaz a jelenet).
Ha a kép mozog vagy vágás volt, a környék eltér az emlékezettől: ott marad az elmosásos kitöltés.

A munka fele felbontáson megy, és másodpercenként legfeljebb LEARN_PER_SECOND-szor tanulunk; a hihetőséget
csak akkor számoljuk ki, amikor egy takarás tényleg kirajzolódik (RememberedBackground.parts).
"""

from __future__ import annotations

from functools import cached_property

import cv2
import numpy as np

from .frosted_cover import Backdrop, text_mask
from .ocr import Box

SCALE = 2  # ennyiszer kisebb képen keressük a betűket és vetjük össze a jelenetet
LEARN_PER_SECOND = 10.0
# A betű (és körvonala) körül ennyivel (kicsinyített képpontban) szélesebb sávot nem jegyzünk meg
GUARD_PX = 2
SAME_PIXEL_TOLERANCE = 12  # ennyi csatornaeltérésig ugyanaz a képpont (JPEG-zaj, apró fényváltozás)
# A jelenet akkor ugyanaz, ha szűk és tág környéken is (kicsinyített képpontban) szinte minden egyezik
NEAR_PX, WIDE_PX = 21, 41
MIN_SAME_FRACTION = 0.96
MIN_COMPARED_FRACTION = 0.05  # a környéknek legalább ennyi része legyen összevethető (nem betű, van emlék)
TRUST_SOFTEN_SIGMA = 2.0  # a hihető és nem hihető rész lágyan vált: nincs éles határ a kitöltésben

_GUARD_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * GUARD_PX + 1, 2 * GUARD_PX + 1))


def _small(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    return cv2.resize(image, (max(1, width // SCALE), max(1, height // SCALE)), interpolation=cv2.INTER_AREA)


def _busy(small: np.ndarray) -> np.ndarray:
    """uint8 1: betű, körvonala vagy közvetlen környéke (kicsinyített képen)."""
    return cv2.dilate(text_mask(small, scale=SCALE), _GUARD_KERNEL)


class RememberedBackground:
    """Egy képhez tartozó emlék: a háttér a betűk alatt, és hol hihető. A hihetőséget csak kérésre számolja."""

    def __init__(self, image: np.ndarray, memory: np.ndarray, known: np.ndarray, busy: np.ndarray | None) -> None:
        self._image, self._memory, self._known, self._busy = image, memory, known, busy

    @cached_property
    def parts(self) -> tuple[np.ndarray, np.ndarray]:
        """(emlékezett háttér teljes felbontásban, hihetősége 0–1 teljes felbontásban)."""
        small_image = _small(self._image)
        small_memory = _small(self._memory)
        busy = self._busy if self._busy is not None else _busy(small_image)
        known = self._known[:busy.shape[0], :busy.shape[1]]
        comparable = (known & (busy == 0)).astype(np.float32)
        difference = cv2.cvtColor(cv2.absdiff(small_image, small_memory), cv2.COLOR_BGR2GRAY)
        same = ((difference <= SAME_PIXEL_TOLERANCE) & (comparable > 0)).astype(np.float32)
        trusted = known.copy()
        for size in (NEAR_PX, WIDE_PX):
            compared = cv2.boxFilter(comparable, -1, (size, size))
            agreeing = cv2.boxFilter(same, -1, (size, size))
            trusted &= (compared >= MIN_COMPARED_FRACTION) & (agreeing >= MIN_SAME_FRACTION * compared)
        soft = cv2.GaussianBlur(trusted.astype(np.float32), (0, 0), TRUST_SOFTEN_SIGMA)
        height, width = self._image.shape[:2]
        return self._memory, cv2.resize(soft, (width, height), interpolation=cv2.INTER_LINEAR)


class BackgroundMemory:
    """Egy sáv képeiből (mindig ugyanaz a hely és méret) megjegyzi a betűk nélküli hátteret.

    Egyetlen szálból hívandó, a képek időrendjében."""

    def __init__(self) -> None:
        self._area: Box | None = None
        self._memory: np.ndarray | None = None  # uint8 BGR, teljes felbontás: a betű nélkül látott képpontok
        self._known: np.ndarray | None = None  # bool, kicsinyített: van-e emlék az adott helyről
        self._learned_at = float("-inf")

    def backdrop(self, image: np.ndarray, area: Box, captured_at: float) -> Backdrop:
        """A takarás háttere ebből a képből, a betűk alatti emlékezett háttérrel (ahol hihető)."""
        if self._area != area or self._memory is None or self._memory.shape != image.shape:
            self._area, self._learned_at = area, float("-inf")
            self._memory = image.copy()
            self._known = np.zeros(_small(image).shape[:2], dtype=bool)
        busy = None
        remembered = RememberedBackground(image, self._memory.copy(), self._known.copy(), None)
        if captured_at - self._learned_at >= 1.0 / LEARN_PER_SECOND:
            busy = self._learn(image)
            remembered = RememberedBackground(image, remembered._memory, remembered._known, busy)
            self._learned_at = captured_at
        return Backdrop(image, area, remembered=remembered)

    def _learn(self, image: np.ndarray) -> np.ndarray:
        busy = _busy(_small(image))
        free = (busy == 0).astype(np.uint8)
        height, width = image.shape[:2]
        free_full = cv2.resize(free, (width, height), interpolation=cv2.INTER_NEAREST)
        cv2.copyTo(image, free_full, self._memory)
        self._known[:free.shape[0], :free.shape[1]] |= free > 0
        return busy
