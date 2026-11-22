"""A betűk alatti háttér megjegyzése: álló képen a szöveg helyére a valódi háttér kerül, nem elkent folt."""

import cv2
import numpy as np

from gta_felirat.background_memory import BackgroundMemory
from gta_felirat.compositing import composite
from gta_felirat.frosted_cover import frosted_layer, text_mask
from gta_felirat.ocr import Box

HEIGHT, WIDTH = 90, 700
AREA = Box(100, 900, 100 + WIDTH, 900 + HEIGHT)
FEATHER = 6
STEP = 0.2  # a képek között eltelt idő (a memória ennyinél sűrűbben nem tanul)


def brick_wall(seed: int = 1) -> np.ndarray:
    """Mintás, világos háttér (mint a téglafal): ezen látszott az elkent folt."""
    rng = np.random.default_rng(seed)
    wall = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    for top in range(0, HEIGHT, 15):
        shift = 20 if (top // 15) % 2 else 0
        for left in range(-shift, WIDTH, 40):
            colour = (int(rng.integers(60, 110)), int(rng.integers(100, 150)), int(rng.integers(150, 200)))
            cv2.rectangle(wall, (left + 1, top + 1), (left + 38, top + 13), colour, -1)
    return wall


def with_subtitle(background: np.ndarray) -> np.ndarray:
    image = background.copy()
    org = (60, 58)
    cv2.putText(image, "No, fuck you, two juniors!", org, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (25, 25, 25), 5, cv2.LINE_AA)
    cv2.putText(image, "No, fuck you, two juniors!", org, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2,
                cv2.LINE_AA)
    return image


def erased(backdrop) -> np.ndarray:
    source, clean, weight = backdrop.crop(AREA)
    frosted, alpha = frosted_layer(source, FEATHER, strength=0.0, clean=clean, clean_weight=weight)
    return np.clip(composite(source, frosted, alpha), 0, 255).astype(np.uint8)


def letter_error(result: np.ndarray, background: np.ndarray, subtitled: np.ndarray) -> float:
    letters = text_mask(subtitled) > 0
    return float(np.abs(result.astype(int) - background.astype(int)).mean(axis=2)[letters].mean())


def test_still_scene_gets_the_real_background_under_the_letters():
    wall = brick_wall()
    memory = BackgroundMemory()
    memory.backdrop(wall, AREA, 0 * STEP)  # a felirat előtti kép
    subtitled = with_subtitle(wall)

    with_memory = letter_error(erased(memory.backdrop(subtitled, AREA, 1 * STEP)), wall, subtitled)
    without_memory = letter_error(erased(BackgroundMemory().backdrop(subtitled, AREA, 2 * STEP)), wall, subtitled)

    assert with_memory < 8
    assert with_memory < without_memory / 3


def test_memory_stays_while_the_subtitle_stays_on_screen():
    wall = brick_wall()
    memory = BackgroundMemory()
    memory.backdrop(wall, AREA, 3 * STEP)
    subtitled = with_subtitle(wall)
    for _ in range(30):  # a felirat sokáig kint van: a betűk alatt nem írjuk felül az emléket
        backdrop = memory.backdrop(subtitled, AREA, 4 * STEP)

    assert letter_error(erased(backdrop), wall, subtitled) < 8


def test_changed_scene_is_not_trusted():
    memory = BackgroundMemory()
    memory.backdrop(brick_wall(seed=1), AREA, 5 * STEP)
    other_scene = with_subtitle(brick_wall(seed=2)[:, ::-1].copy())

    backdrop = memory.backdrop(other_scene, AREA, 6 * STEP)

    letters = text_mask(other_scene) > 0
    assert backdrop.remembered.parts[1][letters].max() < 0.1  # vágás után a régi kép nem kerülhet a betűk helyére


def test_first_picture_has_nothing_to_remember():
    subtitled = with_subtitle(brick_wall())

    backdrop = BackgroundMemory().backdrop(subtitled, AREA, 7 * STEP)

    assert backdrop.remembered.parts[1].max() == 0.0


def test_other_place_or_size_starts_afresh():
    wall = brick_wall()
    memory = BackgroundMemory()
    memory.backdrop(wall, AREA, 8 * STEP)

    moved = Box(AREA.left + 10, AREA.top, AREA.right + 10, AREA.bottom)
    backdrop = memory.backdrop(with_subtitle(wall), moved, 9 * STEP)

    assert backdrop.remembered.parts[1].max() == 0.0


def test_earlier_backdrop_does_not_change_later():
    wall = brick_wall()
    memory = BackgroundMemory()
    first = memory.backdrop(wall, AREA, 10 * STEP)
    kept = first.remembered.parts[0].copy()

    memory.backdrop(brick_wall(seed=3), AREA, 11 * STEP)

    assert np.array_equal(first.remembered.parts[0], kept)


def test_backdrop_without_memory_crops_like_before():
    image = brick_wall()
    from gta_felirat.frosted_cover import Backdrop

    source, clean, weight = Backdrop(image, AREA).crop(Box(AREA.left + 5, AREA.top + 5, AREA.left + 50, AREA.top + 40))

    assert source.shape[:2] == (35, 45) and clean is None and weight is None


def test_crop_outside_the_backdrop_is_none():
    from gta_felirat.frosted_cover import Backdrop

    assert Backdrop(brick_wall(), AREA).crop(Box(AREA.left - 5, AREA.top, AREA.left + 50, AREA.top + 40)) is None
