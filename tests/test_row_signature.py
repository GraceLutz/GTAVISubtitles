from dataclasses import replace

import cv2
import numpy as np
import pytest

from gta_felirat.row_detector import detect
from gta_felirat.row_signature import (DIFFERENT, LOCAL_CHANGE, LOCAL_WINDOW, MAX_SHIFT_Y, MAX_WIDTH_DIFFERENCE, SAME,
                                       SAME_OVERLAP, VERIFY, compare, row_signature, similarity)

FRAME_WIDTH = 1920
BAND_LEFT = 240
BAND_WIDTH, BAND_HEIGHT = 1440, 216
CENTER = FRAME_WIDTH // 2 - BAND_LEFT
LOWER_SLOT, UPPER_SLOT = 140, 97  # két szomszédos feliratsor alapvonala a sávban
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.76  # ekkora cv2-betű hasonlít a játék feliratára 1920 széles képen
OUTLINE_PX = 2
DARK, MID_GREY, WHITE_SHIRT = 40, 110, 235
STROKE_PITCH = 6  # a mesterséges „betűk”: 2 pixel vastag függőleges vonások ennyi pixelenként
COMB_SIZE = (17, 120)  # egy mesterséges sor maszkjának magassága és szélessége
MASK_TOP = 20  # a mesterséges sor teteje a próbamaszkon


def subtitle_band(text, baseline=LOWER_SLOT, background=DARK):
    """Egyszínű, enyhén zajos jelenet közepén fehér, fekete körvonalas feliratsor (mint a GTA felirata)."""
    jitter = np.random.default_rng(background).integers(-12, 13, (BAND_HEIGHT, BAND_WIDTH, 3))
    band = np.clip(background + jitter, 0, 255).astype(np.uint8)
    (width, _), _ = cv2.getTextSize(text, FONT, FONT_SCALE, 1)
    letters = np.zeros((BAND_HEIGHT, BAND_WIDTH), np.uint8)
    cv2.putText(letters, text, (round(CENTER - width / 2), baseline), FONT, FONT_SCALE, 255, 1, cv2.LINE_AA)
    side = 2 * OUTLINE_PX + 1
    band[cv2.dilate(letters, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))) > 0] = 0
    alpha = letters[:, :, None] / 255.0
    return (band * (1.0 - alpha) + 255 * alpha + 0.5).astype(np.uint8)


def signature_of_text(text, **scene):
    detection = detect(subtitle_band(text, **scene), BAND_LEFT, FRAME_WIDTH)
    row = detection.rows[0]
    return row_signature(detection.core, row.ink_left, row.ink_top, row.ink_right, row.ink_bottom, CENTER)


def comb(width=COMB_SIZE[1]):
    """Mesterséges sor: függőleges vonások, középre igazítva (a szélessége a vonásköz többszöröse legyen)."""
    mask = np.zeros((COMB_SIZE[0], width), np.uint8)
    mask[:, 0::STROKE_PITCH] = 255
    mask[:, 1::STROKE_PITCH] = 255
    return mask


def stripes():
    """Mesterséges sor vízszintes csíkokkal: a függőleges elcsúszásra érzékeny."""
    mask = np.zeros(COMB_SIZE, np.uint8)
    mask[0::8] = 255
    mask[1::8] = 255
    return mask


def signature_of_mask(mask):
    """A maszk középre igazított sorként egy üres sávmaszkon."""
    core = np.zeros((BAND_HEIGHT, BAND_WIDTH), np.uint8)
    left = CENTER - mask.shape[1] // 2
    core[MASK_TOP:MASK_TOP + mask.shape[0], left:left + mask.shape[1]] = mask
    return row_signature(core, left, MASK_TOP, left + mask.shape[1], MASK_TOP + mask.shape[0], CENTER)


def test_signature_is_anchored_at_the_frame_centre_and_counts_the_ink():
    mask = comb()

    signature = signature_of_mask(mask)

    assert (signature.x0, signature.y0) == (-(mask.shape[1] // 2), -(mask.shape[0] // 2))
    assert (signature.width, signature.weight) == (mask.shape[1], cv2.countNonZero(mask))


def test_signature_keeps_its_own_read_only_copy_of_the_mask():
    core = np.full((BAND_HEIGHT, BAND_WIDTH), 255, np.uint8)
    signature = row_signature(core, 0, MASK_TOP, BAND_WIDTH, MASK_TOP + 1, CENTER)  # teljes szélességű, egysoros

    core[:] = 0

    assert signature.mask.min() == 255
    assert not signature.mask.flags.writeable


def test_same_text_on_another_background_is_the_same():
    first = signature_of_text("Lucia: I said nothing, okay?", background=DARK)
    second = signature_of_text("Lucia: I said nothing, okay?", background=MID_GREY)

    assert compare(first, second) == SAME


def test_same_text_in_another_row_slot_is_the_same():
    first = signature_of_text("Lucia: I said nothing, okay?", baseline=LOWER_SLOT)
    second = signature_of_text("Lucia: I said nothing, okay?", baseline=UPPER_SLOT)

    assert compare(first, second) == SAME


def test_same_text_on_a_white_shirt_is_not_taken_for_another_text():
    first = signature_of_text("Lucia: I said nothing, okay?", background=DARK)
    second = signature_of_text("Lucia: I said nothing, okay?", background=WHITE_SHIRT)

    assert similarity(first, second) >= SAME_OVERLAP
    assert compare(first, second) != DIFFERENT


def test_different_text_is_different():
    first = signature_of_text("Lucia: I said nothing, okay?")
    second = signature_of_text("Jason: We need to go, right?")

    assert similarity(first, second) < SAME_OVERLAP
    assert compare(first, second) == DIFFERENT


def test_one_replaced_character_asks_for_a_new_read():
    first = signature_of_text("Jason: Go.")
    second = signature_of_text("Jason: Go!")

    assert compare(first, second) == VERIFY


@pytest.mark.parametrize("extra_width, scale, is_different", [(MAX_WIDTH_DIFFERENCE + 8, 1.0, True),
                                                             (MAX_WIDTH_DIFFERENCE - 4, 1.0, False),
                                                             (MAX_WIDTH_DIFFERENCE + 8, 2.0, False)])
def test_row_wider_than_the_tolerance_is_different_even_if_it_contains_the_other(extra_width, scale, is_different):
    narrow = signature_of_mask(comb())
    wide = signature_of_mask(comb(COMB_SIZE[1] + extra_width))  # a többlet a két oldalon egy-egy vonásköz többszöröse

    verdict = compare(narrow, wide, scale)

    assert similarity(narrow, wide) == 1.0
    assert (verdict == DIFFERENT) is is_different


@pytest.mark.parametrize("shift, expected", [(MAX_SHIFT_Y, SAME), (MAX_SHIFT_Y + 2, DIFFERENT)])
def test_vertical_jitter_is_tolerated_only_up_to_the_tried_shifts(shift, expected):
    signature = signature_of_mask(stripes())

    assert compare(signature, replace(signature, y0=signature.y0 + shift)) == expected


def test_horizontally_shifted_row_is_another_text():
    signature = signature_of_mask(comb())

    assert compare(signature, replace(signature, x0=signature.x0 + STROKE_PITCH // 2)) == DIFFERENT


@pytest.mark.parametrize("changed_pixels, expected", [(LOCAL_CHANGE - 6, SAME), (LOCAL_CHANGE, VERIFY)])
def test_change_as_big_as_a_character_asks_for_a_new_read(changed_pixels, expected):
    changed = comb()
    changed[:changed_pixels // 3, 2:5] = 255  # három oszlopnyi folt két vonás között

    assert compare(signature_of_mask(comb()), signature_of_mask(changed)) == expected


def test_empty_signature_matches_nothing():
    empty = signature_of_mask(np.zeros(COMB_SIZE, np.uint8))

    assert similarity(empty, empty) == 0.0
    assert compare(empty, empty) == DIFFERENT


def test_thresholds_are_the_values_measured_on_the_recordings():
    # A felvételeken mért értékek, a tesztek viszont nem látják a felvételeket: módosítás csak újraméréssel
    assert (SAME_OVERLAP, MAX_WIDTH_DIFFERENCE, LOCAL_WINDOW, LOCAL_CHANGE, MAX_SHIFT_Y) == (0.65, 16, 14, 18, 2)
