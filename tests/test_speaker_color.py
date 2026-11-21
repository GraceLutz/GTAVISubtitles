import time

import cv2
import numpy as np
import pytest

from band_drawing import AREA, CENTER_X, DIALOGUE_BASELINE, FONT, FONT_SCALE, OUTLINE_PX
from gta_felirat.ocr import Box
from gta_felirat.speaker_color import SpeakerPalette, sample_speaker_colors

TEAL, ORANGE, RED = (126, 207, 214), (240, 170, 60), (220, 70, 70)  # RGB: a nevek színe a játékban
GAME_WHITE, DIMMED_WHITE = 235, 180  # a játék fehérje, és a sötétített jelenetben mért fehér
STROKE_PX = 2  # a játék betűvonala vastagabb a cv2 egy pixeles vonalánál
TOLERANCE = 12  # csatornánként ennyi eltérés megengedett
DARK_LEVEL = 40
BRIGHT_LEVEL = 225  # világos jelenet (ég, fehér fal): a fehér szövegnél alig sötétebb
MAX_SECONDS_PER_LINE = 0.05  # bő keret: a mérés 1920 széles képen ~10 ms alatt van


def dark_scene(seed=1):
    noise = np.random.default_rng(seed).integers(-12, 13, (AREA.height, AREA.width, 3))
    return np.clip(DARK_LEVEL + noise, 0, 255).astype(np.uint8)


def textured_scene():
    """Világos és sötét foltok (lombozat, burkolat): felnagyított véletlen rács."""
    coarse = np.random.default_rng(3).integers(30, 231, (AREA.height // 8 + 2, AREA.width // 8 + 2, 3))
    return cv2.resize(coarse.astype(np.uint8), (AREA.width, AREA.height), interpolation=cv2.INTER_CUBIC)


def width_of(text):
    (width, _), _ = cv2.getTextSize(text, FONT, FONT_SCALE, STROKE_PX)
    return width


def write_line(band, parts, white=GAME_WHITE, baseline=DIALOGUE_BASELINE):
    """Középre igazított sor (szöveg, RGB szín vagy None = fehér) darabokból, fekete körvonallal.

    A színeket `white / 235` arányban sötétíti (mint a sötétített jelenet); visszaadja a teljes szöveget,
    és a darabok tintájának dobozait a kép koordinátáiban.
    """
    text = "".join(part for part, _color in parts)
    letters = np.zeros(band.shape[:2], np.uint8)
    x = round(CENTER_X - width_of(text) / 2)
    cv2.putText(letters, text, (x, baseline), FONT, FONT_SCALE, 255, STROKE_PX, cv2.LINE_AA)
    side = 2 * (OUTLINE_PX + STROKE_PX // 2) + 1
    band[cv2.dilate(letters, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))) > 0] = 0
    colors = np.zeros(band.shape, np.float64)
    boxes = []
    for part, rgb in parts:
        level = np.array(rgb[::-1] if rgb else (GAME_WHITE,) * 3, float) * white / GAME_WHITE
        part_width = width_of(part)
        colors[:, x:x + part_width + 1] = level
        boxes.append(ink_box(letters, x, x + part_width))
        x += part_width
    alpha = letters[:, :, None] / 255.0
    band[:] = (band * (1.0 - alpha) + colors * alpha + 0.5).astype(np.uint8)
    return text, boxes


def ink_box(letters, left, right):
    ys, xs = np.nonzero(letters[:, left:right + 1])
    return Box(AREA.left + left + int(xs.min()), AREA.top + int(ys.min()),
               AREA.left + left + int(xs.max()) + 1, AREA.top + int(ys.max()) + 1)


def line_box(boxes):
    return Box(min(b.left for b in boxes), min(b.top for b in boxes),
               max(b.right for b in boxes), max(b.bottom for b in boxes))


def expected(rgb):
    return tuple(min(255, round(channel * 255 / GAME_WHITE)) for channel in rgb)


def assert_close(measured, rgb):
    assert np.abs(np.array(measured) - np.array(expected(rgb))).max() <= TOLERANCE, (measured, expected(rgb))


@pytest.mark.parametrize("rgb", [TEAL, ORANGE, RED], ids=["teal", "orange", "red"])
@pytest.mark.parametrize("scene", [dark_scene, textured_scene], ids=["dark", "textured"])
def test_measures_name_colour_on_dark_and_textured_scenes(rgb, scene):
    # Arrange
    band = scene()
    text, boxes = write_line(band, [("Lucia:", rgb), (" Stop talking, will you?", None)])

    # Act
    colors = sample_speaker_colors(band, AREA, text, line_box(boxes))

    # Assert
    assert [name for name, _color in colors] == ["Lucia"]
    assert_close(colors[0][1], rgb)


@pytest.mark.parametrize("rgb", [TEAL, ORANGE, RED], ids=["teal", "orange", "red"])
def test_dimmed_scene_is_corrected_by_the_measured_white(rgb):
    band = dark_scene()
    text, boxes = write_line(band, [("Jason:", rgb), (" We go now, alright?", None)], white=DIMMED_WHITE)

    colors = sample_speaker_colors(band, AREA, text, line_box(boxes))

    assert_close(dict(colors)["Jason"], rgb)


def test_name_left_of_the_line_box_is_still_found():
    band = dark_scene()
    text, (_name_box, body_box) = write_line(band, [("Staff Member:", ORANGE), (" Please stand back.", None)])

    colors = sample_speaker_colors(band, AREA, text, body_box)

    assert [name for name, _color in colors] == ["Staff Member"]
    assert_close(colors[0][1], ORANGE)


def test_bright_background_between_and_inside_letters_does_not_split_the_name():
    # Arrange: világos jelenet, a betűk közé és belsejébe szorult háttér fehér foltnak látszik
    band = np.full((AREA.height, AREA.width, 3), BRIGHT_LEVEL, np.uint8)
    text, boxes = write_line(band, [("Staff Member:", RED), (" Please stand back.", None)])

    # Act
    colors = sample_speaker_colors(band, AREA, text, line_box(boxes))

    # Assert
    assert [name for name, _color in colors] == ["Staff Member"]
    assert_close(colors[0][1], RED)


def test_two_speakers_in_one_line_get_their_own_colours():
    band = dark_scene()
    parts = [("Vagrant:", RED), (" Hey there. ", None), ("Lucia:", TEAL), (" Go away.", None)]
    text, boxes = write_line(band, parts)

    colors = sample_speaker_colors(band, AREA, text, line_box(boxes))

    assert [name for name, _color in colors] == ["Vagrant", "Lucia"]
    assert_close(colors[0][1], RED)
    assert_close(colors[1][1], TEAL)


def test_more_names_in_text_than_coloured_runs_gives_nothing():
    band = dark_scene()
    text, boxes = write_line(band, [("Lucia:", TEAL), (" Stop talking, will you?", None)])

    colors = sample_speaker_colors(band, AREA, "Jason: Hey. " + text, line_box(boxes))

    assert colors == ()


def test_white_speaker_name_gives_nothing():
    band = dark_scene()
    text, boxes = write_line(band, [("Lucia: Stop talking, will you?", None)])

    assert sample_speaker_colors(band, AREA, text, line_box(boxes)) == ()


def test_line_without_speaker_gives_nothing():
    band = dark_scene()
    text, boxes = write_line(band, [("Stop", ORANGE), (" talking, will you?", None)])

    assert sample_speaker_colors(band, AREA, text, line_box(boxes)) == ()


def test_language_note_is_not_a_speaker():
    band = dark_scene()
    text, boxes = write_line(band, [("In Spanish: ", None), ("Stop it!", None)])

    assert sample_speaker_colors(band, AREA, text, line_box(boxes)) == ()


def test_bgra_band_is_accepted():
    band = dark_scene()
    text, boxes = write_line(band, [("Lucia:", ORANGE), (" Stop talking, will you?", None)])
    bgra = cv2.cvtColor(band, cv2.COLOR_BGR2BGRA)

    colors = sample_speaker_colors(bgra, AREA, text, line_box(boxes))

    assert_close(dict(colors)["Lucia"], ORANGE)


def test_measurement_is_fast():
    band = dark_scene()
    text, boxes = write_line(band, [("Lucia:", TEAL), (" Stop talking, will you?", None)])
    box = line_box(boxes)
    sample_speaker_colors(band, AREA, text, box)

    started = time.perf_counter()
    sample_speaker_colors(band, AREA, text, box)

    assert time.perf_counter() - started < MAX_SECONDS_PER_LINE


def test_palette_keeps_the_median_so_one_bad_read_does_not_flip_the_colour():
    palette = SpeakerPalette()
    palette.learn((("Lucia", TEAL),))
    palette.learn((("Lucia", TEAL),))

    palette.learn((("Lucia", RED),))

    assert palette.for_text("Lucia: Stop talking.") == (("Lucia", TEAL),)


def test_palette_forgets_samples_older_than_its_history():
    palette = SpeakerPalette(history=3)
    palette.learn((("Lucia", TEAL),))

    for _ in range(2):
        palette.learn((("Lucia", ORANGE),))

    assert palette.for_text("Lucia: Hi.") == (("Lucia", ORANGE),)


def test_palette_matches_names_case_insensitively_and_keeps_the_spelling_of_the_text():
    palette = SpeakerPalette()
    palette.learn((("Staff Member ", ORANGE),))

    colors = palette.for_text("STAFF MEMBER: Please stand back.")

    assert colors == (("STAFF MEMBER", ORANGE),)


def test_palette_returns_sorted_entries_only_for_known_speakers():
    palette = SpeakerPalette()
    palette.learn((("Vagrant", RED), ("Lucia", TEAL)))

    colors = palette.for_text("Vagrant: Hey. Jason: Hm. Lucia: Go.")

    assert colors == (("Lucia", TEAL), ("Vagrant", RED))
    assert hash(colors) == hash((("Lucia", TEAL), ("Vagrant", RED)))


def test_palette_ignores_language_notes_and_lines_without_speaker():
    palette = SpeakerPalette()
    palette.learn((("Lucia", TEAL),))

    assert palette.for_text("In Spanish: Lucia is here.") == ()
    assert palette.for_text("Stop talking.") == ()
