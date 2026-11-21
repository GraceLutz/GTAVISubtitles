from dataclasses import astuple

import cv2
import numpy as np
import pytest

from gta_felirat.row_detector import DEFAULT_PARAMS, NOTHING, detect, detect_rows, to_frame

FRAME_WIDTH = 1920
BAND_LEFT = 240  # a sáv bal széle a képen: így a sáv és a kép közepe 2/3-ra kicsinyítve is egész pixelre esik
BAND_WIDTH, BAND_HEIGHT = 1440, 216
BAND_TOP = 842  # az alapértelmezett sáv teteje 1080 magas képen
CENTER = FRAME_WIDTH // 2 - BAND_LEFT
ROW_BASELINES = (54, 97, 140)  # a három párbeszédsor alapvonala a sávban; a legalsó sor a 140-es
OBJECTIVE_BASELINE = 200  # a küldetésutasítás külön sorban áll a párbeszédsorok alatt
DIALOGUE_BOTTOM = 162.0  # a képmagasság 93%-a a sáv koordinátáiban (ocr.frame_center)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.76  # ekkora cv2-betű hasonlít a játék feliratára 1920 széles képen (nagybetű ~16 px)
OUTLINE_PX = 2
DARK, WHITE_SHIRT, DARKENED = 40, 235, 25  # háttérszintek: sötét jelenet, fehér póló, elsötétített tévékép
MID_GREY = 160  # se nem fehér, se nem sötét: mellette a betűnek nincs körvonalnak számító szomszédja
DIM_TEXT = 183  # a tévéműsor halvány, szürkésfehér felirata
DIALOGUE = "Jason: Stop talking, will you?"
ROW_TEXTS = ("Ow! Godammit.", "In Spanish: Ay stop it!", "She a bloodsucker, I tell you.")


def flat(level, noise=12):
    """Egyszínű jelenet enyhe zajjal."""
    jitter = np.random.default_rng(1).integers(-noise, noise + 1, (BAND_HEIGHT, BAND_WIDTH, 3))
    return np.clip(level + jitter, 0, 255).astype(np.uint8)


def textured():
    """Világos és sötét foltok (lombozat, burkolat): felnagyított véletlen rács."""
    coarse = np.random.default_rng(3).integers(30, 231, (BAND_HEIGHT // 8 + 2, BAND_WIDTH // 8 + 2, 3))
    return cv2.resize(coarse.astype(np.uint8), (BAND_WIDTH, BAND_HEIGHT), interpolation=cv2.INTER_CUBIC)


def checker():
    """Erős kontrasztú sakktábla: csupa vékony világos alakzat sötét szomszéddal, mégsem betű."""
    ys, xs = np.mgrid[0:BAND_HEIGHT, 0:BAND_WIDTH]
    grey = np.where((xs // 10 + ys // 10) % 2 == 0, 240, 90).astype(np.uint8)
    return cv2.merge([grey, grey, grey])


def glyphs(text, baseline, center_x=CENTER, scale=FONT_SCALE):
    """A szöveg betűi fehérrel, fekete alapon, vízszintesen a center_x köré igazítva (egy csatorna)."""
    (width, _), _ = cv2.getTextSize(text, FONT, scale, 1)
    mask = np.zeros((BAND_HEIGHT, BAND_WIDTH), np.uint8)
    cv2.putText(mask, text, (round(center_x - width / 2), baseline), FONT, scale, 255, 1, cv2.LINE_AA)
    return mask


def write(band, text, baseline, level=255, outline=OUTLINE_PX, **placement):
    """A sávra írja a szöveget `level` színnel és fekete körvonallal, mint a GTA felirata; a sávot adja vissza."""
    letters = glyphs(text, baseline, **placement)
    if outline:
        side = 2 * outline + 1
        band[cv2.dilate(letters, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))) > 0] = 0
    alpha = letters[:, :, None] / 255.0
    band[:] = (band * (1.0 - alpha) + level * alpha + 0.5).astype(np.uint8)
    return band


def dialogue_band():
    """Sötét jelenet, a legalsó párbeszédsor helyén egy feliratsorral."""
    return write(flat(DARK), DIALOGUE, ROW_BASELINES[-1])


def tv_scene_with_objective():
    """Halvány tévéfelirat a párbeszéd helyén, alatta hófehér küldetésutasítás."""
    band = write(flat(DARKENED), "In Spanish: Show some respect!", ROW_BASELINES[-1], level=DIM_TEXT)
    return write(band, "Follow Jay", OBJECTIVE_BASELINE)


def rows_in(band, **options):
    return detect_rows(band, BAND_LEFT, FRAME_WIDTH, **options)


def boxes(rows):
    return [(row.left, row.top, row.right, row.bottom) for row in rows]


@pytest.mark.parametrize("background", [lambda: flat(DARK), lambda: flat(WHITE_SHIRT), textured, checker,
                                        lambda: np.zeros((0, BAND_WIDTH, 3), np.uint8)],
                         ids=["dark", "white-shirt", "textured", "checker", "empty-band"])
def test_band_without_text_has_no_rows(background):
    assert rows_in(background()) == ()


def test_detection_without_rows_carries_no_mask():
    assert detect(flat(DARK), BAND_LEFT, FRAME_WIDTH) is NOTHING
    assert NOTHING.core is None


def test_white_outlined_row_on_dark_background_is_sure_at_first_sight():
    rows = rows_in(dialogue_band())

    assert len(rows) == 1
    assert rows[0].sure and not rows[0].dim


def test_row_reports_the_tight_extent_of_the_letters_and_a_padded_box():
    letters = glyphs(DIALOGUE, ROW_BASELINES[-1]) >= DEFAULT_PARAMS.white_level
    lines, columns = np.flatnonzero(letters.any(axis=1)), np.flatnonzero(letters.any(axis=0))

    row = rows_in(dialogue_band())[0]

    pad = DEFAULT_PARAMS
    assert (row.ink_left, row.ink_right) == (columns[0], columns[-1] + 1)
    assert (row.ink_top, row.ink_bottom) == (lines[0], lines[-1] + 1)
    assert (row.left, row.top) == (row.ink_left - pad.pad_left, row.ink_top - pad.pad_top)
    assert (row.right, row.bottom) == (row.ink_right + pad.pad_right, row.ink_bottom + pad.pad_bottom)
    assert abs(row.center_offset) <= DEFAULT_PARAMS.sure_offset


@pytest.mark.parametrize("background", [lambda: flat(WHITE_SHIRT), textured], ids=["white-shirt", "textured"])
def test_row_on_a_busy_background_is_found(background):
    rows = rows_in(write(background(), DIALOGUE, ROW_BASELINES[-1]))

    assert len(rows) == 1
    assert rows[0].top < ROW_BASELINES[-1] < rows[0].bottom


def test_dim_grey_tv_row_on_a_darkened_picture_is_found_as_dim():
    rows = rows_in(write(flat(DARKENED), "In Spanish: Show some respect!", ROW_BASELINES[-1], level=DIM_TEXT))

    assert len(rows) == 1
    assert rows[0].dim


@pytest.mark.parametrize("count", [2, 3])
def test_stacked_rows_are_returned_separately_from_top_to_bottom(count):
    band = flat(DARK)
    for text, baseline in zip(ROW_TEXTS[-count:], ROW_BASELINES[-count:]):
        write(band, text, baseline)

    rows = rows_in(band)

    assert len(rows) == count
    assert all(row.top < baseline < row.bottom for row, baseline in zip(rows, ROW_BASELINES[-count:]))
    assert all(upper.ink_bottom <= lower.ink_top for upper, lower in zip(rows, rows[1:]))


def test_rows_glued_together_by_a_bright_post_are_still_two_rows():
    band = write(dialogue_band(), ROW_TEXTS[1], ROW_BASELINES[1])
    cv2.line(band, (CENTER + 40, 70), (CENTER + 40, 150), (0, 0, 0), 6)
    cv2.line(band, (CENTER + 40, 70), (CENTER + 40, 150), (255, 255, 255), 2)

    rows = rows_in(band)

    assert [row.top < baseline < row.bottom for row, baseline in zip(rows, ROW_BASELINES[1:])] == [True, True]


@pytest.mark.parametrize("text, shift", [("Moses Bucked You $825", -420), ("Unbelievable!", 100)],
                         ids=["hud-notification-on-the-left", "word-beside-the-centre"])
def test_text_away_from_the_frame_centre_is_not_a_row(text, shift):
    assert rows_in(write(flat(DARK), text, ROW_BASELINES[-1], center_x=CENTER + shift)) == ()


def test_hud_text_on_the_lines_of_a_dialogue_row_does_not_change_the_row():
    hud = "Moses Bucked You $825 and it keeps going on"

    rows = rows_in(write(dialogue_band(), hud, ROW_BASELINES[-1], center_x=CENTER - 470))

    assert rows == rows_in(dialogue_band())


def test_licence_plate_with_dark_letters_is_not_a_row():
    band = flat(DARK)
    cv2.rectangle(band, (CENTER - 70, 105), (CENTER + 70, 150), (225, 225, 225), -1)

    assert rows_in(write(band, "BM472H", ROW_BASELINES[-1], level=20, outline=0, scale=1.0)) == ()


def test_white_blob_with_a_dark_edge_is_not_a_row():
    band = flat(DARK)
    cv2.rectangle(band, (CENTER - 150, 110), (CENTER + 150, 150), (255, 255, 255), -1)
    cv2.rectangle(band, (CENTER - 153, 107), (CENTER + 153, 153), (0, 0, 0), 3)

    assert rows_in(band) == ()


def test_white_text_without_a_dark_outline_is_not_a_row():
    assert rows_in(write(flat(MID_GREY, noise=5), DIALOGUE, ROW_BASELINES[-1], outline=0)) == ()


def test_narrow_row_is_found_but_not_sure():
    rows = rows_in(write(flat(DARK), "No.", ROW_BASELINES[-1]))

    assert len(rows) == 1
    assert not rows[0].sure


def test_band_that_does_not_reach_the_frame_centre_has_no_rows():
    assert detect_rows(dialogue_band(), 0, 2 * (BAND_WIDTH + 200)) == ()


@pytest.mark.parametrize("limit, count", [(None, 1), (DIALOGUE_BOTTOM, 0)], ids=["no-limit", "dialogue-limit"])
def test_objective_row_is_returned_only_without_a_dialogue_limit(limit, count):
    band = write(flat(DARK), "Find Dunk", OBJECTIVE_BASELINE)

    assert len(rows_in(band, dialogue_bottom=limit)) == count


def test_dialogue_row_is_kept_when_the_objective_row_below_it_is_dropped():
    band = write(dialogue_band(), "Find Dunk", OBJECTIVE_BASELINE)

    assert boxes(rows_in(band, dialogue_bottom=DIALOGUE_BOTTOM)) == boxes(rows_in(dialogue_band()))


def test_sure_white_row_ends_the_search_before_dim_text_is_looked_for():
    rows = rows_in(tv_scene_with_objective())

    assert [(row.dim, row.top > DIALOGUE_BOTTOM) for row in rows] == [(False, True)]


def test_objective_row_below_the_limit_does_not_hide_a_dim_dialogue_row():
    rows = rows_in(tv_scene_with_objective(), dialogue_bottom=DIALOGUE_BOTTOM)

    assert len(rows) == 1
    assert rows[0].dim and rows[0].top < ROW_BASELINES[-1] < rows[0].bottom


@pytest.mark.parametrize("below_centre, kept", [(0.0, True), (-0.5, False)])
def test_row_is_kept_exactly_when_its_box_centre_is_not_below_the_limit(below_centre, kept):
    row = rows_in(dialogue_band())[0]

    rows = rows_in(dialogue_band(), dialogue_bottom=(row.top + row.bottom) / 2 + below_centre)

    assert bool(rows) is kept


def test_any_limit_keeps_exactly_the_rows_whose_centre_is_not_below_it():
    band = write(flat(DARK), "Find Dunk", OBJECTIVE_BASELINE)
    for text, baseline in zip(ROW_TEXTS, ROW_BASELINES):
        write(band, text, baseline)
    everything = rows_in(band)

    for limit in range(0, BAND_HEIGHT + 8, 4):
        expected = [row for row in everything if (row.top + row.bottom) / 2 <= limit]
        assert boxes(rows_in(band, dialogue_bottom=limit)) == boxes(expected), limit


@pytest.mark.parametrize("frame_width, interpolation", [(1280, cv2.INTER_AREA), (2560, cv2.INTER_LINEAR)])
def test_same_scene_in_another_frame_width_gives_the_same_rows_scaled(frame_width, interpolation):
    band = write(dialogue_band(), ROW_TEXTS[1], ROW_BASELINES[1])
    scale = frame_width / FRAME_WIDTH
    size = (round(BAND_WIDTH * scale), round(BAND_HEIGHT * scale))

    rows = detect_rows(cv2.resize(band, size, interpolation=interpolation), round(BAND_LEFT * scale), frame_width)

    reference = np.array(boxes(rows_in(band)))
    assert len(rows) == len(reference) == 2
    assert np.abs(np.array(boxes(rows)) / scale - reference).max() <= 2


@pytest.mark.parametrize("repack", [lambda band: cv2.cvtColor(band, cv2.COLOR_BGR2BGRA),
                                    lambda band: np.pad(band, ((9, 9), (9, 9), (0, 0)))[9:-9, 9:-9],
                                    lambda band: np.frombuffer(band.tobytes(), np.uint8).reshape(band.shape)],
                         ids=["bgra", "view-cut-from-a-bigger-image", "read-only"])
def test_band_in_another_memory_layout_gives_the_same_rows(repack):
    assert rows_in(repack(dialogue_band())) == rows_in(dialogue_band())


@pytest.mark.parametrize("band, frame_width", [(np.zeros((BAND_HEIGHT, BAND_WIDTH), np.uint8), FRAME_WIDTH),
                                               (np.zeros((BAND_HEIGHT, BAND_WIDTH, 3), np.float32), FRAME_WIDTH),
                                               (np.zeros((BAND_HEIGHT, BAND_WIDTH, 3), np.uint8), 0)],
                         ids=["grayscale", "float-image", "zero-frame-width"])
def test_unusable_input_is_rejected(band, frame_width):
    with pytest.raises(ValueError):
        detect(band, BAND_LEFT, frame_width)


def test_mask_of_a_detection_is_a_read_only_band_sized_mask_of_the_row_letters():
    detection = detect(dialogue_band(), BAND_LEFT, FRAME_WIDTH)

    assert detection.core.shape == (BAND_HEIGHT, BAND_WIDTH) and detection.core.dtype == np.uint8
    assert not detection.core.flags.writeable
    assert cv2.countNonZero(detection.core) == detection.rows[0].pixels
    assert detection.level == DEFAULT_PARAMS.white_level


def test_to_frame_moves_the_row_into_frame_coordinates():
    row = rows_in(dialogue_band())[0]

    moved = to_frame(row, BAND_LEFT, BAND_TOP)

    assert boxes([moved]) == [(row.left + BAND_LEFT, row.top + BAND_TOP, row.right + BAND_LEFT, row.bottom + BAND_TOP)]
    assert (moved.ink_left - row.ink_left, moved.ink_right - row.ink_right) == (BAND_LEFT, BAND_LEFT)
    assert (moved.ink_top - row.ink_top, moved.ink_bottom - row.ink_bottom) == (BAND_TOP, BAND_TOP)
    assert (moved.width, moved.height, moved.sure, moved.pixels) == (row.width, row.height, row.sure, row.pixels)


def test_default_thresholds_are_the_values_measured_on_the_recordings():
    # A felvételeken hangolt értékek, a tesztek viszont nem látják a felvételeket: módosítás csak újraméréssel
    assert astuple(DEFAULT_PARAMS) == (
        200, 150, 4.0, 8.0, 150, 25, 0.005, 40, 0.7, 90, 110.0, 2, 2.0, 9.0, 18.0, 32.0, 13.0, 26.0, 6.0,
        45.0, 0.07, 0.30, 0.062, 3.2, 3.0, 34.0, 15.0, 29.0, 0.085, 0.22, 0.08, 2.5, 7.0, 5.0, 8.0, 5.0, 120, 30, 80.0)


# --- színes beszélőnév (Lucia:, Brian:) a fehér sor elején vagy közepén -----------------------------

TEAL, ORANGE = (214, 207, 126), (60, 170, 240)  # BGR: a nevek színe a játékban (becsült)
PALE_TEAL = (200, 214, 168)  # BGR: Kirk halvány türkiz neve a felvételen (a csatornák szórása ~45)


def write_spoken(band, parts, baseline):
    """Egy középre igazított sor több darabból: (szöveg, BGR szín) párok, fekete körvonallal."""
    text = "".join(part for part, _color in parts)
    letters = glyphs(text, baseline)
    side = 2 * OUTLINE_PX + 1
    band[cv2.dilate(letters, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))) > 0] = 0
    (width, _), _ = cv2.getTextSize(text, FONT, FONT_SCALE, 1)
    x = round(CENTER - width / 2)
    colors = np.zeros(band.shape, np.float64)
    for part, color in parts:
        (part_width, _), _ = cv2.getTextSize(part, FONT, FONT_SCALE, 1)
        colors[:, x:x + part_width + 1] = color
        x += part_width
    alpha = letters[:, :, None] / 255.0
    band[:] = (band * (1.0 - alpha) + colors * alpha + 0.5).astype(np.uint8)
    return band


def ink_columns(text, baseline):
    columns = np.flatnonzero(glyphs(text, baseline).any(axis=0))
    return columns[0], columns[-1] + 1


@pytest.mark.parametrize("color", [TEAL, ORANGE, PALE_TEAL], ids=["teal", "orange", "pale teal"])
def test_coloured_speaker_name_belongs_to_its_row(color):
    band = write_spoken(flat(DARK), [("Lucia:", color), (" Stop talking, will you?", (255, 255, 255))],
                        ROW_BASELINES[-1])

    rows = rows_in(band)

    first, last = ink_columns("Lucia: Stop talking, will you?", ROW_BASELINES[-1])
    assert len(rows) == 1
    assert rows[0].ink_left <= first + 2 and rows[0].ink_right >= last - 2  # a név és a sor vége is benne van
    assert rows[0].sure


def test_coloured_name_in_the_middle_of_a_two_speaker_row_keeps_the_row_whole():
    parts = [("Vagrant:", ORANGE), (" Hey you. ", (255, 255, 255)), ("Lucia:", TEAL), (" Go away.", (255, 255, 255))]
    band = write_spoken(flat(DARK), parts, ROW_BASELINES[-1])

    rows = rows_in(band)

    first, last = ink_columns("".join(part for part, _ in parts), ROW_BASELINES[-1])
    assert len(rows) == 1
    assert rows[0].ink_left <= first + 2 and rows[0].ink_right >= last - 2


def test_coloured_name_in_the_middle_without_a_name_in_front_keeps_the_row_whole():
    parts = [("Hey you, over there. ", (255, 255, 255)), ("Lucia:", TEAL), (" Go.", (255, 255, 255))]
    band = write_spoken(flat(DARK), parts, ROW_BASELINES[-1])

    rows = rows_in(band)

    first, last = ink_columns("".join(part for part, _ in parts), ROW_BASELINES[-1])
    assert len(rows) == 1
    assert rows[0].ink_left <= first + 2 and rows[0].ink_right >= last - 2


def test_coloured_blob_right_of_the_text_does_not_join_the_row():
    # Valós eset (frames4): a mondat végi „?” mellé érő bőrszínű kéz miatt a sor jobbra nőtt, és nem lett biztos
    parts = [("Lucia:", TEAL), (" Hey, how did you get here again?", (255, 255, 255))]
    text = "".join(part for part, _ in parts)
    band = write_spoken(flat(DARK), parts, ROW_BASELINES[-1])
    first, last = ink_columns(text, ROW_BASELINES[-1])
    top = ROW_BASELINES[-1] - 14
    band[top:top + 16, last + 3:last + 6] = (90, 140, 210)  # vékony, bőrszínű (BGR) folt a „?” mellett

    rows = rows_in(band)

    assert len(rows) == 1
    assert rows[0].ink_left <= first + 2  # a név a sorhoz tartozik
    assert rows[0].ink_right <= last + 2  # a folt nem


def test_coloured_text_alone_is_not_a_subtitle_row():
    band = write_spoken(flat(DARK), [("Lucia: Stop talking, will you?", TEAL)], ROW_BASELINES[-1])

    assert rows_in(band) == ()


def test_coloured_blob_glued_right_of_an_off_centre_row_is_not_taken_for_a_name():
    # A balra csúszott sor jobb végéhez tapadó színes folt (bőr, háttér) középre húzná a sort,
    # mégsem a sor része: a név után mindig fehér szöveg jön, a sor jobb vége fehér betű
    from gta_felirat.row_detector import DEFAULT_PARAMS, Measures
    from gta_felirat.row_geometry import MASK_ON, _with_name

    measures = Measures(DEFAULT_PARAMS, 1.0, 500.0, (390, 610), None)
    lines = (0, 10)
    named = np.zeros((10, 1000), np.uint8)
    named[:, 300:700] = MASK_ON  # a fehér szöveg és a hozzá tapadó színes folt
    white_ink = np.zeros(1000, np.int32)
    white_ink[300:640] = 10  # a fehér szöveg: a közepe 470, a kép közepe 500

    assert _with_name(named, white_ink, lines, (300, 640), measures) is None
