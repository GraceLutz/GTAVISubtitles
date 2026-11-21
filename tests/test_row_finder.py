from dataclasses import replace

import cv2
import numpy as np
import pytest
from band_drawing import (AREA, CENTER_X, DIALOGUE_BASELINE, FONT, FONT_SCALE, FRAME, GREEN, OBJECTIVE_BASELINE, PINK,
                          UPPER_BASELINE, band_with, band_with_parts)

from gta_felirat.row_detector import DEFAULT_PARAMS

from gta_felirat.fragment_filter import DIALOGUE_BOTTOM_RATIO, MIN_WHITE_FRACTION, white_fraction
from gta_felirat.row_finder import MIN_FRAME_WIDTH, ROW_FINDER, find_rows, is_wide_enough, make_row_finder

TROUBLE = "We ain't lookin' for trouble, okay?"
HOW = "How about you, how you doin' now?"
SIDE_ROW_REACH = 20  # ennyi képsorral az alapvonal fölött és alatt: a felső sor betűi, a párbeszédsor még nem


def test_dialogue_row_is_found_in_frame_coordinates_as_a_sure_subtitle_row():
    rows = ROW_FINDER.detect(band_with((TROUBLE, DIALOGUE_BASELINE)), AREA, FRAME)

    assert len(rows) == 1
    box = rows[0].box
    assert box.left < FRAME[0] // 2 < box.right  # középre igazított sor
    assert AREA.top + DIALOGUE_BASELINE - 40 < box.top < AREA.top + DIALOGUE_BASELINE < box.bottom + 8
    assert rows[0].sure
    assert rows[0].whiteness >= MIN_WHITE_FRACTION


GREY_INTERJECTION = 125  # a beszélő nélküli közbeszólás betűmagja a felvételeken (~110-150)


def test_grey_interjection_under_a_white_row_counts_as_subtitle_coloured():
    band = band_with((TROUBLE, UPPER_BASELINE), ("Lu! I thought...", DIALOGUE_BASELINE, GREY_INTERJECTION))

    rows = ROW_FINDER.detect(band, AREA, FRAME)

    assert len(rows) == 2
    assert all(row.whiteness >= MIN_WHITE_FRACTION for row in rows)


def test_sure_row_with_off_white_letters_on_a_bright_scene_counts_as_subtitle_coloured():
    # Nappali jelenet, a tömörített képen a fehér betű alig éri el a hófehér szintet
    band = band_with((TROUBLE, DIALOGUE_BASELINE, 196), background=110)

    rows = ROW_FINDER.detect(band, AREA, FRAME)

    assert [row.sure for row in rows] == [True]
    assert white_fraction(band, rows[0].box, AREA) < MIN_WHITE_FRACTION  # a régi mérés szerint nem feliratszínű
    assert rows[0].whiteness >= MIN_WHITE_FRACTION


def test_objective_row_under_the_dialogue_rows_comes_last_and_is_never_sure():
    band = band_with((TROUBLE, DIALOGUE_BASELINE), ("Follow Mia", OBJECTIVE_BASELINE))

    rows = ROW_FINDER.detect(band, AREA, FRAME)

    assert [row.box.bottom < AREA.top + OBJECTIVE_BASELINE - 20 for row in rows] == [True, False]
    assert not rows[1].sure
    assert rows[1].box.top > DIALOGUE_BOTTOM_RATIO * FRAME[1]  # nem a párbeszédsor kétszer


def test_dialogue_row_alone_gives_no_objective_row():
    rows = ROW_FINDER.detect(band_with((TROUBLE, DIALOGUE_BASELINE)), AREA, FRAME)

    assert len(rows) == 1


def test_same_text_on_the_next_capture_is_the_same_row():
    first = ROW_FINDER.detect(band_with((TROUBLE, DIALOGUE_BASELINE), seed=1), AREA, FRAME)[0]
    second = ROW_FINDER.detect(band_with((TROUBLE, DIALOGUE_BASELINE), seed=2), AREA, FRAME)[0]

    assert ROW_FINDER.is_same_row(first.signature, second.signature)


def test_another_text_at_the_same_place_is_a_different_row():
    first = ROW_FINDER.detect(band_with((TROUBLE, DIALOGUE_BASELINE)), AREA, FRAME)[0]
    second = ROW_FINDER.detect(band_with((HOW, DIALOGUE_BASELINE)), AREA, FRAME)[0]

    assert not ROW_FINDER.is_same_row(first.signature, second.signature)


def test_band_without_text_has_no_rows():
    assert list(ROW_FINDER.detect(band_with(), AREA, FRAME)) == []


def test_only_a_wide_enough_picture_is_read_by_rows():
    # A sorkereső küszöbei 1920 széles képre készültek: keskenyebb ablakban a teljes olvasás marad
    assert is_wide_enough(1920)
    assert is_wide_enough(MIN_FRAME_WIDTH)
    assert not is_wide_enough(1280)


def side_text_band():
    """Középre igazított párbeszédsor, és egy szöveg a sáv bal oldalán, a fölötte lévő sorban."""
    band = band_with((TROUBLE, DIALOGUE_BASELINE))
    side = band_with((HOW, UPPER_BASELINE))
    shift = CENTER_X // 2  # a felső sor közepe a sáv bal negyedébe kerül
    lines = slice(UPPER_BASELINE - SIDE_ROW_REACH, UPPER_BASELINE + SIDE_ROW_REACH)
    band[lines, :-shift] = side[lines, shift:]
    return band


def test_the_default_finder_sees_only_the_centred_row():
    rows = ROW_FINDER.detect(side_text_band(), AREA, FRAME)

    assert len(rows) == 1 and rows[0].sure


def test_the_whole_band_finder_also_finds_text_away_from_the_centre_and_never_sure():
    rows = make_row_finder(whole_band=True).detect(side_text_band(), AREA, FRAME)

    assert len(rows) == 2
    side, centred = rows
    assert side.box.right < FRAME[0] // 2 and not side.sure
    assert centred.sure and centred.box.left < FRAME[0] // 2 < centred.box.right


def test_the_whole_band_finder_knows_a_side_row_again_on_the_next_capture():
    finder = make_row_finder(whole_band=True)
    first = finder.detect(side_text_band(), AREA, FRAME)[0]
    second = finder.detect(side_text_band(), AREA, FRAME)[0]

    assert finder.is_same_row(first.signature, second.signature)


def test_the_finder_without_the_whole_band_is_the_default_one():
    band = side_text_band()

    assert ([row.box for row in make_row_finder(whole_band=False).detect(band, AREA, FRAME)]
            == [row.box for row in ROW_FINDER.detect(band, AREA, FRAME)])


WHITE = (255, 255, 255)
OFF_WHITE = (186, 186, 186)  # a tömörített kép fehér betűje világos háttér (rendszám) előtt


def text_span(parts, thickness, font_scale=FONT_SCALE):
    """A középre igazított, darabokból álló sor első és utolsó oszlopa a teljes képen."""
    width = sum(cv2.getTextSize(text, FONT, font_scale, thickness)[0][0] for text, _colour in parts)
    return AREA.left + CENTER_X - width / 2, AREA.left + CENTER_X + width / 2


@pytest.mark.parametrize("soft_white_level, is_found_whole", [
    pytest.param(DEFAULT_PARAMS.soft_white_level, True, id="with the soft white pass"),
    pytest.param(DEFAULT_PARAMS.white_level, False, id="without it the row lost its name"),
])
def test_off_white_named_row_over_a_bright_scene_is_found_whole_and_sure(soft_white_level, is_found_whole):
    parts = [("Brian: ", GREEN), ("My advice... stay dead a while longer.", OFF_WHITE)]
    band = band_with_parts(parts, DIALOGUE_BASELINE, background=130)
    params = replace(DEFAULT_PARAMS, soft_white_level=soft_white_level)

    rows = find_rows(band, AREA, FRAME, params)

    left, right = text_span(parts, thickness=1)
    whole = [row for row in rows if row.sure and row.box.left <= left and row.box.right >= right]
    assert bool(whole) == is_found_whole


@pytest.mark.parametrize("parts", [
    pytest.param([("Go to your ", WHITE), ("car", PINK)], id="coloured keyword at the end"),
    pytest.param([("Lose the cops", WHITE)], id="all white"),
    # A hosszú kulcsszó a fehér részt messze a középtől balra tolja
    pytest.param([("Find a way ", WHITE), ("backstage", PINK)], id="long coloured keyword"),
])
def test_bold_objective_row_is_found_with_its_whole_text(parts):
    band = band_with_parts(parts, OBJECTIVE_BASELINE, thickness=2)

    rows = ROW_FINDER.detect(band, AREA, FRAME)

    left, right = text_span(parts, thickness=2)
    assert len(rows) == 1 and not rows[0].sure
    assert rows[0].box.left <= left and rows[0].box.right >= right


@pytest.mark.parametrize("parts", [
    # Valós hiba: a rövid fehér rész a középtől balra áll, a tükrös vágás egyetlen szóra szűkítette
    pytest.param([("Go to the ", WHITE), ("office", PINK)], id="short white part off the centre"),
    pytest.param([("Leave ", WHITE), ("Delights", PINK)], id="one short white word"),
    pytest.param([("Rob the ", WHITE), ("safe", PINK)], id="short and dense"),
])
def test_short_bold_objective_row_is_found_with_its_whole_text(parts):
    band = band_with_parts(parts, OBJECTIVE_BASELINE, thickness=2, font_scale=BIGGER_OBJECTIVE_FONT)

    rows = ROW_FINDER.detect(band, AREA, FRAME)

    left, right = text_span(parts, thickness=2, font_scale=BIGGER_OBJECTIVE_FONT)
    assert len(rows) == 1 and not rows[0].sure
    assert rows[0].box.left <= left and rows[0].box.right >= right


def test_coloured_background_after_the_keyword_does_not_widen_the_objective_row():
    # A színes kulcsszó mögött színes háttérfolt (lámpa, plakát) tapad a sorhoz: a sor a fehér eleje tükörképéig tart
    parts = [("Find a way ", WHITE), ("backstage", PINK)]
    band = band_with_parts(parts, OBJECTIVE_BASELINE, thickness=2)
    left, right = text_span(parts, thickness=2)
    blob = np.zeros(band.shape[:2], np.uint8)
    cv2.putText(blob, "oooo", (round(right - AREA.left) + 25, OBJECTIVE_BASELINE), FONT, FONT_SCALE, 255, 2,
                cv2.LINE_AA)
    band[cv2.dilate(blob, np.ones((5, 5), np.uint8)) > 0] = 0
    band[blob > 0] = PINK

    rows = ROW_FINDER.detect(band, AREA, FRAME)

    assert len(rows) == 1
    assert rows[0].box.left <= left and right <= rows[0].box.right < right + 20


BIGGER_OBJECTIVE_FONT = 0.85  # az ablakból rögzített képen a küldetéssor betűje vastagabb (vonalvastagság ~4,7–5)


def test_objective_row_with_thicker_letters_is_found():
    band = band_with_parts([("Lose the cops", WHITE)], OBJECTIVE_BASELINE, thickness=2,
                           font_scale=BIGGER_OBJECTIVE_FONT)

    rows = ROW_FINDER.detect(band, AREA, FRAME)

    assert len(rows) == 1 and not rows[0].sure


def test_white_text_off_the_centre_in_the_objective_strip_is_not_an_objective_row():
    # Egy balra álló, a középig sem érő fehér szöveg (pl. kijelzőfelirat) nem küldetéssor, ha színes rész sem
    # egészíti ki középre
    band = band_with_parts([("Lose the cops", WHITE), (" " * 30, WHITE)], OBJECTIVE_BASELINE, thickness=2)

    assert ROW_FINDER.detect(band, AREA, FRAME) == []
