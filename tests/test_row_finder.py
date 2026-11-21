from band_drawing import AREA, CENTER_X, DIALOGUE_BASELINE, FRAME, OBJECTIVE_BASELINE, UPPER_BASELINE, band_with

from gta_felirat.fragment_filter import DIALOGUE_BOTTOM_RATIO, MIN_WHITE_FRACTION, white_fraction
from gta_felirat.row_finder import MIN_FRAME_WIDTH, ROW_FINDER, is_wide_enough, make_row_finder

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
