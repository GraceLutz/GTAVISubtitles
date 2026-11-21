from band_drawing import AREA, DIALOGUE_BASELINE, FRAME, OBJECTIVE_BASELINE, band_with

from gta_felirat.fragment_filter import MIN_WHITE_FRACTION
from gta_felirat.row_finder import MIN_FRAME_WIDTH, ROW_FINDER, is_wide_enough

TROUBLE = "We ain't lookin' for trouble, okay?"
HOW = "How about you, how you doin' now?"


def test_dialogue_row_is_found_in_frame_coordinates_as_a_sure_subtitle_row():
    rows = ROW_FINDER.detect(band_with((TROUBLE, DIALOGUE_BASELINE)), AREA, FRAME)

    assert len(rows) == 1
    box = rows[0].box
    assert box.left < FRAME[0] // 2 < box.right  # középre igazított sor
    assert AREA.top + DIALOGUE_BASELINE - 40 < box.top < AREA.top + DIALOGUE_BASELINE < box.bottom + 8
    assert rows[0].sure
    assert rows[0].whiteness >= MIN_WHITE_FRACTION


def test_objective_row_under_the_dialogue_rows_is_left_out():
    band = band_with((TROUBLE, DIALOGUE_BASELINE), ("Follow Mia", OBJECTIVE_BASELINE))

    rows = ROW_FINDER.detect(band, AREA, FRAME)

    assert [row.box.bottom < AREA.top + OBJECTIVE_BASELINE - 20 for row in rows] == [True]


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
