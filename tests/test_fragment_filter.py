import numpy as np
import pytest

from gta_felirat.config import Region
from gta_felirat.fragment_filter import (Fragment, FrameCenter, is_hud_label, select_subtitle,
                                         white_fraction)
from gta_felirat.ocr import Box, frame_center, region_to_box

SUBTITLE_WHITENESS = 0.08
FULL_HD_CENTER = FrameCenter(960, 1920)


def fragment(text, left, top, right, bottom, whiteness=SUBTITLE_WHITENESS):
    return Fragment(text, Box(left, top, right, bottom), whiteness)


def texts(fragments):
    return [f.text for f in fragments]


def test_licence_plate_above_the_subtitle_is_dropped():
    # Valós eset: az autó rendszáma (sötét betű világos alapon) a felirat fölött
    plate = fragment("E36H", 1001, 875, 1076, 919, whiteness=0.0)
    line = fragment("Suck every dick!", 864, 957, 1057, 996)
    assert texts(select_subtitle([plate, line])) == ["Suck every dick!"]


def test_minimap_letter_far_left_is_dropped_even_though_it_is_white():
    minimap_n = fragment("N", 257, 876, 271, 893, whiteness=0.08)
    line = fragment("Eyes on the road, prick!", 823, 960, 1096, 994)
    assert texts(select_subtitle([minimap_n, line])) == ["Eyes on the road, prick!"]


def test_two_centred_subtitle_lines_are_both_kept():
    first = fragment("In Spanish: You're gonna kill somebody, asshole!", 689, 915, 1230, 948)
    second = fragment("It's not my fault your life is shit!", 777, 957, 1142, 994)
    assert texts(select_subtitle([second, first])) == [first.text, second.text]


def test_white_text_on_the_same_row_but_far_to_the_side_is_dropped():
    line = fragment("Stop! Now!", 896, 958, 1026, 992)
    hud = fragment("SALTW", 1700, 960, 1800, 990)
    assert texts(select_subtitle([line, hud])) == ["Stop! Now!"]


def test_words_split_into_pieces_on_one_line_stay_together():
    left = fragment("In Spanish:", 751, 957, 900, 996)
    right = fragment("Open your eyes, asshole!", 910, 957, 1168, 996)
    assert texts(select_subtitle([right, left])) == ["In Spanish:", "Open your eyes, asshole!"]


def test_a_line_too_far_below_is_not_part_of_the_subtitle():
    line = fragment("Go to hell!", 894, 900, 1025, 935)
    far = fragment("Hey", 930, 1020, 990, 1055)
    assert texts(select_subtitle([line, far])) == ["Go to hell!"]


def test_notification_above_the_minimap_is_not_dialogue():
    # Valós eset: „Moses Bucked You $825” kártya a minitérkép fölött, a kép bal szélén
    notification = fragment("Moses Bucked You $825", 240, 842, 520, 874)
    assert select_subtitle([notification], FULL_HD_CENTER) == []


def test_dialogue_wins_over_a_longer_notification_at_the_side():
    notification = fragment("Hungry Health recovers until fed.", 240, 842, 640, 874)
    line = fragment("Bring it in.", 885, 957, 1035, 994)
    assert texts(select_subtitle([notification, line], FULL_HD_CENTER)) == ["Bring it in."]


def test_centred_two_line_dialogue_is_kept_with_a_frame_centre():
    first = fragment("In Spanish: You're gonna kill somebody, asshole!", 689, 915, 1230, 948)
    second = fragment("It's not my fault your life is shit!", 777, 957, 1142, 994)
    assert texts(select_subtitle([second, first], FULL_HD_CENTER)) == [first.text, second.text]


@pytest.mark.parametrize("width, height", [(1920, 1080), (1280, 720), (2560, 1440)])
def test_frame_centre_is_derived_from_the_subtitle_band(width, height):
    region = Region()
    center = frame_center(region, region_to_box(region, width, height))
    assert center.x == pytest.approx(width / 2, abs=2)
    assert center.frame_width == pytest.approx(width, abs=4)


def test_nothing_white_means_no_subtitle():
    assert select_subtitle([fragment("EEZ36H", 916, 885, 990, 925, whiteness=0.0)]) == []


@pytest.mark.parametrize("white_rows, expected", [(0, 0.0), (5, 0.5), (10, 1.0)])
def test_white_fraction_counts_pure_white_pixels_of_the_box(white_rows, expected):
    crop = np.full((20, 10, 3), 80, np.uint8)
    crop[10:10 + white_rows] = 255
    area = Box(100, 200, 110, 220)
    assert white_fraction(crop, Box(100, 210, 110, 220), area) == pytest.approx(expected)


def dim_text_patch(text_level, background_level):
    """Sáv 4 képpont vastag vonásokkal: a „betűk” a terület ~17%-a."""
    crop = np.full((30, 120, 3), background_level, np.uint8)
    for x in range(10, 110, 12):
        crop[5:25, x:x + 4] = text_level
    return crop


def test_dim_grey_tv_subtitle_on_a_dark_scene_counts_as_subtitle_text():
    # Valós eset: a játékbeli tévéműsor felirata szürkésfehér (~183), nem hófehér
    crop = dim_text_patch(text_level=183, background_level=30)
    assert white_fraction(crop, Box(0, 0, 120, 30), Box(0, 0, 120, 30)) >= 0.04


def test_grey_text_on_a_light_background_is_still_not_a_subtitle():
    crop = dim_text_patch(text_level=183, background_level=140)
    assert white_fraction(crop, Box(0, 0, 120, 30), Box(0, 0, 120, 30)) == 0.0


def test_dark_letters_on_a_light_plate_are_not_a_subtitle():
    crop = dim_text_patch(text_level=20, background_level=185)
    assert white_fraction(crop, Box(0, 0, 120, 30), Box(0, 0, 120, 30)) == 0.0


# --- küldetésutasítás és kijelző-feliratok: nem párbeszéd ----------------------

def full_hd_frame():
    region = Region()
    return frame_center(region, region_to_box(region, 1920, 1080))


def test_objective_row_under_the_dialogue_rows_is_not_translated():
    # Valós eset: a „Find Dunk” utasítás a párbeszédsorok alatt, külön sorban áll (y 1016–1051)
    dialogue = fragment("A'ight.", 918, 957, 1001, 994)
    objective = fragment("Find Dunk", 891, 1016, 1029, 1051, whiteness=0.17)
    assert texts(select_subtitle([dialogue, objective], full_hd_frame())) == ["A'ight."]


def test_objective_alone_is_not_a_subtitle():
    objective = fragment("Follow Mia", 885, 1009, 1034, 1057, whiteness=0.15)
    assert select_subtitle([objective], full_hd_frame()) == []


def test_lowest_dialogue_row_is_still_inside_the_dialogue_rows():
    line = fragment("Dunk! Let's go!", 870, 953, 1049, 998)
    assert texts(select_subtitle([line], full_hd_frame())) == ["Dunk! Let's go!"]


@pytest.mark.parametrize("width, height", [(1920, 1080), (1280, 720), (2560, 1440)])
def test_dialogue_bottom_scales_with_the_frame(width, height):
    region = Region()
    center = frame_center(region, region_to_box(region, width, height))
    assert center.dialogue_bottom == pytest.approx(0.93 * height, abs=3)


def test_weapon_label_near_the_centre_does_not_join_the_dialogue():
    # Valós eset: a mozgó „BASEBALL BAT” felirat a kép közepe táján, a párbeszédsor fölött
    label = fragment("BASEBALL BAT", 900, 920, 1087, 958, whiteness=0.19)
    line = fragment("Dunk! Let's go!", 870, 953, 1049, 996)
    assert texts(select_subtitle([label, line], full_hd_frame())) == ["Dunk! Let's go!"]


def test_hud_numbers_do_not_join_the_dialogue():
    # Valós eset: „0 100 20 Well get it…” – a kijelző számai a felirat mellett és alatt
    numbers = [fragment("0", 800, 960, 815, 990), fragment("100", 930, 1000, 985, 1030)]
    line = fragment("Where's the doctor?", 830, 957, 1090, 994)
    assert texts(select_subtitle([*numbers, line], full_hd_frame())) == ["Where's the doctor?"]


@pytest.mark.parametrize("text, expected", [
    ("BASEBALL BAT", True), ("100", True), ("0", True), ("$825", True),
    ("BM472H", True), ("BM4.72H", True), ("E36H", True), ("EEZ36H", True),
    ("OK.", False), ("I", False), ("Dunk! Let's go!", False), ("...", False), ("Yeah?", False),
])
def test_hud_label_is_digits_capitals_or_a_plate_code(text, expected):
    assert is_hud_label(text) is expected


def test_brightly_lit_licence_plate_does_not_join_the_dialogue():
    # Valós eset: a fényszóró megvilágította a táblát, így a színe alapján feliratnak tűnt:
    # „BM472H He just ran out there like, "ahh!"”
    plate = fragment("BM472H", 905, 905, 1015, 945, whiteness=0.55)
    line = fragment('He just ran out there like, "ahh!"', 760, 957, 1160, 994)
    assert texts(select_subtitle([plate, line], full_hd_frame())) == ['He just ran out there like, "ahh!"']


def test_plate_alone_is_not_a_subtitle():
    plate = fragment("BM4.72H", 905, 950, 1015, 990, whiteness=0.55)
    assert select_subtitle([plate], full_hd_frame()) == []
