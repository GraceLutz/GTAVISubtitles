"""A figyelt sáv és a feliratpanel egérrel: áthelyezés, méretezés, mentés."""

import cv2
import numpy as np
import pytest

from gta_felirat.config import MAX_PANEL_WIDTH_RATIO, MIN_PANEL_WIDTH_RATIO, AppConfig, Region, load_config, save_settings
from gta_felirat.layer_compositor import CLICKABLE_ALPHA, LayerCompositor
from gta_felirat.layer_plan import LayerView, Shown, make_plan
from gta_felirat.ocr import Box, region_to_box
from gta_felirat.region_editor import (EDGE_GRAB_PX, MIN_HEIGHT_RATIO, MIN_WIDTH_RATIO, RegionEditor, edge_cursor,
                                        route_mouse)
from gta_felirat.subtitle_layer import SubtitleLayer

WIDTH, HEIGHT = 1920, 1080
START = Region(0.1, 0.7, 0.9, 0.95)
LONG_TEXT = ("Ez egy nagyon hosszú magyar mondat, amely biztosan nem fér el egyetlen sorban, ha a panel keskeny, "
             "de széles panelben kevesebb sorba törik.")


@pytest.fixture
def editor():
    saved = []
    region_editor = RegionEditor(START, on_change=saved.append)
    region_editor.enabled = True
    region_editor.box(WIDTH, HEIGHT)  # a kirajzolás adja meg a kép méretét
    region_editor.saved = saved
    return region_editor


def drag(target, start, end, *flags):
    target.handle_mouse(cv2.EVENT_LBUTTONDOWN, *start, *flags)
    target.handle_mouse(cv2.EVENT_MOUSEMOVE, *end, *flags)
    target.handle_mouse(cv2.EVENT_LBUTTONUP, *end, *flags)


def box():
    return region_to_box(START, WIDTH, HEIGHT)


# --- a figyelt sáv ------------------------------------------------------------------------

def test_band_moves_when_dragged_inside_and_is_saved_on_release(editor):
    editor.handle_mouse(cv2.EVENT_LBUTTONDOWN, 960, 900)
    editor.handle_mouse(cv2.EVENT_MOUSEMOVE, 960, 846)  # 54 képpont = a kép 5%-a feljebb

    assert editor.region == pytest.approx(Region(0.1, 0.65, 0.9, 0.9))
    assert editor.saved == []  # húzás közben még nem ment

    editor.handle_mouse(cv2.EVENT_LBUTTONUP, 960, 846)
    assert editor.saved == [{"subtitle_region": editor.region}]


@pytest.mark.parametrize("grab, to, expected", [
    pytest.param("left", -192, Region(0.0, 0.7, 0.9, 0.95), id="left edge widens to the frame edge"),
    pytest.param("right", 96, Region(0.1, 0.7, 0.95, 0.95), id="right edge"),
    pytest.param("top", -108, Region(0.1, 0.6, 0.9, 0.95), id="top edge: one more row of text fits"),
    pytest.param("bottom", 27, Region(0.1, 0.7, 0.9, 0.975), id="bottom edge"),
])
def test_band_edges_resize_only_their_own_side(editor, grab, to, expected):
    area = box()
    start = {"left": (area.left, 900), "right": (area.right, 900),
             "top": (960, area.top), "bottom": (960, area.bottom)}[grab]
    end = (start[0] + to, start[1]) if grab in ("left", "right") else (start[0], start[1] + to)

    drag(editor, start, end)

    assert editor.region == pytest.approx(expected)


def test_band_corner_resizes_both_sides(editor):
    area = box()
    drag(editor, (area.left + 2, area.top + 2), (area.left + 2 - 96, area.top + 2 - 54))

    assert editor.region == pytest.approx(Region(0.05, 0.65, 0.9, 0.95))


def test_band_cannot_leave_the_picture(editor):
    drag(editor, (960, 900), (960 + 1000, 900 + 1000))

    region = editor.region
    assert (region.right, region.bottom) == (1.0, 1.0)
    assert region.right - region.left == pytest.approx(START.right - START.left)


def test_band_cannot_shrink_to_nothing(editor):
    area = box()
    drag(editor, (area.left, 900), (area.right + 500, 900))
    drag(editor, (960, region_to_box(editor.region, WIDTH, HEIGHT).top), (960, 2000))

    assert editor.region.right - editor.region.left >= MIN_WIDTH_RATIO - 1e-9
    assert editor.region.bottom - editor.region.top >= MIN_HEIGHT_RATIO - 1e-9


def test_band_is_left_alone_when_not_being_set_up(editor):
    editor.enabled = False

    assert not editor.handle_mouse(cv2.EVENT_LBUTTONDOWN, 960, 900)
    assert editor.cursor_at(960, 900) == ""


def test_click_outside_the_band_goes_on_to_the_game(editor):
    assert not editor.handle_mouse(cv2.EVENT_LBUTTONDOWN, 960, box().top - EDGE_GRAB_PX - 5)


def test_cursor_shows_whether_the_band_moves_or_resizes(editor):
    area = box()
    assert editor.cursor_at(960, 900) == "fleur"
    assert editor.cursor_at(area.left, 900) == "sb_h_double_arrow"
    assert editor.cursor_at(960, area.top) == "sb_v_double_arrow"


def test_click_without_moving_saves_nothing(editor):
    drag(editor, (960, 900), (960, 900))

    assert editor.saved == []


def test_moved_band_is_saved_to_config_json_and_loaded_back(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")
    moved = Region(0.05, 0.62, 0.95, 0.97)

    save_settings(path, {"subtitle_region": moved})

    assert load_config(path).subtitle_region == moved


# --- a feliratpanel szélessége ------------------------------------------------------------------

def preview_layer(saved, **config):
    layer = SubtitleLayer(AppConfig(**{"panel_opacity": 0, **config}), on_change=saved.append)
    layer.plan((WIDTH, HEIGHT), None, True, preview_anchor=box())
    return layer


def test_preview_panel_spans_the_whole_translation_width():
    layer = preview_layer([], panel_width_ratio=0.7)

    assert layer.last_rect[2] == pytest.approx(WIDTH * 0.7, abs=2)


def test_panel_is_widened_by_dragging_its_edge_in_preview():
    saved = []
    layer = preview_layer(saved, panel_width_ratio=0.6)
    left, top, width, height = layer.last_rect
    center = left + width // 2

    drag(layer, (left + width, top + height // 2), (center + int(WIDTH * 0.45), top + height // 2), 0)

    assert saved == [{"panel_width_ratio": pytest.approx(0.9, abs=0.002)}]


def test_panel_width_stays_within_its_limits():
    saved = []
    layer = preview_layer(saved, panel_width_ratio=0.6)
    left, top, width, height = layer.last_rect

    drag(layer, (left, top + 5), (left + width // 2, top + 5), 0)  # a középpontig húzva

    assert saved == [{"panel_width_ratio": MIN_PANEL_WIDTH_RATIO}]
    assert MAX_PANEL_WIDTH_RATIO == 1.0


def test_panel_edge_only_widens_in_preview_while_playing_it_moves_the_panel():
    saved = []
    layer = SubtitleLayer(AppConfig(panel_opacity=200), on_change=saved.append)
    layer.plan((WIDTH, HEIGHT), Shown("Szia, mi újság?", Box(800, 950, 1120, 990)), True)
    left, top, width, height = layer.last_rect

    drag(layer, (left + 1, top + height // 2), (left - 300, top + height // 2), 0)

    assert list(saved[0]) == ["subtitle_x", "subtitle_y"]


@pytest.mark.parametrize("ratio", [0.5, 1.0])
def test_translation_wraps_inside_the_panel_width(ratio):
    config = AppConfig(panel_opacity=200, panel_width_ratio=ratio, subtitle_x=0.5, subtitle_y=0.5)
    view = LayerView(WIDTH, HEIGHT, True, custom_center=(960, 540))

    plan = make_plan(config, view, Shown(LONG_TEXT, Box(700, 950, 1220, 990)))

    assert plan.last_rect[2] <= WIDTH * ratio


# --- megfogható a sáv az átlátszó rétegen ----------------------------------------------------

def test_watched_band_can_be_grabbed_on_the_transparent_layer():
    target = np.zeros((HEIGHT, WIDTH, 4), np.uint8)
    band = box()

    LayerCompositor().render(target, None, None, outlines=(band,))

    inside = target[band.top + 10:band.bottom - 10, band.left + 10:band.right - 10, 3]
    assert inside.min() >= CLICKABLE_ALPHA  # az átlátszó réteg itt nem engedi át a kattintást a játékra
    assert inside.max() <= CLICKABLE_ALPHA  # szemmel nem látszik
    assert target[:band.top, :, 3].max() == 0  # a sávon kívül minden átlátszó marad


# --- a sáv és a mintapanel együtt (Minta mód) --------------------------------------------------

def test_preview_panel_does_not_cover_the_band():
    layer = preview_layer([])
    left, top, width, height = layer.last_rect
    band = box()

    assert top + height <= band.top or top >= band.bottom  # a sáv közepe így megfogható marad


def test_click_on_the_panel_without_moving_does_not_pin_it():
    saved = []
    layer = SubtitleLayer(AppConfig(panel_opacity=200), on_change=saved.append)
    layer.plan((WIDTH, HEIGHT), Shown("Szia, mi újság?", Box(800, 950, 1120, 990)), True)
    left, top, width, height = layer.last_rect

    drag(layer, (left + width // 2, top + height // 2), (left + width // 2, top + height // 2), 0)

    assert saved == []


def test_band_edge_wins_over_a_panel_lying_on_it(editor):
    saved = []
    band = box()
    layer = SubtitleLayer(AppConfig(panel_opacity=200, subtitle_x=band.left / WIDTH, subtitle_y=0.8),
                          on_change=saved.append)
    layer.plan((WIDTH, HEIGHT), None, True, preview_anchor=band)
    y = int(0.8 * HEIGHT)
    assert edge_cursor(editor, layer, band.left, y) == "sb_h_double_arrow"

    route_mouse(editor, layer, cv2.EVENT_LBUTTONDOWN, band.left, y, 0)
    route_mouse(editor, layer, cv2.EVENT_MOUSEMOVE, band.left - 96, y, 0)
    route_mouse(editor, layer, cv2.EVENT_LBUTTONUP, band.left - 96, y, 0)

    assert editor.region.left == pytest.approx(START.left - 0.05)
    assert saved == []


def test_panel_drag_keeps_its_moves_when_crossing_a_band_edge(editor):
    saved = []
    band = box()
    layer = SubtitleLayer(AppConfig(panel_opacity=200), on_change=saved.append)
    layer.plan((WIDTH, HEIGHT), Shown("Szia, mi újság?", Box(800, 950, 1120, 990)), True)
    left, top, width, height = layer.last_rect
    start = (left + width // 2, top + height // 2)

    route_mouse(editor, layer, cv2.EVENT_LBUTTONDOWN, *start, 0)
    route_mouse(editor, layer, cv2.EVENT_MOUSEMOVE, start[0], band.top, 0)
    route_mouse(editor, layer, cv2.EVENT_LBUTTONUP, start[0], band.top, 0)

    assert saved and saved[0]["subtitle_y"] == pytest.approx(band.top / HEIGHT, abs=0.002)
    assert editor.region == START


def test_preview_is_not_blanked_out_of_the_band_being_read():
    layer = preview_layer([])

    assert layer.is_preview
    layer.plan((WIDTH, HEIGHT), Shown("Szia", Box(800, 950, 1120, 990)), True, preview_anchor=box())
    assert not layer.is_preview
