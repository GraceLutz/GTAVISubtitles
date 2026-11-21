"""A feliratpanel magassága: a Minta módban a panel tetejét vagy alját húzva állítható, és mentődik."""

import cv2
import pytest

from gta_felirat.config import (MAX_PANEL_HEIGHT_RATIO, MIN_PANEL_HEIGHT_RATIO, AppConfig, ConfigError, Region,
                                load_config, save_settings)
from gta_felirat.layer_plan import LayerView, Shown, ShownLine, make_plan
from gta_felirat.ocr import Box, region_to_box
from gta_felirat.region_editor import EDGE_GRAB_PX, RegionEditor, edge_cursor, route_mouse
from gta_felirat.subtitle_layer import SubtitleLayer

WIDTH, HEIGHT = 1920, 1080
BAND = region_to_box(Region(0.1, 0.7, 0.9, 0.95), WIDTH, HEIGHT)


def preview_layer(saved, **config):
    layer = SubtitleLayer(AppConfig(**{"panel_opacity": 0, **config}), on_change=saved.append)
    layer.plan((WIDTH, HEIGHT), None, True, preview_anchor=BAND)
    return layer


def replan(layer):
    layer.plan((WIDTH, HEIGHT), None, True, preview_anchor=BAND)
    return layer.last_rect


def drag(layer, start, end):
    layer.handle_mouse(cv2.EVENT_LBUTTONDOWN, *start, 0)
    layer.handle_mouse(cv2.EVENT_MOUSEMOVE, *end, 0)
    layer.handle_mouse(cv2.EVENT_LBUTTONUP, *end, 0)


def test_preview_panel_is_as_tall_as_set():
    layer = preview_layer([], panel_height_ratio=0.2)

    assert layer.last_rect[3] == pytest.approx(HEIGHT * 0.2, abs=2)


def test_dragging_the_top_edge_up_makes_the_panel_taller_and_its_bottom_stays():
    saved = []
    layer = preview_layer(saved)
    left, top, width, height = layer.last_rect
    bottom = top + height

    layer.handle_mouse(cv2.EVENT_LBUTTONDOWN, left + width // 2, top, 0)
    layer.handle_mouse(cv2.EVENT_MOUSEMOVE, left + width // 2, top - 108, 0)
    _, new_top, _, new_height = replan(layer)  # húzás közben is az egeret követi
    layer.handle_mouse(cv2.EVENT_LBUTTONUP, left + width // 2, top - 108, 0)

    assert new_top == pytest.approx(top - 108, abs=2)
    assert new_top + new_height == pytest.approx(bottom, abs=2)
    assert saved[0]["panel_height_ratio"] == pytest.approx((height + 108) / HEIGHT, abs=0.002)


def test_dragging_the_bottom_edge_up_makes_the_panel_shorter():
    saved = []
    layer = preview_layer(saved, panel_height_ratio=0.25)
    left, top, width, height = layer.last_rect

    drag(layer, (left + width // 2, top + height - 1), (left + width // 2, top + height - 1 - 108))

    assert saved[0]["panel_height_ratio"] == pytest.approx(0.15, abs=0.002)


@pytest.mark.parametrize("to, expected", [
    pytest.param(-2000, MAX_PANEL_HEIGHT_RATIO, id="no taller than the limit"),
])
def test_panel_height_stays_within_its_limits(to, expected):
    saved = []
    layer = preview_layer(saved)
    left, top, width, _height = layer.last_rect

    drag(layer, (left + width // 2, top), (left + width // 2, top + to))

    assert saved[0]["panel_height_ratio"] == expected


def test_grabbing_the_top_edge_without_moving_saves_nothing():
    saved = []
    layer = preview_layer(saved)
    left, top, width, _height = layer.last_rect

    drag(layer, (left + width // 2, top), (left + width // 2, top))

    assert saved == []


def test_a_moved_panel_keeps_its_fixed_edge_after_the_height_is_saved():
    saved = []
    layer = preview_layer(saved, subtitle_x=0.5, subtitle_y=0.4)
    left, top, width, height = layer.last_rect

    drag(layer, (left + width // 2, top), (left + width // 2, top - 100))

    assert saved[0]["panel_height_ratio"] == pytest.approx((height + 100) / HEIGHT, abs=0.002)
    assert saved[0]["subtitle_y"] == pytest.approx((top + height - (height + 100) / 2) / HEIGHT, abs=0.002)


def test_top_edge_shows_a_resize_cursor_only_in_preview():
    preview = preview_layer([])
    left, top, width, _height = preview.last_rect
    assert preview.cursor_at(left + width // 2, top) == "sb_v_double_arrow"

    playing = SubtitleLayer(AppConfig(panel_opacity=200), on_change=lambda _changes: None)
    playing.plan((WIDTH, HEIGHT), Shown("Szia, mi újság?", Box(800, 950, 1120, 990)), True)
    left, top, width, _height = playing.last_rect
    assert playing.cursor_at(left + width // 2, top) == "fleur"


def test_panel_bottom_and_band_top_can_both_be_grabbed():
    editor = RegionEditor(Region(0.1, 0.7, 0.9, 0.95), on_change=lambda _changes: None)
    editor.enabled = True
    editor.box(WIDTH, HEIGHT)
    layer = preview_layer([])
    left, top, width, height = layer.last_rect
    x = left + width // 2

    assert top + height + 2 * EDGE_GRAB_PX < BAND.top  # a két megfogható sáv nem fedi egymást
    assert edge_cursor(editor, layer, x, top + height - 1) == "sb_v_double_arrow"
    assert route_mouse(editor, layer, cv2.EVENT_LBUTTONDOWN, x, top + height - 1, 0)
    assert layer.is_dragging and not editor.is_dragging


def test_translation_panel_is_at_least_as_tall_as_set():
    config = AppConfig(panel_opacity=200, panel_height_ratio=0.2)
    view = LayerView(WIDTH, HEIGHT, True, custom_center=(960, 540))

    plan = make_plan(config, view, Shown("Szia!", Box(800, 950, 1120, 990)))

    assert plan.last_rect[3] == pytest.approx(HEIGHT * 0.2, abs=2)


def test_line_by_line_panel_is_at_least_as_tall_as_set():
    config = AppConfig(panel_opacity=200, panel_height_ratio=0.2)
    anchor = Box(800, 950, 1120, 990)
    shown = Shown("Szia!", anchor, lines=(ShownLine("Szia!", anchor),))

    plan = make_plan(config, LayerView(WIDTH, HEIGHT, True), shown)

    assert plan.panel.box.height == pytest.approx(HEIGHT * 0.2, abs=2)
    assert plan.last_rect[3] == plan.panel.box.height


def test_without_a_set_height_the_panel_fits_the_text():
    plan = make_plan(AppConfig(panel_opacity=200), LayerView(WIDTH, HEIGHT, True, custom_center=(960, 540)),
                     Shown("Szia!", Box(800, 950, 1120, 990)))

    assert plan.last_rect[3] < HEIGHT * MIN_PANEL_HEIGHT_RATIO * 2


@pytest.mark.parametrize("ratio", [MIN_PANEL_HEIGHT_RATIO / 2, MAX_PANEL_HEIGHT_RATIO + 0.1])
def test_panel_height_out_of_range_is_rejected(ratio):
    with pytest.raises(ConfigError):
        AppConfig(panel_height_ratio=ratio).validate()


def test_panel_height_is_saved_to_config_json_and_loaded_back(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")

    save_settings(path, {"panel_height_ratio": 0.18})

    assert load_config(path).panel_height_ratio == 0.18
    assert AppConfig().panel_height_ratio is None


def released(saved):
    """A mentett beállítással újra kirajzolt minta (amit elengedés után látsz)."""
    layer = SubtitleLayer(AppConfig(panel_opacity=0, **saved[-1]), on_change=lambda _changes: None)
    return replan(layer)


@pytest.mark.parametrize("band", [
    pytest.param(Region(0.1, 0.7, 0.9, 0.95), id="panel above the band, bottom edge pulled down"),
    pytest.param(Region(0.1, 0.05, 0.9, 0.2), id="panel below the band, top edge pulled up"),
])
def test_panel_stays_where_it_was_drawn_after_release(band):
    global BAND
    BAND, old_band = region_to_box(band, WIDTH, HEIGHT), BAND
    try:
        saved = []
        layer = preview_layer(saved)
        left, top, width, height = layer.last_rect
        grab_y, to_y = (top + height - 1, top + height + 29) if band.top > 0.5 else (top, top - 30)

        layer.handle_mouse(cv2.EVENT_LBUTTONDOWN, left + width // 2, grab_y, 0)
        layer.handle_mouse(cv2.EVENT_MOUSEMOVE, left + width // 2, to_y, 0)
        during = replan(layer)
        layer.handle_mouse(cv2.EVENT_LBUTTONUP, left + width // 2, to_y, 0)

        after = released(saved)
        assert after[1] == pytest.approx(during[1], abs=2)
        assert after[1] + after[3] == pytest.approx(during[1] + during[3], abs=2)
    finally:
        BAND = old_band


def test_panel_dragged_shorter_than_its_text_keeps_its_fixed_edge_and_real_height():
    saved = []
    layer = preview_layer(saved, subtitle_x=0.5, subtitle_y=0.3)
    left, top, width, height = layer.last_rect
    bottom = top + height

    layer.handle_mouse(cv2.EVENT_LBUTTONDOWN, left + width // 2, top, 0)
    layer.handle_mouse(cv2.EVENT_MOUSEMOVE, left + width // 2, top + 200, 0)
    during = replan(layer)
    layer.handle_mouse(cv2.EVENT_LBUTTONUP, left + width // 2, top + 200, 0)

    assert during[1] + during[3] == pytest.approx(bottom, abs=2)
    assert saved[0]["panel_height_ratio"] == pytest.approx(during[3] / HEIGHT, abs=0.002)
    after = released(saved)
    assert after[1] + after[3] == pytest.approx(bottom, abs=2)


def test_panel_with_nothing_to_draw_while_heightening_does_not_fail():
    # Valós hiba: magasítás közben eltűnt a minta (felirat jött), a rajzoló TypeError-t dobott
    layer = preview_layer([])
    left, top, width, _height = layer.last_rect
    layer.handle_mouse(cv2.EVENT_LBUTTONDOWN, left + width // 2, top, 0)
    layer.handle_mouse(cv2.EVENT_MOUSEMOVE, left + width // 2, top - 50, 0)

    english = Box(800, 950, 1120, 990)
    plan = layer.plan((WIDTH, HEIGHT), Shown("", english, (), (english,)), True)  # csak takarás, panel nélkül

    assert plan is not None and plan.last_rect is None
