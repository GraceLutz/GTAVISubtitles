import json

import cv2
import numpy as np
import pytest

from gta_felirat.config import AppConfig, ConfigError, config_from_dict, load_config, save_settings
from gta_felirat.ocr import Box
from gta_felirat.subtitle_layer import Shown, SubtitleLayer

ORIGINAL = Box(600, 900, 1300, 1000)


def make_layer(**config_values):
    changes = []
    layer = SubtitleLayer(AppConfig(**config_values), on_change=changes.append)
    frame = np.full((1080, 1920, 3), 100, np.uint8)
    layer.draw(frame, Shown("Szia, mi újság?", ORIGINAL), hide_original=True)
    return layer, changes, frame


def drawn_center(layer):
    left, top, width, height = layer._last_rect
    return left + width // 2, top + height // 2


def test_panel_sits_on_original_subtitle_by_default():
    layer, _, _ = make_layer()
    center_x, center_y = drawn_center(layer)
    assert abs(center_x - 950) <= 2 and abs(center_y - 950) <= 2


def test_dragging_moves_panel_and_saves_position_on_release():
    layer, changes, _ = make_layer()
    start = drawn_center(layer)
    layer.handle_mouse(cv2.EVENT_LBUTTONDOWN, *start, 0)
    layer.handle_mouse(cv2.EVENT_MOUSEMOVE, 960, 200, 0)
    assert changes == []  # húzás közben még nem ment
    layer.handle_mouse(cv2.EVENT_LBUTTONUP, 960, 200, 0)
    assert changes == [{"subtitle_x": pytest.approx(0.5), "subtitle_y": pytest.approx(200 / 1080)}]


def test_panel_is_drawn_at_custom_position():
    layer, _, frame = make_layer(subtitle_x=0.5, subtitle_y=0.2)
    layer.draw(frame, Shown("Szia", ORIGINAL), hide_original=False)
    center_x, center_y = drawn_center(layer)
    assert abs(center_x - 960) <= 2 and abs(center_y - 216) <= 2


def test_custom_position_still_covers_original_english_text():
    layer, _, frame = make_layer(subtitle_x=0.5, subtitle_y=0.2)
    output = layer.draw(frame, Shown("Szia", ORIGINAL), hide_original=True)
    assert output[950, 950].mean() < frame[950, 950].mean()


def test_scroll_wheel_changes_font_size():
    layer, changes, _ = make_layer()
    layer.handle_mouse(cv2.EVENT_MOUSEWHEEL, *drawn_center(layer), 120 << 16)
    assert changes[0]["font_size_ratio"] > AppConfig().font_size_ratio


def test_right_click_resets_to_original_place():
    layer, changes, _ = make_layer(subtitle_x=0.5, subtitle_y=0.2)
    layer.handle_mouse(cv2.EVENT_RBUTTONDOWN, *drawn_center(layer), 0)
    assert changes == [{"subtitle_x": None, "subtitle_y": None}]


def test_clicks_outside_panel_are_not_taken():
    layer, changes, _ = make_layer()
    assert layer.handle_mouse(cv2.EVENT_LBUTTONDOWN, 5, 5, 0) is False
    assert changes == []


def test_preview_text_shows_panel_when_no_subtitle():
    layer = SubtitleLayer(AppConfig(), on_change=lambda _u: None)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    output = layer.draw(frame, None, hide_original=True, preview_anchor=ORIGINAL)
    assert output.max() > 0


@pytest.mark.parametrize("raw", [
    {"subtitle_x": 0.5},
    {"subtitle_x": 1.5, "subtitle_y": 0.5},
    {"panel_opacity": 300},
    {"font_size_ratio": 0.5},
    {"target_lang": " "},
])
def test_invalid_panel_settings_are_rejected(raw):
    with pytest.raises(ConfigError):
        config_from_dict(raw)


def test_save_settings_keeps_other_keys_and_round_trips(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"device_index": 3}), encoding="utf-8")
    save_settings(path, {"target_lang": "de", "subtitle_x": 0.123456, "subtitle_y": 0.5})
    config = load_config(path)
    assert config.device_index == 3
    assert config.target_lang == "de"
    assert config.subtitle_x == 0.1235


def test_save_settings_rejects_unknown_keys(tmp_path):
    with pytest.raises(ConfigError):
        save_settings(tmp_path / "config.json", {"api_key": "nope"})


def test_scroll_down_shrinks_font_with_negative_flags():
    layer, changes, _ = make_layer()
    layer.handle_mouse(cv2.EVENT_MOUSEWHEEL, *drawn_center(layer), -120 << 16)
    assert changes[0]["font_size_ratio"] < AppConfig().font_size_ratio
