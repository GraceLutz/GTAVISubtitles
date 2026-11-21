import json

import numpy as np
import pytest

from gta_felirat.config import (AppConfig, ConfigError, Region, config_from_dict, load_config, save_settings,
                                with_overrides)
from gta_felirat.ocr import Box, has_bright_text, join_lines, region_to_box
from gta_felirat.overlay import blend, render_subtitle, subtitle_position


def test_missing_config_file_gives_defaults(tmp_path):
    assert load_config(tmp_path / "nincs.json") == AppConfig()


def test_config_file_values_are_loaded(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"device_index": 2, "subtitle_region": {"left": 0.1, "top": 0.7,
                                                                       "right": 0.9, "bottom": 0.95}}))
    config = load_config(path)
    assert config.device_index == 2
    assert config.subtitle_region == Region(0.1, 0.7, 0.9, 0.95)


@pytest.mark.parametrize("raw", [
    {"unknown_key": 1},
    {"translator": "babelfish"},
    {"subtitle_region": {"left": 0.9, "top": 0.1, "right": 0.1, "bottom": 0.9}},
    {"subtitle_region": {"left": -0.1, "top": 0.1, "right": 0.5, "bottom": 0.9}},
    {"stable_reads": 0},
])
def test_invalid_config_is_rejected(raw):
    with pytest.raises(ConfigError):
        config_from_dict(raw)


def test_continuous_capture_is_on_unless_the_config_switches_it_off(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"fast_capture": false}', encoding="utf-8")

    assert load_config(None).fast_capture is True
    assert load_config(path).fast_capture is False


def test_row_reading_is_on_unless_the_config_switches_it_off(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"row_reading": false}', encoding="utf-8")

    assert load_config(None).row_reading is True
    assert load_config(path).row_reading is False


def test_whole_band_text_is_on_unless_the_config_switches_it_off(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"whole_band_text": false}', encoding="utf-8")

    assert load_config(None).whole_band_text is True
    assert load_config(path).whole_band_text is False


def test_whole_band_text_survives_save_and_load(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"device_index": 1}', encoding="utf-8")

    save_settings(path, {"whole_band_text": False})

    config = load_config(path)
    assert config.whole_band_text is False
    assert config.device_index == 1


def test_overrides_ignore_none_values():
    config = with_overrides(AppConfig(), device_index=None, translator="google")
    assert config.device_index == 0
    assert config.translator == "google"


def test_region_to_box_scales_to_frame():
    assert region_to_box(Region(0.1, 0.5, 0.9, 1.0), 1000, 200) == Box(100, 100, 900, 200)


def test_bright_text_detection():
    dark = np.zeros((100, 400, 3), np.uint8)
    assert not has_bright_text(dark)
    dark[40:60, 50:350] = 255
    assert has_bright_text(dark)


def test_join_lines_orders_top_to_bottom_left_to_right():
    fragments = [("second", Box(10, 50, 100, 70)), ("line,", Box(110, 5, 200, 25)),
                 ("First", Box(10, 0, 100, 20)), ("line", Box(110, 52, 200, 72))]
    assert join_lines(fragments) == "First line, second line"


def test_rendered_panel_covers_original_subtitle_area():
    rendered = render_subtitle("Rövid ő ű", "C:/Windows/Fonts/segoeuib.ttf", 30, 800, (600, 90))
    assert rendered.width >= 600 and rendered.height >= 90
    assert rendered.rgba.shape[2] == 4


def test_position_is_clamped_inside_frame():
    rendered = render_subtitle("Szöveg", "C:/Windows/Fonts/segoeuib.ttf", 30, 800, (400, 80))
    x, y = subtitle_position(rendered, Box(0, 1000, 100, 1080), 1920, 1080)
    assert x == 0
    assert y + rendered.height <= 1080


def test_blend_returns_new_frame_and_leaves_input_untouched():
    frame = np.full((200, 400, 3), 255, np.uint8)
    rendered = render_subtitle("Szia", "C:/Windows/Fonts/segoeuib.ttf", 20, 300, (100, 40))
    result = blend(frame, rendered, 10, 10)
    assert frame.min() == 255
    assert result[10:10 + rendered.height, 10:10 + rendered.width].mean() < 255


def test_short_line_in_a_dark_scene_still_counts_as_bright_text():
    # Valós eset: az „It's me.” a feliratsáv ~0,1%-a; a régi küszöbbel be sem olvastuk
    band = np.zeros((216, 1459, 3), np.uint8)
    band[100:112, 700:726] = 255  # kb. 300 fehér pixel
    assert has_bright_text(band)


def test_dim_tv_subtitle_counts_as_bright_text():
    band = np.zeros((216, 1459, 3), np.uint8)
    band[100:112, 600:800] = 183
    assert has_bright_text(band)
