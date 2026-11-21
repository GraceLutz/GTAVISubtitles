import numpy as np
import pytest

from gta_felirat.compositing import composite, paste
from gta_felirat.layered_surface import premultiply
from gta_felirat.overlay import blend, render_subtitle

FONT = "C:/Windows/Fonts/segoeui.ttf"


def alpha(value, shape=(2, 3)):
    return np.full((*shape, 1), value, np.float32)


def test_on_a_game_frame_the_color_is_mixed_with_the_picture():
    frame = np.full((2, 3, 3), 100, np.uint8)
    mixed = composite(frame, np.array([200, 200, 200]), alpha(0.25))
    assert mixed.shape == (2, 3, 3) and (mixed == 125).all()


def test_on_a_transparent_layer_the_color_keeps_its_own_opacity():
    layer = np.zeros((2, 3, 4), np.uint8)
    mixed = composite(layer, np.array([10, 20, 30]), alpha(0.5))
    assert tuple(mixed[0, 0]) == (10, 20, 30, 128)


def test_layering_over_a_half_transparent_layer_adds_up_the_opacity():
    layer = np.zeros((2, 3, 4), np.uint8)
    layer[:] = (0, 0, 0, 128)
    mixed = composite(layer, np.array([255, 255, 255]), alpha(0.5))
    assert int(mixed[0, 0, 3]) == pytest.approx(191, abs=1)
    assert int(mixed[0, 0, 0]) == pytest.approx(170, abs=1)  # 255·0,5 / 0,75


def test_paste_leaves_the_original_untouched_and_only_changes_the_box():
    layer = np.zeros((10, 10, 4), np.uint8)
    result = paste(layer, 2, 3, np.array([255, 255, 255]), alpha(1.0, (4, 5)))
    assert layer.max() == 0
    assert result[2:6, 3:8, 3].min() == 255
    assert result[:, :3].max() == 0 and result[6:].max() == 0


def test_subtitle_on_transparent_layer_has_soft_text_edges_and_clear_surroundings():
    rendered = render_subtitle("Szia!", FONT, 40, 800, opacity=0)
    layer = np.zeros((200, 400, 4), np.uint8)
    result = blend(layer, rendered, 20, 20)
    text_alpha = result[20:20 + rendered.height, 20:20 + rendered.width, 3]
    assert text_alpha.max() == 255  # a betűk teljesen fednek
    assert ((text_alpha > 0) & (text_alpha < 255)).any()  # élsimítás: lágy betűszél
    assert result[150:, :, 3].max() == 0  # a panelen kívül átlátszó


def test_premultiply_scales_color_by_opacity():
    pixel = np.array([[[200, 100, 50, 128]]], np.uint8)
    assert tuple(premultiply(pixel)[0, 0]) == (100, 50, 25, 128)
