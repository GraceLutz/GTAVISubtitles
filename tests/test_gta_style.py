import numpy as np
import pytest

from gta_felirat.config import THIN_FONT_PATH, AppConfig, ConfigError
from gta_felirat.overlay import hex_to_rgb, render_subtitle

TEAL = (126, 207, 214)


def pixels_of(rendered, rgb):
    rgba = rendered.rgba
    mask = np.all(rgba[:, :, :3] == rgb, axis=2) & (rgba[:, :, 3] == 255)
    return np.nonzero(mask)[1]  # x-koordináták


def render(text):
    return render_subtitle(text, THIN_FONT_PATH, 48, 1600, opacity=0, speaker_color=TEAL)


def test_speaker_name_is_drawn_in_gta_color_and_the_rest_in_white():
    rendered = render("Locals: Fogd be a pofád.")

    teal_x = pixels_of(rendered, TEAL)
    white_x = pixels_of(rendered, (255, 255, 255))

    assert teal_x.size > 0 and white_x.size > 0
    assert teal_x.max() < white_x.min()  # csak a név színes, a szöveg fehér


def test_every_speaker_line_gets_a_colored_name():
    two = render("Vagrant: Szia.\nLucia: Hallgass.")
    rows_with_teal = np.unique(np.nonzero(np.all(two.rgba[:, :, :3] == TEAL, axis=2))[0])

    assert rows_with_teal.max() - rows_with_teal.min() > 48  # mindkét sorban van színes név


def test_line_without_speaker_is_all_white():
    assert pixels_of(render("Menj a kikötőbe."), TEAL).size == 0


def test_default_style_is_thin_font_with_gta_speaker_color():
    config = AppConfig()

    assert config.font_path == THIN_FONT_PATH
    assert hex_to_rgb(config.speaker_color) == TEAL


@pytest.mark.parametrize("bad", ["teal", "#12345", "7ECFD6", "#GGGGGG"])
def test_invalid_speaker_color_is_rejected(bad):
    with pytest.raises(ConfigError):
        AppConfig(speaker_color=bad).validate()
