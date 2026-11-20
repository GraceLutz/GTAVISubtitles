import numpy as np

from gta_felirat.config import AppConfig
from gta_felirat.frosted_cover import Backdrop, cover_area, frosted_patch
from gta_felirat.ocr import Box
from gta_felirat.overlay_window import KEY_COLOR_BGR, protect_from_key
from gta_felirat.remote_play import effective_hide_original
from gta_felirat.subtitle_layer import Shown, SubtitleLayer

FEATHER = 12


def game_band(height=120, width=600):
    """Színátmenetes „játékkép” közepén fehér, fekete körvonalas angol szöveggel."""
    ys, xs = np.mgrid[0:height, 0:width]
    band = np.stack([xs % 256, (ys * 2) % 256, np.full_like(xs, 90)], axis=2).astype(np.uint8)
    band[50:70, 150:450] = 0  # körvonal
    band[54:66, 154:446] = 255  # betűk
    return band


def test_patch_keeps_size_and_type():
    source = game_band()
    patch = frosted_patch(source, FEATHER)
    assert patch.shape == source.shape and patch.dtype == np.uint8


def test_patch_edges_blend_into_the_sharp_game_image():
    source = game_band()
    patch = frosted_patch(source, FEATHER)
    assert np.array_equal(patch[0], source[0])
    assert np.array_equal(patch[:, -1], source[:, -1])


def test_english_text_disappears_under_the_patch():
    source = game_band()
    patch = frosted_patch(source, FEATHER)
    assert source[54:66, 154:446].min() == 255
    assert patch[54:66, 154:446].max() < 200


def test_patch_center_is_darker_than_the_game_for_readable_translation():
    source = game_band()
    patch = frosted_patch(source, FEATHER)
    assert patch[20:40, 40:100].mean() < source[20:40, 40:100].mean()


def test_patch_on_all_bright_background_does_not_fail():
    source = np.full((80, 300, 3), 240, np.uint8)
    patch = frosted_patch(source, FEATHER)
    assert patch.shape == source.shape


def test_cover_area_grows_around_text_and_stays_inside_limits():
    area = cover_area(Box(100, 900, 500, 940), margin=20, limits=Box(0, 850, 1920, 1080))
    assert area == Box(80, 880, 520, 960)
    clipped = cover_area(Box(5, 860, 500, 940), margin=20, limits=Box(0, 850, 1920, 1080))
    assert clipped.left == 0 and clipped.top == 850


def remote_play_layer(**overrides):
    config = AppConfig(hide_original=True, panel_opacity=0, **overrides)
    return SubtitleLayer(config, on_change=lambda _: None)


def key_colored_frame(height=1080, width=1920):
    frame = np.empty((height, width, 3), np.uint8)
    frame[:] = (3, 2, 1)
    return frame


def test_layer_draws_frosted_cover_from_the_backdrop():
    layer = remote_play_layer()
    band_area = Box(200, 850, 800, 970)
    shown = Shown("Lucia: Szia!", Box(350, 900, 650, 920))

    output = layer.draw(key_colored_frame(), shown, True, backdrop=Backdrop(game_band(), band_area))

    cover = layer.last_cover
    assert cover is not None
    assert band_area.left <= cover.left and cover.right <= band_area.right
    region = output[cover.top:cover.bottom, cover.left:cover.right]
    # A takarás nem kulcsszínű (nem átlátszó); legfeljebb a betűk élsimításánál akad egy-egy pixel,
    # azt az OverlayWindow (protect_from_key) javítja
    assert (region == (3, 2, 1)).all(axis=2).mean() < 0.001


def test_layer_without_backdrop_keeps_the_dark_box_cover():
    layer = remote_play_layer()
    shown = Shown("Lucia: Szia!", Box(350, 900, 650, 920))
    frame = np.full((1080, 1920, 3), 200, np.uint8)

    output = layer.draw(frame, shown, True)

    assert layer.last_cover == shown.anchor
    assert output[910, 360].max() < 50


def test_layer_does_not_cover_when_hiding_is_off():
    layer = remote_play_layer()
    shown = Shown("Lucia: Szia!", Box(350, 900, 650, 920))
    layer.draw(key_colored_frame(), shown, False, backdrop=Backdrop(game_band(), Box(200, 850, 800, 970)))
    assert layer.last_cover is None


def test_hiding_works_on_recorded_layer_when_ocr_reads_the_window_directly():
    config = AppConfig(hide_original=True, visible_in_recordings=True)
    assert effective_hide_original(config, ocr_sees_overlay=False)
    assert not effective_hide_original(config, ocr_sees_overlay=True)


def test_key_colored_pixels_inside_the_cover_become_opaque_black():
    frame = key_colored_frame(100, 200)
    frame[40:60, 50:150] = (90, 90, 90)
    frame[50, 60] = KEY_COLOR_BGR  # élsimításból véletlenül kulcsszínű pixel a takarásban

    output = protect_from_key(frame, Box(50, 40, 150, 60))

    assert tuple(output[50, 60]) == (0, 0, 0)
    assert tuple(output[10, 10]) == KEY_COLOR_BGR  # a takaráson kívül marad átlátszó
    assert tuple(frame[50, 60]) == KEY_COLOR_BGR  # az eredeti képet nem módosítja


def test_backdrop_from_a_larger_window_is_clipped_to_the_frame():
    layer = remote_play_layer()
    shown = Shown("Lucia: Szia!", Box(350, 560, 650, 590))  # az ablakot közben kisebbre húzták
    frame = key_colored_frame(600, 1000)

    layer.draw(frame, shown, True, backdrop=Backdrop(game_band(120, 600), Box(200, 540, 800, 660)))

    assert layer.last_cover.bottom <= 600


def test_preview_sample_text_does_not_blur_the_whole_region():
    layer = remote_play_layer()
    region = Box(200, 850, 800, 970)
    layer.draw(key_colored_frame(), None, True, preview_anchor=region, backdrop=Backdrop(game_band(), region))
    assert layer.last_cover is None
