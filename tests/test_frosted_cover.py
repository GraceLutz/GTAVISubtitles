import numpy as np

from gta_felirat.config import AppConfig
from gta_felirat.frosted_cover import Backdrop, cover_area, frosted_layer, frosted_patch
from gta_felirat.ocr import Box
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


def test_transparent_cover_erases_only_the_letters():
    source = game_band()
    patch = frosted_patch(source, FEATHER, strength=0.0)
    assert patch[54:66, 154:446].max() < 200
    assert np.array_equal(patch[20:40, 40:100], source[20:40, 40:100])


def test_transparent_cover_is_not_darkened():
    source = game_band()
    clear, _ = frosted_layer(source, FEATHER, strength=0.0)
    frosted, _ = frosted_layer(source, FEATHER, strength=1.0)
    assert clear.mean() > frosted.mean()


def bright_band(height=120, width=600):
    """Világos „póló” háttér, rajta fehér, fekete körvonalas angol szöveg."""
    band = np.full((height, width, 3), 235, np.uint8)
    band[50:70, 150:450] = 0  # körvonal
    band[54:66, 154:446] = 255  # betűk
    return band


def test_bright_background_is_left_alone_and_only_the_outlined_letters_go():
    # Valós hiba: fehér pólón a takarás szürke foltot hagyott, mert az egész világos hátteret betűnek vette
    source = bright_band()
    patch = frosted_patch(source, FEATHER, strength=0.0)
    assert np.array_equal(patch[20:35, 40:560], source[20:35, 40:560])  # a háttér érintetlen
    assert patch[50:70, 150:450].min() > 200  # a fekete körvonal is eltűnt: nem marad szürke folt


def test_dim_grey_tv_subtitle_is_erased_too():
    # Valós eset: a tévéműsor halvány (~183) feliratát a hófehér küszöb nem tüntette volna el
    source = np.full((120, 600, 3), 30, np.uint8)
    for x in range(150, 450, 10):
        source[52:68, x:x + 4] = 183
    patch = frosted_patch(source, FEATHER, strength=0.0)
    assert patch[52:68, 150:450].max() < 120


def test_patch_on_all_bright_background_does_not_fail():
    source = np.full((80, 300, 3), 240, np.uint8)
    patch = frosted_patch(source, FEATHER)
    assert patch.shape == source.shape


def test_cover_area_grows_around_text_and_stays_inside_limits():
    area = cover_area(Box(100, 900, 500, 940), margin=20, limits=Box(0, 850, 1920, 1080))
    assert area == Box(80, 880, 520, 960)
    clipped = cover_area(Box(5, 860, 500, 940), margin=20, limits=Box(0, 850, 1920, 1080))
    assert clipped.left == 0 and clipped.top == 850


def remote_play_layer(panel_opacity=0, **overrides):
    config = AppConfig(hide_original=True, panel_opacity=panel_opacity, **overrides)
    return SubtitleLayer(config, on_change=lambda _: None)


def transparent_layer(height=1080, width=1920):
    """A Remote Play réteg üres vászna: BGRA, mindenhol teljesen átlátszó."""
    return np.zeros((height, width, 4), np.uint8)


def test_layer_with_transparent_panel_erases_only_the_english_letters():
    layer = remote_play_layer(panel_opacity=0, subtitle_x=0.5, subtitle_y=0.2)
    band_area = Box(200, 850, 800, 970)
    shown = Shown("Szia!", Box(350, 900, 650, 920))

    output = layer.draw(transparent_layer(), shown, True, backdrop=Backdrop(game_band(), band_area))

    cover = layer.last_cover
    assert output[910, 500, 3] == 255  # az angol betű helyén teljesen fed
    assert output[cover.top + 6, (cover.left + cover.right) // 2, 3] == 0  # a betűk fölött átlátszó
    assert output[910, cover.left + 8, 3] == 0  # a szöveg mellett is átlátszó


def test_layer_draws_frosted_cover_from_the_backdrop():
    # A fordítás máshol áll, így csak a takarást nézzük
    layer = remote_play_layer(panel_opacity=255, subtitle_x=0.5, subtitle_y=0.2)
    band_area = Box(200, 850, 800, 970)
    shown = Shown("Lucia: Szia!", Box(350, 900, 650, 920))

    output = layer.draw(transparent_layer(), shown, True, backdrop=Backdrop(game_band(), band_area))

    cover = layer.last_cover
    assert cover is not None
    assert band_area.left <= cover.left and cover.right <= band_area.right
    anchor = shown.anchor
    # Az angol szöveg helyén teljesen fed, a széle felé átlátszóvá válik (az élő képbe olvad)
    assert output[anchor.top:anchor.bottom, anchor.left:anchor.right, 3].min() == 255
    assert output[cover.top, cover.left:cover.right, 3].max() == 0
    assert 0 < output[cover.top + 6, (cover.left + cover.right) // 2, 3] < 255


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
    layer.draw(transparent_layer(), shown, False, backdrop=Backdrop(game_band(), Box(200, 850, 800, 970)))
    assert layer.last_cover is None


def test_hiding_works_on_recorded_layer_when_ocr_reads_the_window_directly():
    config = AppConfig(hide_original=True, visible_in_recordings=True)
    assert effective_hide_original(config, ocr_sees_overlay=False)
    assert not effective_hide_original(config, ocr_sees_overlay=True)


def test_backdrop_from_a_larger_window_is_clipped_to_the_frame():
    layer = remote_play_layer()
    shown = Shown("Lucia: Szia!", Box(350, 560, 650, 590))  # az ablakot közben kisebbre húzták
    frame = transparent_layer(600, 1000)

    layer.draw(frame, shown, True, backdrop=Backdrop(game_band(120, 600), Box(200, 540, 800, 660)))

    assert layer.last_cover.bottom <= 600


def test_preview_sample_text_does_not_blur_the_whole_region():
    layer = remote_play_layer()
    region = Box(200, 850, 800, 970)
    layer.draw(transparent_layer(), None, True, preview_anchor=region, backdrop=Backdrop(game_band(), region))
    assert layer.last_cover is None
