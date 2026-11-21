import cv2
import numpy as np
import pytest

from gta_felirat import layer_compositor
from gta_felirat.config import AppConfig
from gta_felirat.frosted_cover import Backdrop
from gta_felirat.i18n import t
from gta_felirat.layer_compositor import OUTLINE_COLOR, LayerCompositor, draw_outline, put, to_patch
from gta_felirat.layer_plan import LayerView, make_plan
from gta_felirat.layered_surface import premultiply
from gta_felirat.ocr import Box
from gta_felirat.subtitle_layer import Shown, ShownLine, SubtitleLayer

WIDTH, HEIGHT = 1920, 1080
BAND = Box(230, 842, 1689, 1058)
UPPER = Box(700, 915, 1220, 948)
LOWER = Box(780, 957, 1140, 994)
MAX_ROUNDING_DIFFERENCE = 2  # a 8 bites előszorzott keverés ennyivel térhet el a lebegőpontostól
TWO_LINES = Shown("Nem akarunk bajt, oké?\nNa és te, hogy vagy?", Box(700, 915, 1220, 994),
                  (ShownLine("Nem akarunk bajt, oké?", UPPER), ShownLine("Na és te, hogy vagy?", LOWER)))


def game_band(letter_level=255):
    """Mozgalmas hátterű feliratsáv két angol sorral: fehér vonások fekete körvonallal."""
    noise = np.random.default_rng(7).integers(0, 255, (BAND.height, BAND.width, 3), dtype=np.uint8)
    band = cv2.GaussianBlur(noise, (0, 0), 6)
    for box in (UPPER, LOWER):
        top, left = box.top - BAND.top, box.left - BAND.left
        for x in range(left, left + box.width - 6, 10):
            band[top:top + box.height, x:x + 6] = 0
            band[top + 2:top + box.height - 2, x + 1:x + 5] = letter_level
    return band


def backdrop():
    return Backdrop(game_band(), BAND)


def config_with(**overrides):
    return AppConfig(**{"hide_original": True, "panel_opacity": 0, "font_size_ratio": 0.024, **overrides})


def reference(config, shown, hide_original, preview_anchor, game):
    """A lebegőpontos út képe előszorozva, és amit a réteg közben megjegyzett."""
    layer = SubtitleLayer(config, on_change=lambda _updates: None)
    canvas = np.zeros((HEIGHT, WIDTH, 4), np.uint8)
    drawn = layer.draw(canvas, shown, hide_original, preview_anchor, game)
    return premultiply(drawn), layer.last_rect, layer.last_cover


def fast(config, shown, hide_original, preview_anchor, game):
    center = None if config.subtitle_x is None else (int(config.subtitle_x * WIDTH), int(config.subtitle_y * HEIGHT))
    is_preview = shown is None and preview_anchor is not None
    view = LayerView(WIDTH, HEIGHT, hide_original, game.area if game else None, center, False, is_preview)
    plan = make_plan(config, view, Shown(t("layer.preview_text"), preview_anchor) if is_preview else shown)
    target = np.zeros((HEIGHT, WIDTH, 4), np.uint8)
    changed = LayerCompositor().render(target, plan, game)
    return target, plan, changed


SCENARIOS = {
    "two translated lines, transparent panel": (config_with(), TWO_LINES, True, None, True),
    "two translated lines, dark panel": (config_with(panel_opacity=245), TWO_LINES, True, None, True),
    "two translated lines, half transparent panel": (config_with(panel_opacity=128), TWO_LINES, True, None, True),
    "hidden lines only": (config_with(), Shown("", LOWER, (), (UPPER, LOWER)), True, None, True),
    "one translated, one hidden": (
        config_with(), Shown("Nem akarunk bajt, oké?", UPPER, (ShownLine("Nem akarunk bajt, oké?", UPPER),), (LOWER,)),
        True, None, True),
    "block text at the original place": (config_with(panel_opacity=245), Shown("Szia, mi újság?", LOWER), True,
                                         None, True),
    "block text, transparent panel": (config_with(), Shown("Szia, mi újság?", LOWER), True, None, True),
    "panel dragged to a custom place": (config_with(subtitle_x=0.4, subtitle_y=0.3, panel_opacity=200), TWO_LINES,
                                        True, None, True),
    "translation beside the original": (config_with(hide_original=False, panel_opacity=245),
                                        Shown("Szia, mi újság?", LOWER), False, None, True),
    "no game picture: dark box cover": (config_with(panel_opacity=245), Shown("Szia, mi újság?", LOWER), True,
                                        None, False),
    "preview sample text": (config_with(panel_opacity=245), None, True, BAND, True),
}


@pytest.mark.parametrize("name", SCENARIOS)
def test_fast_path_draws_the_same_picture_as_the_float_path(name):
    config, shown, hide_original, preview_anchor, with_game = SCENARIOS[name]
    game = backdrop() if with_game else None

    expected, last_rect, last_cover = reference(config, shown, hide_original, preview_anchor, game)
    target, plan, changed = fast(config, shown, hide_original, preview_anchor, game)

    difference = np.abs(target.astype(np.int16) - expected.astype(np.int16))
    assert difference.max() <= MAX_ROUNDING_DIFFERENCE
    assert plan.last_rect == last_rect
    assert plan.last_cover == last_cover
    outside = target.copy()
    outside[changed.top:changed.bottom, changed.left:changed.right] = 0
    assert not outside.any()  # a jelentett területen kívül semmit sem rajzolt


def test_nothing_shown_clears_what_was_drawn_before():
    compositor = LayerCompositor()
    target = np.zeros((HEIGHT, WIDTH, 4), np.uint8)
    view = LayerView(WIDTH, HEIGHT, True, BAND)
    drawn = compositor.render(target, make_plan(config_with(), view, TWO_LINES), backdrop())

    cleared = compositor.render(target, None, None)

    assert not target.any()
    assert cleared == drawn  # ezt a területet kell újra kitenni a képernyőre
    assert compositor.render(target, None, None) is None  # másodszor már nincs változás


def test_covers_are_reused_until_the_game_picture_changes(monkeypatch):
    calls = []
    original = layer_compositor.frosted_layer
    monkeypatch.setattr(layer_compositor, "frosted_layer",
                        lambda *args: calls.append(args[1:]) or original(*args))
    compositor = LayerCompositor()
    target = np.zeros((HEIGHT, WIDTH, 4), np.uint8)
    plan = make_plan(config_with(), LayerView(WIDTH, HEIGHT, True, BAND), TWO_LINES)
    game = backdrop()

    compositor.render(target, plan, game)
    compositor.render(target, plan, game)
    assert len(calls) == 2  # soronként egyszer, a második rajzolás a kész foltokat használja

    compositor.render(target, plan, backdrop())
    assert len(calls) == 4  # új játékkép: új takarás


def test_hidden_box_without_letters_draws_nothing_when_the_panel_is_transparent():
    empty_row = Box(700, 870, 1220, 900)  # itt nincs betű a játékképen
    plan = make_plan(config_with(), LayerView(WIDTH, HEIGHT, True, BAND), Shown("", empty_row, (), (empty_row,)))
    target = np.zeros((HEIGHT, WIDTH, 4), np.uint8)

    LayerCompositor().render(target, plan, backdrop())

    assert not target.any()


def test_plan_for_another_game_picture_size_does_not_crash():
    plan = make_plan(config_with(), LayerView(WIDTH, HEIGHT, True, BAND), TWO_LINES)
    smaller = Backdrop(game_band()[:100, :400], Box(BAND.left, BAND.top, BAND.left + 400, BAND.top + 100))
    target = np.zeros((HEIGHT, WIDTH, 4), np.uint8)

    LayerCompositor().render(target, plan, smaller)  # közben átméreteződött az ablak: nem takarunk, de nem is hal el

    assert target[UPPER.top:UPPER.bottom, :, 3].max() > 0  # a fordítás szövege azért kirajzolódik


def test_dragging_adds_an_outline_around_the_panel():
    view = LayerView(WIDTH, HEIGHT, True, BAND, custom_center=(800, 400), is_dragging=True)
    plan = make_plan(config_with(panel_opacity=200), view, TWO_LINES)
    target = np.zeros((HEIGHT, WIDTH, 4), np.uint8)

    LayerCompositor().render(target, plan, backdrop())

    x, y, width, height = plan.last_rect
    assert plan.outline == Box(x, y, x + width, y + height)
    assert tuple(target[y, x + width // 2]) == OUTLINE_COLOR
    assert tuple(target[y + height - 1, x + width // 2]) == OUTLINE_COLOR


def test_extra_outline_is_drawn_and_cleared_with_the_rest():
    compositor = LayerCompositor()
    target = np.zeros((HEIGHT, WIDTH, 4), np.uint8)

    changed = compositor.render(target, None, None, outlines=(BAND,))

    assert changed == BAND
    assert tuple(target[BAND.top, BAND.left + 50]) == OUTLINE_COLOR
    compositor.render(target, None, None)
    assert not target.any()


def test_reset_forgets_the_old_surface():
    compositor = LayerCompositor()
    compositor.render(np.zeros((HEIGHT, WIDTH, 4), np.uint8), None, None, outlines=(BAND,))
    compositor.reset()

    smaller = np.zeros((300, 400, 4), np.uint8)  # új, kisebb felület: a régi területet nem szabad törölni rajta
    assert compositor.render(smaller, None, None) is None


@pytest.mark.parametrize("alpha, expected", [(255, 200), (0, 100), (128, 150)])
def test_put_blends_a_premultiplied_patch_over_the_target(alpha, expected):
    target = np.full((4, 4, 4), 100, np.uint8)
    patch = to_patch(np.full((4, 4, 3), 200, np.uint8), np.full((4, 4), alpha, np.uint8))

    put(target, 0, 0, patch)

    assert abs(int(target[0, 0, 0]) - expected) <= 1


def test_put_clips_a_patch_that_hangs_over_the_edges():
    target = np.zeros((10, 10, 4), np.uint8)
    patch = to_patch(np.full((6, 6, 3), 200, np.uint8), np.full((6, 6), 255, np.uint8))

    put(target, 7, 7, patch)
    put(target, -3, -3, patch)

    assert target[9, 9, 3] == 255 and target[0, 0, 3] == 255
    assert target[5, 5, 3] == 0


def test_outline_is_clipped_to_the_target():
    target = np.zeros((20, 20, 4), np.uint8)

    draw_outline(target, Box(-5, -5, 10, 10))

    assert tuple(target[9, 3]) == OUTLINE_COLOR
    assert target[15, 15, 3] == 0
