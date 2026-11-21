import numpy as np

from gta_felirat.config import THIN_FONT_PATH, AppConfig
from gta_felirat.layer_plan import LayerView, Shown, ShownLine, make_plan
from gta_felirat.line_placement import TranslatedLine, build_shown
from gta_felirat.ocr import Box, OcrLine
from gta_felirat.overlay import hex_to_rgb, render_subtitle

TEAL = (126, 207, 214)  # a config tartalék színe
PINK = (230, 60, 160)
GOLD = (240, 190, 40)
COLORS = (("Lucia", PINK), ("Jason", GOLD))
FRAME_WIDTH, FRAME_HEIGHT = 1920, 1080
UPPER = Box(700, 915, 1220, 948)
LOWER = Box(780, 957, 1140, 994)


def x_of(rgba, rgb):
    mask = np.all(rgba[:, :, :3] == rgb, axis=2) & (rgba[:, :, 3] == 255)
    return np.nonzero(mask)[1]


def rows_of(rgba, rgb):
    return np.unique(np.nonzero(np.all(rgba[:, :, :3] == rgb, axis=2))[0])


def render(text, speaker_colors=COLORS, speaker_color=TEAL):
    return render_subtitle(text, THIN_FONT_PATH, 48, 1600, opacity=0, speaker_color=speaker_color,
                           speaker_colors=speaker_colors).rgba


# --- render_subtitle -----------------------------------------------------------

def test_known_speaker_name_is_drawn_in_its_own_color_and_the_rest_white():
    rgba = render("Lucia: Hallgass.")

    pink, white = x_of(rgba, PINK), x_of(rgba, (255, 255, 255))
    assert pink.size > 0 and white.size > 0
    assert pink.max() < white.min()
    assert x_of(rgba, TEAL).size == 0


def test_unknown_speaker_falls_back_to_config_color():
    rgba = render("Vagrant: Szia.")

    assert x_of(rgba, TEAL).size > 0
    assert x_of(rgba, PINK).size == 0 and x_of(rgba, GOLD).size == 0


def test_name_lookup_ignores_case_and_surrounding_spaces():
    rgba = render("LUCIA: Hallgass.", speaker_colors=((" lucia ", PINK),))

    assert x_of(rgba, PINK).size > 0
    assert x_of(rgba, TEAL).size == 0


def test_speaker_colors_alone_split_the_name_without_config_color():
    rgba = render("Jason: Menjünk.", speaker_color=None)

    assert x_of(rgba, GOLD).size > 0


def test_two_speakers_on_two_lines_each_get_their_own_color():
    rgba = render("Lucia: Hallgass.\nJason: Menjünk.")

    pink_rows, gold_rows = rows_of(rgba, PINK), rows_of(rgba, GOLD)
    assert pink_rows.size > 0 and gold_rows.size > 0
    assert pink_rows.max() < gold_rows.min()  # Lucia az első, Jason a második sorban


def test_second_known_speaker_inside_one_line_is_colored_too():
    rgba = render("Lucia: Hallgass. Jason: Menjünk.")

    pink, gold, white = x_of(rgba, PINK), x_of(rgba, GOLD), x_of(rgba, (255, 255, 255))
    assert pink.size > 0 and gold.size > 0
    assert pink.max() < gold.min() < white.max()


def test_unknown_name_in_the_middle_of_a_line_stays_white():
    rgba = render("Lucia: Azt mondta. Note: semmi.")

    assert x_of(rgba, TEAL).size == 0


def test_empty_speaker_colors_render_exactly_as_before():
    text = "Lucia: Hallgass. Jason: Menjünk.\nVagrant: Szia."
    legacy = render_subtitle(text, THIN_FONT_PATH, 48, 1600, (0, 0), 0, TEAL)

    assert np.array_equal(render(text, speaker_colors=()), legacy.rgba)


# --- build_shown ---------------------------------------------------------------

def pairs():
    return [(TranslatedLine("Lucia: Stop talking.", "Lucia: Hallgass."), OcrLine("Lucia: Stop talking.", UPPER)),
            (TranslatedLine("Jason: Let's go.", "Jason: Menjünk."), OcrLine("Jason: Let's go.", LOWER))]


def test_build_shown_carries_speaker_colors():
    shown = build_shown(pairs(), [], None, speaker_colors=COLORS)

    assert shown is not None and shown.speaker_colors == COLORS


def test_speaker_color_change_produces_a_new_shown():
    first = build_shown(pairs(), [], None, speaker_colors=COLORS)

    same = build_shown(pairs(), [], first, speaker_colors=COLORS)
    changed = build_shown(pairs(), [], first, speaker_colors=(("Lucia", GOLD),))

    assert same is first
    assert changed is not first and changed.speaker_colors == (("Lucia", GOLD),)


# --- make_plan -----------------------------------------------------------------

def plan_texts(view):
    shown = Shown("Lucia: Hallgass.\nJason: Menjünk.", Box(700, 915, 1220, 994),
                  (ShownLine("Lucia: Hallgass.", UPPER), ShownLine("Jason: Menjünk.", LOWER)),
                  speaker_colors=COLORS)
    plan = make_plan(AppConfig(), view, shown)
    assert plan is not None
    return [job.rendered.rgba for job in plan.texts]


def test_line_by_line_plan_uses_speaker_colors():
    texts = plan_texts(LayerView(FRAME_WIDTH, FRAME_HEIGHT, hide_original=True))

    assert len(texts) == 2
    assert any(x_of(rgba, PINK).size for rgba in texts)
    assert any(x_of(rgba, GOLD).size for rgba in texts)


def test_block_plan_uses_speaker_colors():
    texts = plan_texts(LayerView(FRAME_WIDTH, FRAME_HEIGHT, hide_original=False))

    assert len(texts) == 1
    assert x_of(texts[0], PINK).size > 0 and x_of(texts[0], GOLD).size > 0


def test_config_fallback_color_is_still_the_default_teal():
    assert hex_to_rgb(AppConfig().speaker_color) == TEAL
