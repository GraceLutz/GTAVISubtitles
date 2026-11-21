import numpy as np
import pytest

from gta_felirat.config import AppConfig
from gta_felirat.layer_plan import LayerView, Shown, ShownLine, make_plan
from gta_felirat.ocr import Box
from gta_felirat.streamer_filter import censor


@pytest.mark.parametrize("text, expected", [
    ("Shut up, nigga.", "Shut up, n****."),
    ("NIGGER!", "N*****!"),
    ("My niggas are here", "My n***** are here"),
    ("Kussolj, niggám!", "Kussolj, n*****!"),  # magyar raggal
    ("Az a néger itt volt.", "Az a n**** itt volt."),
    ("Ezek a négerek", "Ezek a n******"),
    ("You fucking faggot", "You fucking f*****"),
    ("Hé, buzi!", "Hé, b***!"),
])
def test_slurs_are_starred_out_keeping_the_first_letter(text, expected):
    assert censor(text) == expected


@pytest.mark.parametrize("text", [
    "Niger is a country.",  # egy g: ország
    "Spicy tacos, please.",
    "Buzgó mócsing",
    "Lucia: Gyere ide a fenekeddel!",
    "El coche negro está aquí.",  # spanyolul a „negro” fekete
])
def test_ordinary_words_are_left_alone(text):
    assert censor(text) == text


def test_speaker_name_and_punctuation_stay_in_place():
    assert censor("Jason: Mi van, nigga?") == "Jason: Mi van, n****?"


ANCHOR = Box(600, 950, 1300, 990)
VIEW = LayerView(1920, 1080, hide_original=True)


def text_pixels(config, shown):
    plan = make_plan(config, VIEW, shown)
    return [np.asarray(job.rendered.rgba) for job in plan.texts]


@pytest.mark.parametrize("shown_of", [
    lambda text: Shown(text, ANCHOR),
    lambda text: Shown(text, ANCHOR, lines=(ShownLine(text, ANCHOR),)),
], ids=["one block", "line by line"])
def test_streamer_mode_draws_the_starred_text(shown_of):
    streamer = AppConfig(streamer_mode=True)

    drawn = text_pixels(streamer, shown_of("Jason: Mi van, nigga?"))

    expected = text_pixels(streamer, shown_of("Jason: Mi van, n****?"))
    assert all(np.array_equal(a, b) for a, b in zip(drawn, expected))


def test_without_streamer_mode_the_text_is_drawn_as_translated():
    plain = AppConfig(streamer_mode=False)

    drawn = text_pixels(plain, Shown("Jason: Mi van, nigga?", ANCHOR))

    starred = text_pixels(plain, Shown("Jason: Mi van, n****?", ANCHOR))
    assert not np.array_equal(drawn[0], starred[0])
