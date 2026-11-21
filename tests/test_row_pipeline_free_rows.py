"""A soralapú út a sáv bárhol álló szövegével (whole_band_text): a kép közepétől távoli sorok.

A középre igazított felirat útja nem változik; a máshol álló sort csak elolvasása után, a teljes
szövegszűrő-lánc szerint fordítjuk, a saját angol sorára, akár a középső felirat mellé is.
"""

import pytest
from test_row_pipeline import LOWER, OBJECTIVE, TROUBLE, hu, make_scene, row  # noqa: F401 - make_scene: fixture
from test_subtitle_reader import AREA as READER_AREA
from test_subtitle_reader import LINE, MIDDLE_WIDTH, FakeEngine
from test_subtitle_reader import band_with as engine_band

from gta_felirat.config import AppConfig, Region
from gta_felirat.layer_plan import ShownLine
from gta_felirat.ocr import Box, SubtitleReader
from gta_felirat.row_pipeline import STILL_SECONDS, RowPipeline
from gta_felirat.translation_service import TranslationService
from gta_felirat.translator import CachedTranslator, TranslatorChoice

SIDE = Box(200, 957, 560, 994)  # a kép bal felén, messze a közepétől (x 960)
AT_BAND_EDGE = Box(1500, 957, 1800, 994)  # a sáv jobb széle 1804-nél: a sor szó közben elvágva
SIDE_OBJECTIVE = Box(200, 1016, 500, 1051)  # a párbeszédsorok alatt, oldalt
WHERE = "Lucia: Where are you going?"


def side_row(row_id, box=SIDE, first_seen=1.0, stable=2, **changes):
    """A kereső a máshol álló sorban sosem biztos (row_detector: a szabad sor sure=False)."""
    return row(row_id, box, first_seen=first_seen, stable=stable, sure=False, **changes)


def read_side(scene, text, box=SIDE):
    scene.show(1.0, side_row(1, box))
    scene.read(1.04, (1, text))
    scene.translated(1.3)


def test_off_centre_row_is_not_hidden_before_it_is_read(make_scene):
    scene = make_scene()

    scene.show(1.0, side_row(1, stable=1))

    assert scene.shown is None
    assert scene.backend.calls == []


def test_off_centre_spoken_row_is_translated_on_its_own_row_after_it_is_read(make_scene):
    scene = make_scene(only_spoken_lines=True)

    read_side(scene, WHERE)

    assert scene.backend.calls == ["Where are you going?"]
    assert scene.shown.lines == (ShownLine("Lucia: " + hu("Where are you going?"), SIDE),)


def test_off_centre_row_is_translated_alongside_a_centred_subtitle(make_scene):
    scene = make_scene()

    scene.show(1.0, row(1, LOWER, first_seen=1.0), side_row(2))
    scene.read(1.04, (1, TROUBLE), (2, WHERE))
    scene.translated(1.3)

    assert sorted(scene.backend.calls) == sorted([TROUBLE, "Where are you going?"])
    assert set(scene.shown.lines) == {ShownLine(hu(TROUBLE), LOWER),
                                      ShownLine("Lucia: " + hu("Where are you going?"), SIDE)}


def test_centred_subtitle_is_translated_alone_when_the_side_text_is_not_dialogue(make_scene):
    scene = make_scene()

    scene.show(1.0, row(1, LOWER, first_seen=1.0), side_row(2))
    scene.read(1.04, (1, TROUBLE), (2, "PICK UP LOOT"))
    scene.translated(1.3)

    assert scene.backend.calls == [TROUBLE]
    assert scene.shown.lines == (ShownLine(hu(TROUBLE), LOWER),)


def test_off_centre_text_that_is_not_spoken_is_left_alone(make_scene):
    scene = make_scene(only_spoken_lines=True)

    read_side(scene, "Go to the Vice City Port.")

    assert scene.shown is None
    assert scene.backend.calls == []


def test_off_centre_text_cut_off_by_the_band_edge_is_left_alone(make_scene):
    scene = make_scene()

    read_side(scene, "Moses: The Crawler I want you to boo", AT_BAND_EDGE)

    assert scene.shown is None
    assert scene.backend.calls == []


def test_off_centre_text_under_the_dialogue_rows_is_left_alone(make_scene):
    scene = make_scene()

    read_side(scene, WHERE, SIDE_OBJECTIVE)

    assert scene.shown is None
    assert scene.backend.calls == []


def test_off_centre_text_is_left_alone_when_the_whole_band_switch_is_off(make_scene):
    scene = make_scene(whole_band_text=False)

    read_side(scene, WHERE)

    assert scene.shown is None
    assert scene.backend.calls == []


def test_objective_under_a_centred_subtitle_stays_out_of_the_free_text(make_scene):
    scene = make_scene(translate_objectives=False)  # a küldetéssor saját útját a test_row_pipeline nézi

    scene.show(1.0, row(1, LOWER, first_seen=1.0), row(2, OBJECTIVE, first_seen=1.0))
    scene.read(1.04, (1, TROUBLE), (2, "Find Dunk"))
    scene.translated(1.3)

    assert scene.backend.calls == [TROUBLE]


# --- a máshol álló, két sorba tördelt mondat egy megszólalás ----------------------------------

SIDE_UP = Box(200, 913, 600, 950)  # a SIDE fölötti sor, ugyanott kezdődik (balra igazított kártya)
WRAPPED = ("Lucia: Got so bad once, I had bounties on me", "in three different states.")


@pytest.mark.parametrize("only_spoken_lines", [True, False])
def test_off_centre_sentence_wrapped_over_two_rows_is_translated_as_one_line_over_both_rows(make_scene,
                                                                                           only_spoken_lines):
    scene = make_scene(only_spoken_lines=only_spoken_lines)

    scene.show(1.0, side_row(1, SIDE_UP), side_row(2, SIDE))
    scene.read(1.04, (1, WRAPPED[0]), (2, WRAPPED[1]))
    scene.translated(1.3)

    whole = "Got so bad once, I had bounties on me in three different states."
    assert scene.backend.calls == [whole]
    assert scene.shown.lines == (ShownLine("Lucia: " + hu(whole), Box(SIDE_UP.left, SIDE_UP.top, SIDE_UP.right,
                                                                      SIDE.bottom)),)


# --- a máshol álló szöveg csak „Név: szöveg” megszólalásként felirat -----------------------------

@pytest.mark.parametrize("text", ["Gellhorn", "Call Ended", "Go to the Vice City Port.", "time to pay"])
def test_off_centre_text_without_a_speaker_is_left_alone_even_when_every_line_is_translated(make_scene, text):
    scene = make_scene(only_spoken_lines=False)

    read_side(scene, text)

    assert scene.shown is None
    assert scene.backend.calls == []


def test_centred_text_without_a_speaker_is_still_translated_when_every_line_is_translated(make_scene):
    scene = make_scene(only_spoken_lines=False)

    scene.speak()

    assert scene.backend.calls == [TROUBLE]


# --- a középen álló, de a középső tömbbe nem tartozó sor nem lesz szabad sor --------------------

NEAR_CENTRE_HIGH = Box(800, 850, 1120, 880)  # a kép közepén, de a párbeszédsor fölött, messze tőle


def test_row_near_the_centre_but_outside_the_centred_block_is_not_a_free_line(make_scene):
    scene = make_scene()

    scene.show(1.0, row(1, LOWER, first_seen=1.0), row(2, NEAR_CENTRE_HIGH, first_seen=1.0, sure=False))
    scene.read(1.04, (1, TROUBLE), (2, WHERE))
    scene.translated(1.3)

    assert scene.backend.calls == [TROUBLE]
    assert scene.shown.lines == (ShownLine(hu(TROUBLE), LOWER),)


# --- a bizonytalan szabad sor nem késlelteti a középső feliratot ---------------------------------

CAPTURE_SECONDS = 0.037  # két rögzítés között (27 kép/mp)
GIVE_UP_AT = 3.0


def translation_requested_at(scene, with_side_row):
    """Mikor kéri a középső felirat fordítását; with_side_row: mellette egy szabad sor, amely minden
    rögzítésen új azonosítót kap (pl. mozgó háttér előtti szöveg), így sosem lesz stabil."""
    now, side_id, captures = 1.0, 100, 0
    while now < GIVE_UP_AT:
        rows = [row(1, LOWER, first_seen=1.0, stable=2 + captures)]
        texts = [(1, TROUBLE)]
        if with_side_row:
            rows.append(side_row(side_id, first_seen=now, stable=1))
            texts.append((side_id, WHERE))
        scene.show(now, *rows)
        scene.read(now + 0.01, *texts)
        if scene.pipeline._translate_future is not None:
            return now
        side_id, captures, now = side_id + 1, captures + 1, now + CAPTURE_SECONDS
    return None


def test_unstable_side_row_does_not_hold_back_the_centred_translation(make_scene):
    alone = translation_requested_at(make_scene(), with_side_row=False)
    beside = translation_requested_at(make_scene(), with_side_row=True)

    assert alone is not None
    assert beside == alone


def test_side_row_that_stays_unstable_is_translated_once_the_picture_stands_still(make_scene):
    scene = make_scene()

    scene.show(1.0, side_row(1, stable=1))
    scene.read(1.01, (1, WHERE))
    scene.pipeline.pump(1.0 + 2 * STILL_SECONDS)  # nem jött újabb kép: áll a kép
    scene.translated(1.3)

    assert scene.backend.calls == ["Where are you going?"]


# --- a teljes olvasású (régi) út a kapcsolótól függetlenül csak a kép közepét nézi -----------------

@pytest.mark.parametrize("whole_band_text", [True, False])
def test_full_read_searches_only_the_middle_part_whatever_the_whole_band_switch_says(whole_band_text):
    # A teljes sáv a neurális keresőnek túl nagy (lekicsinyíti), és a középső feliratsorok elvesznének:
    # a máshol álló szöveg csak a soralapú út dolga
    engine = FakeEngine({LINE: TROUBLE, SIDE: WHERE})
    reader = SubtitleReader(Region(), AppConfig().min_text_score, engine)
    pipeline = RowPipeline(AppConfig(whole_band_text=not whole_band_text), reader, quick_service())
    try:
        pipeline.update_config(AppConfig(whole_band_text=whole_band_text))
        result = reader.read_band(engine_band(LINE, SIDE), READER_AREA)
    finally:
        pipeline.shutdown()

    assert engine.detected_widths == [MIDDLE_WIDTH]
    assert result.text == TROUBLE


class QuickBackend:
    def translate(self, text):
        return hu(text)


def quick_service():
    return TranslationService(TranslatorChoice("deepl", "hu", "", ""), CachedTranslator(QuickBackend()))
