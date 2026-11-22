"""A fordítás minden sora a saját angol sora helyére kerül, akkor is, ha a panelt korábban elhúzták."""

from dataclasses import replace

import pytest

from gta_felirat.config import AppConfig
from gta_felirat.layer_plan import LayerView, Shown, ShownLine, make_plan
from gta_felirat.line_placement import with_gone_lines_marked
from gta_felirat.ocr import Box

WIDTH, HEIGHT = 1920, 1080
LUCIA = Box(700, 900, 1220, 935)  # az első beszélő angol sora
JASON = Box(760, 980, 1160, 1015)  # a második beszélőé, lejjebb


def two_speakers():
    lines = (ShownLine("Lucia: Ne most.", LUCIA), ShownLine("Jason: Gyere már.", JASON))
    return Shown("Lucia: Ne most.\nJason: Gyere már.", Box(700, 900, 1220, 1015), lines)


def test_line_whose_english_is_gone_keeps_its_text_but_gets_no_cover():
    # Valós hiba: az angol sor eltűnése után a tartási időben a takarás tovább mosta el a (világos, mozgó) hátteret
    config = AppConfig(panel_opacity=0, hide_original=True)
    view = LayerView(WIDTH, HEIGHT, True, backdrop_area=Box(0, 0, WIDTH, HEIGHT))
    lucia, jason = two_speakers().lines
    shown = Shown("Lucia: Ne most.\nJason: Gyere már.", Box(700, 900, 1220, 1015),
                  (lucia, replace(jason, is_over_english=False)))

    plan = make_plan(config, view, shown)

    assert len(plan.texts) == 2
    assert len(plan.covers) == 1
    assert plan.covers[0].area.top < LUCIA.bottom < JASON.top


def test_lines_are_marked_where_no_english_row_is_left():
    lucia, jason = two_speakers().lines

    marked = with_gone_lines_marked(two_speakers(), [Box(705, 902, 1215, 933)])

    assert marked.lines == (lucia, replace(jason, is_over_english=False))


def test_marking_keeps_the_same_object_when_nothing_changed():
    shown = two_speakers()

    assert with_gone_lines_marked(shown, [LUCIA, JASON]) is shown


@pytest.mark.parametrize("custom_center", [None, (940, 400)], ids=["default place", "panel dragged earlier"])
def test_each_translated_line_sits_on_its_own_english_line(custom_center):
    # Valós hiba: elhúzott panelnél a két beszélő fordítása egy tömbben a panel helyére került,
    # az angol sorokat pedig máshol takartuk: a szöveg két helyen látszott
    config = AppConfig(panel_opacity=0, hide_original=True)
    view = LayerView(WIDTH, HEIGHT, True, custom_center=custom_center)

    plan = make_plan(config, view, two_speakers())

    centers = sorted(job.y + job.rendered.height // 2 for job in plan.texts)
    assert len(centers) == 2
    assert centers[0] == pytest.approx((LUCIA.top + LUCIA.bottom) / 2, abs=12)
    assert centers[1] == pytest.approx((JASON.top + JASON.bottom) / 2, abs=12)
