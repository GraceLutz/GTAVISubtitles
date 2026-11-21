"""A fordítás minden sora a saját angol sora helyére kerül, akkor is, ha a panelt korábban elhúzták."""

import pytest

from gta_felirat.config import AppConfig
from gta_felirat.layer_plan import LayerView, Shown, ShownLine, make_plan
from gta_felirat.ocr import Box

WIDTH, HEIGHT = 1920, 1080
LUCIA = Box(700, 900, 1220, 935)  # az első beszélő angol sora
JASON = Box(760, 980, 1160, 1015)  # a második beszélőé, lejjebb


def two_speakers():
    lines = (ShownLine("Lucia: Ne most.", LUCIA), ShownLine("Jason: Gyere már.", JASON))
    return Shown("Lucia: Ne most.\nJason: Gyere már.", Box(700, 900, 1220, 1015), lines)


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
