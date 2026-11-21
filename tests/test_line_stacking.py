from gta_felirat.config import AppConfig
from gta_felirat.layer_plan import LayerView, Shown, ShownLine, make_plan
from gta_felirat.ocr import Box

CONFIG = AppConfig(hide_original=True, panel_opacity=0, font_size_ratio=0.024)
VIEW = LayerView(1920, 1080, hide_original=True)
FONT_PX = int(1080 * CONFIG.font_size_ratio)
JASON_ROW = Box(600, 915, 1320, 950)  # a játék egymás alatti sorai kb. 40 képpontra vannak
LUCIA_ROW = Box(650, 955, 1270, 990)
LONG = ("Aktívan a nyomában vagyok, ezért korlátoznom kell a sugárzásnak való kitettségemet, és a legtöbb "
        "estén a völgyben, a kalandpark közelében leszek megtalálható, ha keresel, csak szólj előre. "
        "A Pelican Valleyben, a Thorpe Adventures mellett, a régi benzinkútnál várlak majd, gyere egyedül.")


def _texts(*lines: ShownLine):
    shown = Shown("\n".join(line.text for line in lines), JASON_ROW, tuple(lines))
    return sorted(make_plan(CONFIG, VIEW, shown).texts, key=lambda job: job.y)


def _center_y(job) -> int:
    return job.y + job.rendered.height // 2


def test_each_speaker_stays_on_the_place_of_its_english_line():
    jason, lucia = _texts(ShownLine("Jason: Hé, nálad van a telefonom?", JASON_ROW),
                          ShownLine("Lucia: Igen, nálam, mindjárt viszem.", LUCIA_ROW))

    assert abs(_center_y(jason) - (JASON_ROW.top + JASON_ROW.bottom) // 2) <= 1
    assert abs(_center_y(lucia) - (LUCIA_ROW.top + LUCIA_ROW.bottom) // 2) <= 1


def test_two_row_translation_does_not_run_over_the_next_speaker():
    two_rows = Box(520, 880, 1400, 950)  # két képernyősoros angol mondat Lucia sora fölött
    upper, lower = _texts(ShownLine(LONG, two_rows), ShownLine("Lucia: Rendben.", LUCIA_ROW))

    assert upper.rendered.height > 3 * FONT_PX  # a magyar három sorba tört, az angol kettőben állt
    assert lower.y >= upper.y + upper.rendered.height - FONT_PX // 2


def test_lines_pushed_apart_at_the_bottom_stay_on_the_screen():
    bottom_row = Box(650, 1040, 1270, 1075)
    upper, lower = _texts(ShownLine(LONG, Box(520, 970, 1400, 1040)), ShownLine("Lucia: Rendben.", bottom_row))

    assert lower.y + lower.rendered.height <= 1080
    assert lower.y >= upper.y + upper.rendered.height - FONT_PX // 2
    assert upper.y >= 0
