from types import SimpleNamespace

import numpy as np
import pytest
from band_drawing import AREA as BAND_AREA
from band_drawing import DIALOGUE_BASELINE, band_with

from gta_felirat import eager_hide
from gta_felirat.config import Region
from gta_felirat.eager_hide import (CONFIRMED_HOLD_SECONDS, EAGER_HIDE_SECONDS, EagerCover, EagerHide,
                                    sure_rows, with_eager_boxes)
from gta_felirat.layer_plan import Shown, ShownLine
from gta_felirat.ocr import Box

UPPER = Box(700, 915, 1220, 948)
LOWER = Box(780, 957, 1140, 994)
AREA = Box(230, 842, 1689, 1058)


def test_new_row_is_hidden_from_the_capture_it_first_appears_in():
    eager = EagerHide()

    eager.observe((LOWER,), captured_at=10.0)

    assert eager.active(now=10.02) == (LOWER,)


def test_row_is_shown_again_when_reading_takes_too_long():
    eager = EagerHide()
    eager.observe((LOWER,), captured_at=10.0)

    assert eager.active(now=10.0 + EAGER_HIDE_SECONDS + 0.01) == ()


def test_row_that_stays_in_place_keeps_its_first_seen_time():
    eager = EagerHide()
    eager.observe((LOWER,), captured_at=10.0)
    jittered = Box(LOWER.left + 3, LOWER.top - 2, LOWER.right + 4, LOWER.bottom + 1)

    eager.observe((jittered,), captured_at=10.3)  # ugyanaz a sor a következő rögzítésen: nem indul újra az óra

    assert eager.active(now=10.0 + EAGER_HIDE_SECONDS + 0.01) == ()


def test_row_scrolled_to_another_place_counts_as_new():
    eager = EagerHide()
    eager.observe((LOWER,), captured_at=10.0)

    eager.observe((UPPER, LOWER), captured_at=10.5)  # a játék feljebb görgette, alá új sor jött

    assert eager.active(now=10.52) == (UPPER,)  # a régi helyen álló doboz órája már lejárt


def test_row_that_disappeared_is_forgotten():
    eager = EagerHide()
    eager.observe((LOWER,), captured_at=10.0)
    eager.observe((), captured_at=10.1)

    eager.observe((LOWER,), captured_at=10.2)

    assert eager.active(now=10.25) == (LOWER,)  # újra megjelent: megint új sor


# --- a feldolgozás által már takart sor: akkor is takarva marad, ha egy olvasás kihagyja --------

BOTH_ROWS = Box(UPPER.left, UPPER.top, UPPER.right, LOWER.bottom)
TWO_ROW_LINE = Shown("Szia, hogy vagy?", BOTH_ROWS, (ShownLine("Szia, hogy vagy?", BOTH_ROWS),))
FIRST_ROW_ONLY = Shown("Szia, hogy vagy?", UPPER, (ShownLine("Szia, hogy vagy?", UPPER),))


def test_row_the_pipeline_covered_stays_hidden_when_a_later_read_misses_it():
    eager = EagerHide()
    eager.observe((UPPER, LOWER), captured_at=10.0)
    eager.confirm(TWO_ROW_LINE, now=10.2)
    eager.observe((UPPER, LOWER), captured_at=10.6)  # a sor a helyén van, de az olvasás most csak az elsőt adta

    eager.confirm(FIRST_ROW_ONLY, now=10.7)

    assert with_eager_boxes(FIRST_ROW_ONLY, eager.active(now=10.7)).hidden == (LOWER,)


def test_held_row_is_shown_again_when_the_pipeline_keeps_leaving_it_out():
    eager = EagerHide()
    eager.observe((UPPER, LOWER), captured_at=10.0)
    eager.confirm(TWO_ROW_LINE, now=10.2)
    later = 10.2 + CONFIRMED_HOLD_SECONDS + 0.01
    eager.observe((UPPER, LOWER), captured_at=later - 0.05)

    eager.confirm(FIRST_ROW_ONLY, now=later)

    assert LOWER not in eager.active(now=later)


def test_row_the_pipeline_never_covered_is_not_held():
    eager = EagerHide()
    eager.observe((LOWER,), captured_at=10.0)

    eager.confirm(None, now=10.2)

    assert eager.active(now=10.0 + EAGER_HIDE_SECONDS + 0.01) == ()


def test_held_row_is_forgotten_when_it_leaves_the_screen():
    eager = EagerHide()
    eager.observe((UPPER, LOWER), captured_at=10.0)
    eager.confirm(TWO_ROW_LINE, now=10.2)
    eager.observe((UPPER,), captured_at=10.3)  # a második sor eltűnt

    eager.observe((UPPER, LOWER), captured_at=11.0)  # ugyanott új sor jelent meg: az már ismeretlen szöveg

    assert LOWER not in eager.active(now=11.0 + EAGER_HIDE_SECONDS + 0.01)


def test_eager_boxes_alone_become_a_cover_without_text():
    shown = with_eager_boxes(None, (LOWER,))

    assert shown == Shown("", LOWER, (), (LOWER,))


def test_nothing_to_hide_keeps_the_state_untouched():
    shown = Shown("Szia!", LOWER, (ShownLine("Szia!", LOWER),))

    assert with_eager_boxes(shown, ()) is shown
    assert with_eager_boxes(None, ()) is None


def test_row_already_translated_by_the_pipeline_is_not_hidden_twice():
    shown = Shown("Szia!", LOWER, (ShownLine("Szia!", LOWER),))
    detected = Box(LOWER.left - 3, LOWER.top + 2, LOWER.right + 2, LOWER.bottom)

    assert with_eager_boxes(shown, (detected,)) is shown


def test_new_row_above_a_translated_one_is_added_to_the_hidden_boxes():
    shown = Shown("Szia!", LOWER, (ShownLine("Szia!", LOWER),))

    merged = with_eager_boxes(shown, (UPPER, LOWER))

    assert merged.lines == shown.lines
    assert merged.hidden == (UPPER,)


def test_row_the_pipeline_already_hides_is_not_added_again():
    shown = Shown("", LOWER, (), (LOWER,))

    assert with_eager_boxes(shown, (LOWER,)) is shown


def test_wider_new_row_on_the_place_of_a_narrower_old_one_is_hidden():
    # A régi, rövid sor fordítása még kint van; a helyén megjelent egy hosszabb új angol sor
    shown = Shown("Szia!", LOWER, (ShownLine("Szia!", LOWER),))
    wider = Box(LOWER.left - 200, LOWER.top, LOWER.right + 200, LOWER.bottom)

    assert with_eager_boxes(shown, (wider,)).hidden == (wider,)


def test_hold_ends_when_the_pipeline_says_there_is_no_subtitle():
    eager = EagerHide()
    eager.observe((UPPER, LOWER), captured_at=10.0)
    eager.confirm(TWO_ROW_LINE, now=10.2)

    eager.confirm(None, now=10.5)  # a feldolgozás szándékosan elengedte: nem kimaradt olvasás

    assert eager.active(now=10.5) == ()


def test_row_the_pipeline_only_hid_is_not_held_after_it_lets_it_go():
    # Késő fordítás: a feldolgozás csak takarta a sort, aztán szándékosan megmutatja az angolt
    eager = EagerHide()
    eager.observe((LOWER,), captured_at=10.0)
    eager.confirm(Shown("", LOWER, (), (LOWER,)), now=11.5)
    beside = Shown("Szia!", UPPER, (ShownLine("Szia!", UPPER),))  # mellette egy másik sor lefordítva marad

    eager.confirm(beside, now=11.6)

    assert LOWER not in eager.active(now=11.6)


def test_block_text_covers_its_whole_anchor():
    block = Shown("⚠ Lefordítatlan szöveg", Box(700, 915, 1220, 994))

    assert with_eager_boxes(block, (UPPER, LOWER)) is block


def test_only_sure_rows_are_returned_in_frame_coordinates(monkeypatch):
    def fake_detect(band, band_left, frame_width, dialogue_bottom):
        assert (band_left, frame_width, dialogue_bottom) == (AREA.left, 1920.0, pytest.approx(1004 - AREA.top))
        return (SimpleNamespace(left=550, top=115, right=910, bottom=152, sure=True),
                SimpleNamespace(left=400, top=60, right=1000, bottom=100, sure=False))

    monkeypatch.setattr(eager_hide, "detect_rows", fake_detect)
    band = np.zeros((AREA.height, AREA.width, 3), np.uint8)

    rows = sure_rows(band, AREA, 1920.0, 1004)

    assert rows == (Box(780, 957, 1140, 994),)


def test_sure_rows_on_a_dark_band_finds_nothing():
    band = np.full((AREA.height, AREA.width, 3), 20, np.uint8)

    assert sure_rows(band, AREA, 1920.0, 1004) == ()


# --- híd a rögzítő szál és a megjelenítési szál között ---------------------------------------

def test_row_of_a_published_capture_is_hidden_at_once_and_only_for_a_short_time():
    cover = EagerCover()

    band = band_with(("We ain't lookin' for trouble, okay?", DIALOGUE_BASELINE))

    cover.publish(band, BAND_AREA, Region(), 10.0)

    backdrop, boxes = cover.view(None, now=10.01)
    assert backdrop.image is band and backdrop.area == BAND_AREA  # a takarás ugyanebből a sávból készül
    assert len(boxes) == 1
    assert cover.view(None, now=10.0 + EAGER_HIDE_SECONDS + 0.01)[1] == ()  # ugyanaz a rögzítés nem számít újra újnak
    assert cover.view(None, now=10.5)[0] is backdrop  # változatlan háttér: a takarás gyorstára újra felhasználhatja


def test_capture_without_text_hides_nothing():
    cover = EagerCover()

    assert cover.view(None, now=10.0) == (None, ())  # még nem jött sáv

    cover.publish(band_with(), BAND_AREA, Region(), 10.0)

    assert cover.view(None, now=10.01)[1] == ()


def test_no_rows_are_looked_for_when_the_english_text_is_not_covered():
    cover = EagerCover()
    band = band_with(("We ain't lookin' for trouble, okay?", DIALOGUE_BASELINE))

    cover.publish(band, BAND_AREA, None, 10.0)

    backdrop, boxes = cover.view(None, now=10.01)
    assert backdrop.image is band
    assert boxes == ()


def test_failing_row_detector_switches_the_instant_cover_off(monkeypatch):
    def broken(*_args, **_kwargs):
        raise ValueError("bad band")

    cover = EagerCover()
    cover.publish(band_with(("We ain't lookin' for trouble, okay?", DIALOGUE_BASELINE)), BAND_AREA, Region(), 10.0)
    monkeypatch.setattr(eager_hide, "detect_rows", broken)

    cover.publish(band_with(), BAND_AREA, Region(), 10.1)  # nem dob kivételt: ez csak gyorsítás

    assert cover.view(None, now=10.11)[1] == ()  # a korábban látott sor sem marad takarva
    monkeypatch.undo()
    cover.publish(band_with(("We ain't lookin' for trouble, okay?", DIALOGUE_BASELINE)), BAND_AREA, Region(), 10.2)
    assert cover.view(None, now=10.21)[1] == ()  # kikapcsolva marad


def test_row_far_from_the_frame_centre_is_never_hidden_before_it_is_read():
    # A sáv bárhol álló szövegét (whole_band_text) is csak olvasás után takarjuk: az azonnali takarás
    # továbbra is csak a kép közepén keres
    centred = band_with(("We ain't lookin' for trouble, okay?", DIALOGUE_BASELINE))
    off_centre = np.roll(centred, -500, axis=1)  # ugyanaz a sor 500 képponttal balra
    cover = EagerCover()

    cover.publish(off_centre, BAND_AREA, Region(), 10.0)

    assert cover.view(None, now=10.01)[1] == ()
