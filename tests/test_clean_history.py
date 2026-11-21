from concurrent.futures import Future

import numpy as np
import pytest

from gta_felirat import clean_output
from gta_felirat.clean_history import FrameBuffer, FrameRow, ShownHistory
from gta_felirat.clean_output import CapturedFrame, CleanOutput, draw_on_frame
from gta_felirat.config import AppConfig, ConfigError
from gta_felirat.frosted_cover import Backdrop
from gta_felirat.layer_plan import Shown, ShownLine
from gta_felirat.ocr import Box
from gta_felirat.subtitle_layer import SubtitleLayer

LOWER = Box(780, 957, 1140, 994)
UPPER = Box(700, 915, 1220, 948)
FIRST, SECOND = "Cut it, cut it.", "So what I was doin'?"
COVER_ONLY = Shown("", LOWER, (), (LOWER,))
FIRST_HU = Shown("Vágd le, vágd le!", LOWER, (ShownLine("Vágd le, vágd le!", LOWER),))
SECOND_HU = Shown("Szóval mit csináltam?", LOWER, (ShownLine("Szóval mit csináltam?", LOWER),))
# A játék feljebb görgette az első sort (új beszélő jött alá): a fordítása már fent áll, alatta takarás
FIRST_SCROLLED = Shown("Vágd le, vágd le!", UPPER, (ShownLine("Vágd le, vágd le!", UPPER),), (LOWER,))


def history_of(*entries):
    history = ShownHistory(keep_seconds=5.0)
    for at, shown, source in entries:
        history.record(at, shown, source)
    return history


def test_frame_gets_the_finished_translation_not_the_cover_shown_at_that_moment():
    history = history_of((10.0, COVER_ONLY, FIRST), (10.5, FIRST_HU, FIRST))

    assert history.for_frame(10.0, 11.0, [FrameRow(LOWER)]) == FIRST_HU


def test_frame_before_the_scroll_keeps_the_translation_where_the_line_stood_on_that_frame():
    # A felirat utolsó állapota már a görgetett helyet mutatja: a korábbi képkockán a sor még lent állt
    history = history_of((10.0, FIRST_HU, FIRST), (12.0, FIRST_SCROLLED, FIRST))

    assert history.for_frame(11.5, 12.5, [FrameRow(LOWER)]) == FIRST_HU


def test_next_subtitle_in_the_same_place_does_not_replace_the_current_one():
    history = history_of((10.0, FIRST_HU, FIRST), (11.0, SECOND_HU, SECOND))

    assert history.for_frame(10.2, 11.2, [FrameRow(LOWER)]) == FIRST_HU


def test_row_that_no_state_covers_gets_a_cover_instead_of_showing_english():
    history = history_of((10.0, FIRST_HU, FIRST))

    shown = history.for_frame(10.0, 11.0, [FrameRow(LOWER), FrameRow(UPPER)])

    assert shown.lines == FIRST_HU.lines and UPPER in shown.hidden


def test_row_is_covered_even_without_any_recorded_state():
    shown = ShownHistory(keep_seconds=5.0).for_frame(10.0, 11.0, [FrameRow(UPPER)])

    assert shown.text == "" and shown.hidden == (UPPER,)


def test_frame_without_found_rows_gets_the_state_in_force_after_recognition():
    history = history_of((10.0, FIRST_HU, FIRST), (11.0, None, ""))

    assert history.for_frame(10.0, 11.0, []) == FIRST_HU
    assert history.for_frame(11.0, 12.0, []) is None


def test_old_states_are_dropped_but_the_one_still_in_force_is_kept():
    history = ShownHistory(keep_seconds=1.0)
    history.record(10.0, FIRST_HU, FIRST)
    history.record(20.0, SECOND_HU, SECOND)

    assert history.for_frame(15.0, 16.0, None) == FIRST_HU  # 15-kor még a 10-kor beállt állapot érvényes
    history.record(25.0, None, "")
    assert history.for_frame(15.0, 16.0, None) is None  # a 20 előtti állapot már kiesett


def test_frame_buffer_gives_the_latest_frame_not_newer_than_asked():
    buffer = FrameBuffer(keep_seconds=1.0)
    frames = [np.full((2, 2, 3), value, np.uint8) for value in (1, 2, 3)]
    for at, frame in zip((10.0, 10.1, 10.2), frames):
        buffer.push(at, frame)

    at, frame = buffer.at_or_before(10.15)

    assert at == 10.1 and frame[0, 0, 0] == 2
    assert buffer.at_or_before(9.9) is None


def test_frame_buffer_keeps_only_the_delay_window():
    buffer = FrameBuffer(keep_seconds=1.0)
    for step in range(30):
        buffer.push(10.0 + step * 0.1, np.zeros((1, 1, 3), np.uint8))

    assert buffer.at_or_before(10.5) is None  # 12,9-nél az 1 mp-nél régebbiek már kiestek
    assert buffer.at_or_before(11.95)[0] == 11.9


# --- a tiszta kép ablaka és beállítása ------------------------------------------------------------

FRAME_SIZE = (1920, 1080)


@pytest.mark.parametrize("delay", [0.1, 10.0])
def test_clean_output_delay_outside_the_limits_is_rejected(delay):
    with pytest.raises(ConfigError):
        AppConfig(clean_output_delay_seconds=delay).validate()


def test_clean_output_is_off_by_default():
    assert AppConfig().clean_output is False


def test_delayed_frame_is_drawn_with_the_settled_translation(monkeypatch):
    shown_images = []
    monkeypatch.setattr(clean_output.cv2, "imshow", lambda _title, image: shown_images.append(image))
    config = AppConfig(only_spoken_lines=False)
    output = CleanOutput(hwnd=0, config=config)
    english = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), np.uint8)
    english[LOWER.top:LOWER.bottom, LOWER.left:LOWER.right] = 255  # az angol felirat helye világos
    frames = FrameBuffer(keep_seconds=2.0)
    frames.push(10.0, CapturedFrame(english))
    output.history.record(10.0, COVER_ONLY, FIRST)
    output.history.record(10.3, FIRST_HU, FIRST)  # a fordítás később érkezett

    output._show("clean", SubtitleLayer(config, on_change=lambda _c: None), frames, now=11.05)

    assert len(shown_images) == 1
    drawn = shown_images[0]
    assert drawn.shape == english.shape
    assert not np.array_equal(drawn[LOWER.top:LOWER.bottom, LOWER.left:LOWER.right],
                              english[LOWER.top:LOWER.bottom, LOWER.left:LOWER.right])


def test_nothing_is_shown_until_a_frame_old_enough_exists(monkeypatch):
    shown_images = []
    monkeypatch.setattr(clean_output.cv2, "imshow", lambda _title, image: shown_images.append(image))
    config = AppConfig()
    output = CleanOutput(hwnd=0, config=config)
    frames = FrameBuffer(keep_seconds=2.0)
    frames.push(10.0, CapturedFrame(np.zeros((4, 4, 3), np.uint8)))

    output._show("clean", SubtitleLayer(config, on_change=lambda _c: None), frames, now=10.5)

    assert shown_images == []


def test_drawing_on_the_crop_gives_the_same_picture_as_drawing_on_the_whole_frame():
    config = AppConfig()
    layer = SubtitleLayer(config, on_change=lambda _c: None)
    frame = np.random.default_rng(3).integers(0, 255, (FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
    shown = Shown("Na jó, uram!\nMi a fene bajod van?", Box(700, 915, 1220, 994),
                  (ShownLine("Na jó, uram!", UPPER), ShownLine("Mi a fene bajod van?", LOWER)), (Box(760, 1000, 1160, 1030),))
    backdrop = Backdrop(frame, Box(0, 0, *FRAME_SIZE))
    plan = layer.plan(FRAME_SIZE, shown, True, backdrop=backdrop)

    assert np.array_equal(draw_on_frame(frame, plan), layer.draw(frame, shown, True, backdrop=backdrop))


def test_translation_shown_as_one_block_counts_as_covering_its_rows():
    block = Shown("Vágd le, vágd le!", LOWER)
    history = history_of((10.0, COVER_ONLY, FIRST), (10.5, block, FIRST))

    assert history.for_frame(10.0, 11.0, [FrameRow(LOWER)]) == block


def test_rows_found_in_the_background_are_covered_even_without_any_state(monkeypatch):
    shown_images = []
    monkeypatch.setattr(clean_output.cv2, "imshow", lambda _title, image: shown_images.append(image))
    config = AppConfig()
    output = CleanOutput(hwnd=0, config=config)
    english = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), np.uint8)
    english[UPPER.top:UPPER.bottom, UPPER.left:UPPER.right] = 255
    found = Future()
    found.set_result([FrameRow(UPPER)])
    frames = FrameBuffer(keep_seconds=2.0)
    frames.push(10.0, CapturedFrame(english, found))

    output._show("clean", SubtitleLayer(config, on_change=lambda _c: None), frames, now=11.05)

    assert not np.array_equal(shown_images[0][UPPER.top:UPPER.bottom, UPPER.left:UPPER.right],
                              english[UPPER.top:UPPER.bottom, UPPER.left:UPPER.right])


def test_unsure_row_translated_by_a_nearby_state_is_covered():
    # A sorkereső nem biztos a soron, de közben a pipeline lefordította: feliratsor, nem maradhat angolul
    history = history_of((10.0, None, ""), (10.6, FIRST_HU, FIRST))

    shown = history.for_frame(10.0, 11.0, [FrameRow(UPPER, sure=True), FrameRow(LOWER, sure=False)])

    assert shown.lines == FIRST_HU.lines


def test_unsure_row_that_no_state_knows_is_left_alone():
    plate = Box(880, 780, 980, 820)  # pl. egy rendszámtábla a sávban

    shown = ShownHistory(keep_seconds=5.0).for_frame(10.0, 11.0, [FrameRow(plate, sure=False)])

    assert shown is None
