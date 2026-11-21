"""A beszélőszín útja a méréstől a kirajzolandó Shown-ig, mindkét olvasási úton."""

import time
from concurrent.futures import wait

from band_drawing import AREA, DIALOGUE_BASELINE, FRAME, UPPER_BASELINE
from gta_felirat.config import AppConfig
from gta_felirat.handoff import Inbox, RowRecognizer
from gta_felirat.ocr import Box, OcrLine, OcrResult, _speaker_colors
from gta_felirat.row_pipeline import RowPipeline
from gta_felirat.row_tracker import BandSnapshot, RowObs, RowText
from gta_felirat.speaker_color import PALETTE_DEADBAND, SpeakerPalette
from gta_felirat.subtitle_pipeline import SubtitlePipeline
from gta_felirat.translation_service import TranslationService
from gta_felirat.translator import CachedTranslator, TranslatorChoice
from test_speaker_color import TEAL, assert_close, dark_scene, line_box, write_line

KIRK = "Kirk: If you are hearing this,"
KIRK_COLOR = (130, 210, 200)
LOWER = Box(780, 957, 1140, 994)
WAIT_SECONDS = 5


class EchoBackend:
    def translate(self, text):
        return f"HU({text})"


def service():
    return TranslationService(TranslatorChoice("deepl", "hu", "", ""), CachedTranslator(EchoBackend()))


def config():
    return AppConfig(only_spoken_lines=False)


def test_row_recognizer_measures_the_speaker_colour_of_the_row_it_reads():
    band = dark_scene()
    text, boxes = write_line(band, [("Lucia:", TEAL), (" Stop talking, will you?", None)])
    row = RowObs(1, line_box(boxes), 0.1, 0.0, 2, sure=True)
    recognizer = RowRecognizer(lambda image, area, rows: [(text, 0.99)] * len(rows), Inbox(lambda: None), 0.5)

    (item,) = recognizer.recognise(BandSnapshot(band, AREA, FRAME, 0.0, (row,)))

    ((name, color),) = item.speaker_colors
    assert name == "Lucia"
    assert_close(color, TEAL)


def test_row_pipeline_hands_the_measured_colour_to_what_is_shown():
    pipeline = RowPipeline(config(), None, service(), wake=lambda: None)
    try:
        pipeline.receive(BandSnapshot(dark_scene(), AREA, FRAME, 1.0,
                                      (RowObs(1, LOWER, 0.1, 1.0, 2, sure=True),)), (), 1.0)
        pipeline.receive(None, [RowText(1, KIRK, 0.99, speaker_colors=(("Kirk", KIRK_COLOR),))], 1.04)
        if pipeline._translate_future is not None:  # None: a gyors fordítást már be is gyűjtötte
            wait([pipeline._translate_future], timeout=WAIT_SECONDS)
        pipeline.pump(1.3)
        shown = pipeline.shown
    finally:
        pipeline.shutdown()

    assert shown.speaker_colors == (("Kirk", KIRK_COLOR),)


class ColouredReader:
    """A régi út olvasója: egy megszólalás, a mért beszélőszínnel."""

    def read_band(self, band, area):
        return OcrResult(KIRK, LOWER, (OcrLine(KIRK, LOWER),), speaker_colors=(("Kirk", KIRK_COLOR),))


def test_subtitle_pipeline_learns_the_colour_from_the_full_read():
    pipeline = SubtitlePipeline(config(), ColouredReader(), service())
    area = pipeline.band_area(*FRAME)
    try:
        for now in (0.0, 1.0):
            pipeline.submit_band(lambda: None, area, FRAME, now)
            deadline = time.monotonic() + WAIT_SECONDS
            while pipeline._ocr_future is not None and time.monotonic() < deadline:
                pipeline.poll(now)
                time.sleep(0.01)
            time.sleep(0.05)
            pipeline.poll(now)
        shown = pipeline.shown
    finally:
        pipeline.shutdown()

    assert shown is not None and shown.speaker_colors == (("Kirk", KIRK_COLOR),)


def test_sentence_wrapped_over_two_rows_gets_the_colour_of_a_name_in_its_second_row():
    band = dark_scene()
    first, first_boxes = write_line(band, [("Vagrant:", (220, 70, 70)), (" Hey you, get away from", None)],
                                    baseline=UPPER_BASELINE)
    second, second_boxes = write_line(band, [(" my cart.", None), (" Lucia:", TEAL), (" Go away.", None)],
                                      baseline=DIALOGUE_BASELINE)
    rows = (line_box(first_boxes), line_box(second_boxes))
    line = OcrLine(first + second, line_box([*first_boxes, *second_boxes]), rows)

    colors = dict(_speaker_colors(band, AREA, [line]))

    assert set(colors) == {"Vagrant", "Lucia"}
    assert_close(colors["Lucia"], TEAL)


def test_palette_ignores_small_measurement_noise_but_follows_a_real_change():
    palette = SpeakerPalette(history=1)
    palette.learn((("Kirk", KIRK_COLOR),))
    noisy = tuple(channel + PALETTE_DEADBAND for channel in KIRK_COLOR)

    palette.learn((("Kirk", noisy),))
    after_noise = palette.for_text(KIRK)
    palette.learn((("Kirk", TEAL),))

    assert after_noise == (("Kirk", KIRK_COLOR),)
    assert palette.for_text(KIRK) == (("Kirk", TEAL),)
