import numpy as np

from gta_felirat.config import AppConfig
from gta_felirat.ocr import Box
from gta_felirat.overlay import render_subtitle
from gta_felirat.subtitle_layer import Shown, SubtitleLayer
from gta_felirat.translator import CachedTranslator, split_speakers

FONT = AppConfig().font_path


class EchoBackend:
    def __init__(self):
        self.calls = []

    def translate(self, text):
        self.calls.append(text)
        return f"HU({text})"


# --- beszélőnként külön sor ---------------------------------------------------

def test_two_speakers_are_split_into_separate_lines():
    assert split_speakers("Vagrant: Take a breath now. Lucia: Stop talking.") == [
        "Vagrant: Take a breath now.", "Lucia: Stop talking."]


def test_staff_member_with_two_word_name_is_a_speaker_too():
    assert split_speakers("Locals: I know you? Staff Member: You want something?") == [
        "Locals: I know you?", "Staff Member: You want something?"]


def test_single_speaker_line_stays_whole():
    assert split_speakers("Jason: Me and Lucia. We go.") == ["Jason: Me and Lucia. We go."]


def test_each_speaker_gets_its_own_translated_line_with_name_kept():
    translator = CachedTranslator(EchoBackend())

    result = translator.translate("Vagrant: Take a breath now. Lucia: Stop talking.")

    assert result == "Vagrant: HU(Take a breath now.)\nLucia: HU(Stop talking.)"


def test_language_note_is_translated_but_a_speaker_name_is_kept():
    translator = CachedTranslator(EchoBackend())

    assert translator.translate("In Spanish: Eyes on the road!") == "HU(In Spanish): HU(Eyes on the road!)"
    assert translator.translate("Jason: Eyes on the road!") == "Jason: HU(Eyes on the road!)"


def test_language_note_is_sent_to_the_service_only_once():
    backend = EchoBackend()
    translator = CachedTranslator(backend)

    translator.translate("In Spanish: Go!")
    translator.translate("In Spanish: Stop!")

    assert backend.calls == ["Go!", "In Spanish", "Stop!"]


def test_only_the_new_speaker_line_goes_to_the_service():
    backend = EchoBackend()
    translator = CachedTranslator(backend)
    translator.translate("Vagrant: Take a breath now.")

    translator.translate("Vagrant: Take a breath now. Lucia: Stop talking.")

    assert backend.calls == ["Take a breath now.", "Stop talking."]


def test_speaker_lines_are_drawn_on_separate_rows():
    one = render_subtitle("Vagrant: Szia.", FONT, 30, 1200)
    two = render_subtitle("Vagrant: Szia.\nLucia: Hallgass.", FONT, 30, 1200)

    assert two.height > one.height


# --- a fordítás az angol felirat alá kerül -------------------------------------

def draw_rect(anchor, frame_height=1080, text="Vagrant: Szia.\nLucia: Hallgass."):
    layer = SubtitleLayer(AppConfig(font_size_ratio=0.026), on_change=lambda _u: None)
    frame = np.zeros((frame_height, 1920, 3), np.uint8)
    layer.draw(frame, Shown(text, anchor), hide_original=False)
    return layer.last_rect


def test_translation_goes_right_below_the_english_text():
    english = Box(700, 800, 1200, 870)  # két beszélő angol sora

    left, top, width, _height = draw_rect(english)

    assert top >= english.bottom
    assert abs((left + width // 2) - 950) <= 2


def test_translation_goes_above_when_there_is_no_room_below():
    english = Box(700, 980, 1200, 1060)

    _left, top, _width, height = draw_rect(english)

    assert top + height <= english.top
