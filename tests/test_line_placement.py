import time

import numpy as np

from gta_felirat.config import AppConfig
from gta_felirat.frosted_cover import Backdrop
from gta_felirat.line_placement import (PENDING_HIDE_SECONDS, PendingLines, TranslatedLine, build_shown,
                                        match_lines, pair_lines, place_lines)
from gta_felirat.ocr import Box, OcrLine, OcrResult, spoken_lines
from gta_felirat.subtitle_layer import Shown, ShownLine, SubtitleLayer
from gta_felirat.subtitle_pipeline import SubtitlePipeline
from gta_felirat.subtitle_tracker import normalize
from gta_felirat.translation_service import TranslationService
from gta_felirat.translator import CachedTranslator, TranslatorChoice, split_speakers

UPPER = Box(700, 915, 1220, 948)
LOWER = Box(780, 957, 1140, 994)
TROUBLE = "We ain't lookin' for trouble, okay?"
HOW = "How about you, how you doin'?"
TWO_ROWS = "Alright, you just turn around and go back to wherever the hell you came from."
TWO_ROWS_FIRST = "Alright, you just turn around and go back to wherever the hell"


# --- OCR: megszólalásonként külön sor ------------------------------------------

def test_each_finished_sentence_row_is_its_own_spoken_line():
    lines = spoken_lines([(HOW, LOWER), (TROUBLE, UPPER)])

    assert lines == (OcrLine(TROUBLE, UPPER, (UPPER,)), OcrLine(HOW, LOWER, (LOWER,)))


def test_unfinished_sentence_continues_on_the_next_row():
    lines = spoken_lines([("Got so bad once, I had bounties on me", UPPER), ("in three different states.", LOWER)])

    assert [line.text for line in lines] == ["Got so bad once, I had bounties on me in three different states."]
    assert lines[0].box == Box(700, 915, 1220, 994)
    assert lines[0].rows == (UPPER, LOWER)  # a két képernyősor külön is megvan (soronkénti újraolvasáshoz)


def test_normalize_keeps_line_breaks_between_spoken_lines():
    assert normalize(f"  {TROUBLE} \n {HOW}  ") == f"{TROUBLE}\n{HOW}"


def test_translator_splits_text_at_line_breaks():
    assert split_speakers(f"{TROUBLE}\n{HOW}") == [TROUBLE, HOW]


# --- elhelyezés: a fordítás követi az angol sort --------------------------------

def test_translation_follows_its_english_line_when_the_game_scrolls_it_up():
    translated = (TranslatedLine(TROUBLE, "Nem akarunk bajt, oké?"),)
    live = (OcrLine(TROUBLE, UPPER), OcrLine(HOW, LOWER))  # az új sor alul jelent meg, a régi feljebb csúszott

    shown = place_lines(translated, live, previous=None)

    assert shown.lines == (ShownLine("Nem akarunk bajt, oké?", UPPER),)


def test_line_that_scrolled_out_is_no_longer_shown():
    translated = (TranslatedLine(TROUBLE, "Nem akarunk bajt, oké?"), TranslatedLine(HOW, "Na és te, hogy vagy?"))

    shown = place_lines(translated, (OcrLine(HOW, LOWER),), previous=None)

    assert shown.text == "Na és te, hogy vagy?"
    assert shown.lines == (ShownLine("Na és te, hogy vagy?", LOWER),)


def test_nothing_is_shown_when_no_translated_line_is_on_screen():
    translated = (TranslatedLine(TROUBLE, "Nem akarunk bajt, oké?"),)

    assert place_lines(translated, (OcrLine("Hello? Anybody home?", LOWER),), previous=None) is None


def test_slightly_misread_line_still_matches_its_translation():
    translated = (TranslatedLine(TROUBLE, "Nem akarunk bajt, oké?"),)

    shown = place_lines(translated, (OcrLine("We aint lookin for trouble, okay", LOWER),), previous=None)

    assert shown is not None


def test_small_ocr_jitter_does_not_move_the_translation():
    translated = (TranslatedLine(TROUBLE, "Nem akarunk bajt, oké?"),)
    before = place_lines(translated, (OcrLine(TROUBLE, LOWER),), previous=None)
    jittered = Box(LOWER.left + 2, LOWER.top - 1, LOWER.right + 1, LOWER.bottom)

    after = place_lines(translated, (OcrLine(TROUBLE, jittered),), previous=before)

    assert after is before


def test_reading_the_same_line_a_little_narrower_does_not_move_the_translation():
    # Valós eset: világos háttéren a sor széle olvasásonként 5–10 képponttal odébb került, a magyar szöveg ugrált
    translated = (TranslatedLine(TROUBLE, "Nem akarunk bajt, oké?"),)
    before = place_lines(translated, (OcrLine(TROUBLE, LOWER),), previous=None)
    narrower = Box(LOWER.left + 9, LOWER.top + 1, LOWER.right - 2, LOWER.bottom)

    after = place_lines(translated, (OcrLine(TROUBLE, narrower),), previous=before)

    assert after is before


def test_reading_the_same_line_a_little_wider_widens_its_place_once():
    translated = (TranslatedLine(TROUBLE, "Nem akarunk bajt, oké?"),)
    before = place_lines(translated, (OcrLine(TROUBLE, LOWER),), previous=None)
    wider = Box(LOWER.left, LOWER.top + 1, LOWER.right + 10, LOWER.bottom)

    grown = place_lines(translated, (OcrLine(TROUBLE, wider),), previous=before)
    again = place_lines(translated, (OcrLine(TROUBLE, LOWER),), previous=grown)

    assert grown.lines[0].anchor == Box(LOWER.left, LOWER.top, LOWER.right + 10, LOWER.bottom)  # az angol ne lógjon ki
    assert again is grown  # a szűkebb olvasat már nem mozdítja vissza


def test_two_row_line_keeps_its_place_when_a_read_misses_its_second_row():
    # Valós eset: a kétsoros mondat második sorát egy-egy olvasás kihagyta, a fordítás fel-le ugrált
    both_rows = Box(UPPER.left, UPPER.top, UPPER.right, LOWER.bottom)
    translated = (TranslatedLine(TWO_ROWS, "Na jó, akkor fordulj meg, és menj vissza oda, ahonnan jöttél."),)
    before = place_lines(translated, (OcrLine(TWO_ROWS, both_rows, (UPPER, LOWER)),), previous=None)

    after = place_lines(translated, (OcrLine(TWO_ROWS_FIRST, UPPER),), previous=before)

    assert after is before


def test_line_gets_a_smaller_place_when_another_line_took_over_its_lower_row():
    both_rows = Box(UPPER.left, UPPER.top, UPPER.right, LOWER.bottom)
    line = TranslatedLine(TWO_ROWS, "Na jó, akkor fordulj meg, és menj vissza oda, ahonnan jöttél.")
    before = build_shown([(line, OcrLine(TWO_ROWS, both_rows, (UPPER, LOWER)))], [], previous=None)

    after = build_shown([(line, OcrLine(TWO_ROWS_FIRST, UPPER))], [LOWER], previous=before)

    assert after.lines[0].anchor == UPPER
    assert after.hidden == (LOWER,)


def test_pairing_needs_the_same_number_of_lines():
    assert pair_lines("A.\nB.", "X.\nY.") == (TranslatedLine("A.", "X."), TranslatedLine("B.", "Y."))
    assert pair_lines("A.\nB.", "Csak egy sor.") == ()


# --- réteg: minden fordított sor a saját angol sora helyén ----------------------

def game_band_with_two_lines():
    """Sötét játékkép két angol sorral: a betűket 4 képpont vastag fehér vonások adják."""
    band = np.full((1080, 1920, 3), 40, np.uint8)
    for box in (UPPER, LOWER):
        for x in range(box.left, box.right - 4, 8):
            band[box.top:box.bottom, x:x + 4] = 255
    return band


def test_each_translated_line_is_drawn_over_its_own_english_line():
    layer = SubtitleLayer(AppConfig(hide_original=True, panel_opacity=0, font_size_ratio=0.024),
                          on_change=lambda _updates: None)
    shown = Shown("Nem akarunk bajt, oké?\nNa és te, hogy vagy?", Box(700, 915, 1220, 994),
                  (ShownLine("Nem akarunk bajt, oké?", UPPER), ShownLine("Na és te, hogy vagy?", LOWER)))
    canvas = np.zeros((1080, 1920, 4), np.uint8)

    layer.draw(canvas, shown, True, backdrop=Backdrop(game_band_with_two_lines(), Box(0, 0, 1920, 1080)))

    _left, top, _width, height = layer.last_rect
    assert top <= UPPER.top + 10 and top + height >= LOWER.bottom - 10


def test_untranslated_neighbour_line_stays_visible():
    layer = SubtitleLayer(AppConfig(hide_original=True, panel_opacity=0, font_size_ratio=0.024),
                          on_change=lambda _updates: None)
    shown = Shown("Nem akarunk bajt, oké?", UPPER, (ShownLine("Nem akarunk bajt, oké?", UPPER),))
    canvas = np.zeros((1080, 1920, 4), np.uint8)

    output = layer.draw(canvas, shown, True, backdrop=Backdrop(game_band_with_two_lines(), Box(0, 0, 1920, 1080)))

    middle_row = (LOWER.top + LOWER.bottom) // 2
    assert output[middle_row, LOWER.left + 5:LOWER.right - 5, 3].max() == 0  # az alsó angol sort nem takarjuk


# --- teljes folyamat: görgő felirat ---------------------------------------------

class EchoBackend:
    def __init__(self):
        self.calls = []

    def translate(self, text):
        self.calls.append(text)
        return f"HU({text})"


class ScriptedReader:
    """Minden olvasásnál a forgatókönyv következő állapotát adja (az utolsó ismétlődik)."""

    def __init__(self, states):
        self._states = list(states)

    def read_band(self, band, area):
        lines = self._states.pop(0) if len(self._states) > 1 else self._states[0]
        boxes = [line.box for line in lines]
        box = Box(min(b.left for b in boxes), min(b.top for b in boxes),
                  max(b.right for b in boxes), max(b.bottom for b in boxes))
        return OcrResult("\n".join(line.text for line in lines), box, tuple(lines))


def run_reads(pipeline, count, start=0.0):
    area = pipeline.band_area(1920, 1080)
    for step in range(count):
        now = start + step
        pipeline.submit_band(lambda: None, area, (1920, 1080), now)
        deadline = time.monotonic() + 2
        while pipeline._ocr_future is not None and time.monotonic() < deadline:
            pipeline.poll(now)
            time.sleep(0.01)
        time.sleep(0.05)
        pipeline.poll(now)


def test_pipeline_moves_the_translation_up_with_its_line_and_translates_only_the_new_one():
    backend = EchoBackend()
    one_line = [OcrLine(TROUBLE, LOWER)]
    two_lines = [OcrLine(TROUBLE, UPPER), OcrLine(HOW, LOWER)]
    reader = ScriptedReader([one_line, one_line, two_lines, two_lines, two_lines])
    service = TranslationService(TranslatorChoice("deepl", "hu", "", ""), CachedTranslator(backend))
    pipeline = SubtitlePipeline(AppConfig(only_spoken_lines=False), reader, service)
    try:
        run_reads(pipeline, 2)
        first = pipeline.shown
        run_reads(pipeline, 1, start=2.0)
        scrolled = pipeline.shown
        run_reads(pipeline, 2, start=3.0)
        both = pipeline.shown
    finally:
        pipeline.shutdown()

    assert first.lines == (ShownLine(f"HU({TROUBLE})", LOWER),)
    assert scrolled.lines[0] == ShownLine(f"HU({TROUBLE})", UPPER)  # azonnal követi a feljebb csúszott sort
    assert both.lines == (ShownLine(f"HU({TROUBLE})", UPPER), ShownLine(f"HU({HOW})", LOWER))
    assert sorted(backend.calls) == sorted([TROUBLE, HOW])  # a régi sort nem fordítja újra


# --- a még le nem fordított angol sor ne villanjon fel --------------------------

def test_untranslated_line_is_hidden_from_its_first_read():
    pending = PendingLines()

    assert pending.boxes_to_hide([OcrLine(HOW, LOWER)], now=10.0) == [LOWER]


def test_untranslated_line_shows_again_when_the_translation_is_late():
    pending = PendingLines()
    pending.boxes_to_hide([OcrLine(HOW, LOWER)], now=10.0)

    assert pending.boxes_to_hide([OcrLine(HOW, LOWER)], now=10.0 + PENDING_HIDE_SECONDS - 0.1) == [LOWER]
    assert pending.boxes_to_hide([OcrLine(HOW, LOWER)], now=10.0 + PENDING_HIDE_SECONDS + 0.1) == []


def test_pending_clock_follows_the_line_when_it_scrolls_up():
    pending = PendingLines()
    pending.boxes_to_hide([OcrLine(HOW, LOWER)], now=10.0)

    assert pending.boxes_to_hide([OcrLine(HOW, UPPER)], now=10.0 + PENDING_HIDE_SECONDS + 0.1) == []


def test_new_line_under_a_translated_one_is_hidden_while_the_old_one_stays_translated():
    translated = (TranslatedLine(TROUBLE, "Nem akarunk bajt, oké?"),)
    pairs, untranslated = match_lines(translated, (OcrLine(TROUBLE, UPPER), OcrLine(HOW, LOWER)))

    shown = build_shown(pairs, [line.box for line in untranslated], previous=None)

    assert shown.lines == (ShownLine("Nem akarunk bajt, oké?", UPPER),)
    assert shown.hidden == (LOWER,)


def test_only_hidden_lines_means_cover_without_any_text():
    shown = build_shown([], [LOWER], previous=None)

    assert shown.text == "" and shown.lines == () and shown.hidden == (LOWER,)


def test_layer_erases_a_hidden_line_and_draws_no_text():
    layer = SubtitleLayer(AppConfig(hide_original=True, panel_opacity=0, font_size_ratio=0.024),
                          on_change=lambda _updates: None)
    canvas = np.zeros((1080, 1920, 4), np.uint8)

    output = layer.draw(canvas, Shown("", LOWER, (), (LOWER,)), True,
                        backdrop=Backdrop(game_band_with_two_lines(), Box(0, 0, 1920, 1080)))

    middle_row = (LOWER.top + LOWER.bottom) // 2
    assert output[middle_row, LOWER.left + 20:LOWER.right - 20, 3].min() == 255  # az angol betűk eltűnnek
    upper_row = (UPPER.top + UPPER.bottom) // 2
    assert output[upper_row, UPPER.left + 20:UPPER.right - 20, 3].max() == 0  # a másik sorhoz nem nyúlunk
    assert layer.last_rect is None  # szöveget nem rajzoltunk


class SlowBackend:
    def translate(self, text):
        time.sleep(0.5)
        return f"HU({text})"


def test_pipeline_hides_the_english_line_before_its_translation_arrives():
    reader = ScriptedReader([[OcrLine(TROUBLE, LOWER)]])
    service = TranslationService(TranslatorChoice("deepl", "hu", "", ""), CachedTranslator(SlowBackend()))
    pipeline = SubtitlePipeline(AppConfig(only_spoken_lines=False), reader, service)
    try:
        run_reads(pipeline, 1)
        after_first_read = pipeline.shown
    finally:
        pipeline.shutdown()

    assert after_first_read is not None
    assert after_first_read.hidden == (LOWER,) and after_first_read.text == ""


def test_translation_of_a_two_row_sentence_may_wrap_instead_of_running_off_screen():
    layer = SubtitleLayer(AppConfig(hide_original=True, panel_opacity=0, font_size_ratio=0.024),
                          on_change=lambda _updates: None)
    two_rows = Box(520, 915, 1400, 994)
    long_text = ("Aktívan a nyomában vagyok, ezért korlátoznom kell a sugárzásnak való kitettségemet, "
                 "és a legtöbb estén a völgyben, a kalandpark közelében leszek megtalálható, ha keresel.")
    canvas = np.zeros((1080, 1920, 4), np.uint8)

    layer.draw(canvas, Shown(long_text, two_rows, (ShownLine(long_text, two_rows),)), True)

    _left, _top, width, height = layer.last_rect
    assert width <= 1920 * 0.62 + 40
    assert height > 60  # két sorba tört
