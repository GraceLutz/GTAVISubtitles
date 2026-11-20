import pytest

from gta_felirat.ocr import clean_ocr_text, looks_like_subtitle, strip_stray_glyphs


@pytest.mark.parametrize("text", [
    "OD",
    "k Mall",
    "E",
    "12:45",
    "$ 250",
    "CLASSIC LOW TOPS",
    "SALE!",
    "",
])
def test_short_fragments_without_sentence_shape_are_noise(text):
    assert not looks_like_subtitle(text)


@pytest.mark.parametrize("text", [
    "Local: Fuck's wrong with you?",
    "Locals: Get fucked!",
    "We need to get out of here, now.",
    "No!",
    "What?",
    "Yeah...",
    "Lucia: Go",
    "Hold on a second",
])
def test_real_subtitle_lines_are_kept(text):
    assert looks_like_subtitle(text)


@pytest.mark.parametrize("raw, cleaned", [
    ("Lucia: I got you now. 大", "Lucia: I got you now."),
    ("大 Get in the car!", "Get in the car!"),
    ("Jason: Café's closed.", "Jason: Café's closed."),
])
def test_stray_non_latin_glyphs_are_removed(raw, cleaned):
    assert strip_stray_glyphs(raw) == cleaned


@pytest.mark.parametrize("raw, cleaned", [
    ("oes Staff Member: 'Sup?", "Staff Member: 'Sup?"),
    ("k Mall", "Mall"),
    ("so What now?", "What now?"),
    ("and then what?", "and then what?"),
    ("Jason: Go.", "Jason: Go."),
    ("X Cop: Alright, you're good to go.", "Cop: Alright, you're good to go."),
    ("I need a ride.", "I need a ride."),
])
def test_leading_fragment_from_other_text_is_dropped(raw, cleaned):
    assert clean_ocr_text(raw) == cleaned


@pytest.mark.parametrize("text", [
    "Játék indítása",
    "Játék indítása 12 órája",
    "0 Y 2 0 eák Barátok, akik játszanak 2% Egy barátod sem játszik ezzel a já.. enységek",
    "Beállítások",
])
def test_console_menu_text_in_other_language_is_not_a_subtitle(text):
    assert not looks_like_subtitle(text)


@pytest.mark.parametrize("text", [
    "Jason: Café's closed.",
    "Lucia: Meet me at the Señor Taco place, okay?",
])
def test_english_lines_with_an_occasional_accent_are_kept(text):
    assert looks_like_subtitle(text)


@pytest.mark.parametrize("text", [
    "Focus +10% SMALL SALTW",
    "Strength -10% Stamina bonus",
    "+$250 Cash received from mission",
    "Shooting skill +1 LEVEL UP",
    "Focus SMALL SALTW BONUS",
])
def test_hud_stat_notifications_are_noise(text):
    assert not looks_like_subtitle(text)


@pytest.mark.parametrize("text", [
    "I want 50% of the cut.",
    "Get the FIB files out of here",
    "We need 10 more minutes, okay?",
    "Lucia: Give me +10% or walk.",
    "Jason: The LSPD is coming!",
])
def test_dialogue_with_numbers_or_acronyms_is_kept(text):
    assert looks_like_subtitle(text)
