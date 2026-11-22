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
    # A színes beszélőnév kimaradt az olvasásból, csak az utána álló kettőspont maradt meg
    (": Hey, you got Lucia's phone.", "Hey, you got Lucia's phone."),
    (":Hey.", "Hey."),
    ("; Get in.", "Get in."),
])
def test_leading_fragment_from_other_text_is_dropped(raw, cleaned):
    assert clean_ocr_text(raw) == cleaned


@pytest.mark.parametrize("raw, cleaned", [
    # Világos háttér előtt a név utáni kettőspontot pontosvesszőnek olvassa az OCR
    ("Cal; Alright, just don't die on me for real, man. Promise.",
     "Cal: Alright, just don't die on me for real, man. Promise."),
    ("Staff Member; 'Sup?", "Staff Member: 'Sup?"),
    ("Okay, fine; let's go.", "Okay, fine; let's go."),
    ("We go now; you stay here.", "We go now; you stay here."),
])
def test_semicolon_read_after_a_speaker_name_is_a_colon(raw, cleaned):
    assert clean_ocr_text(raw) == cleaned


@pytest.mark.parametrize("raw, cleaned", [
    # A felirat sötét háttérdobozának szélét a mondat végére tapadt jelnek olvassa az OCR
    ("Jason: This your spot, huh?3", "Jason: This your spot, huh?"),
    ("Stranger: I'm busy, you heard?湖", "Stranger: I'm busy, you heard?"),
    ("Stranger: Soft ass.1", "Stranger: Soft ass."),
    ("Go to Unit 3.", "Go to Unit 3."),
    ("Jason: Hey!", "Jason: Hey!"),
])
def test_glyph_glued_after_the_sentence_end_is_dropped(raw, cleaned):
    assert clean_ocr_text(raw) == cleaned


@pytest.mark.parametrize("raw, cleaned", [
    # Az OCR betűhibái, amelyek félrefordítást okoznak („On my od” → „Az én véleményem szerint”)
    ("Raul: On my od, you really fucked us now!", "Raul: On my god, you really fucked us now!"),
    ("Lucia: Oh my od.", "Lucia: Oh my god."),
    ("Raul: Movei Move it!", "Raul: Move! Move it!"),
    ("Lucia: l'm gonna send you the combo.", "Lucia: I'm gonna send you the combo."),
    ("Jason: l'll be back.", "Jason: I'll be back."),
    ("Raul: Go, go, go!", "Raul: Go, go, go!"),
    ("Jason: Hi, Hi there.", "Jason: Hi, Hi there."),
    ("Cop: Taxi drivers know the way.", "Cop: Taxi drivers know the way."),
    ("Jason: That's the OD unit.", "Jason: That's the OD unit."),
    ("Raul: Pull the lever.", "Raul: Pull the lever."),
    ("Dre'Quan: When Isay all that?", "Dre'Quan: When I say all that?"),
    ("Jason: Iwant out.", "Jason: I want out."),
    ("Jason: It is what it is. Into the car, Inside.", "Jason: It is what it is. Into the car, Inside."),
    ("Lucia: Idol, Iceland, Ideas.", "Lucia: Idol, Iceland, Ideas."),
    ("Local: Hey, you can spare some change? l.", "Local: Hey, you can spare some change?"),
    ("Local: Something to eat? l...", "Local: Something to eat?"),
    ("Jason: Who? I.", "Jason: Who? I."),
])
def test_ocr_letter_misreads_are_repaired(raw, cleaned):
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
