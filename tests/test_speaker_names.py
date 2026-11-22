"""A félreolvasott beszélőnevek javítása és az új nevek megtanulása."""

import pytest

from gta_felirat.speaker_names import LEARN_AFTER_READS, LEARNED_NAMES_FILE, SpeakerBook


@pytest.mark.parametrize("misread, fixed", [
    # Valós hibák a felvételek naplójából
    ("Raut: Go!", "Raul: Go!"),
    ("L'ucia: Rox, let's go...", "Lucia: Rox, let's go..."),
    ("Dre Quan: So, what's this address?", "Dre'Quan: So, what's this address?"),
    ("Toufists: So. this is it, right?", "Tourists: So. this is it, right?"),
    ("GTourist: Oh, fuck!", "Tourist: Oh, fuck!"),
    ("Bae-Lauxe: Damn.", "Bae-Luxe: Damn."),
    ("Partygcer: Shake it.", "Partygoer: Shake it."),
    ("Jasor: Alright.", "Jason: Alright."),
    ("Cierk: You're all set.", "Clerk: You're all set."),
    ("Locat: Yeah!", "Local: Yeah!"),
    ("Locats: You askin' the wrong one.", "Locals: You askin' the wrong one."),
])
def test_misread_name_is_replaced_with_the_known_one(misread, fixed):
    assert SpeakerBook().fix(misread) == fixed


@pytest.mark.parametrize("text", ["Jack: Hey.", "Tourists: OW!", "Locals: Um.", "Kamal: Yo.", "Ray: Hi.",
                                  "Lucia: Raut is a word here.", "No name here, Raut."])
def test_known_short_or_unrelated_names_and_plain_lines_stay(text):
    assert SpeakerBook().fix(text) == text


def test_every_line_of_a_two_speaker_read_is_fixed():
    assert SpeakerBook().fix("Raut: Go!\nL'ucia: Wait.") == "Raul: Go!\nLucia: Wait."


def test_new_name_read_many_times_is_learned_and_its_misreads_fixed(tmp_path):
    book = SpeakerBook(tmp_path / LEARNED_NAMES_FILE)
    for _ in range(LEARN_AFTER_READS):
        book.fix("Brian: Yo.")

    assert book.fix("Briam: Yo.") == "Brian: Yo."
    # Megmarad a következő indításra is
    assert SpeakerBook(tmp_path / LEARNED_NAMES_FILE).fix("Briam: Yo.") == "Brian: Yo."


def test_name_read_only_a_few_times_is_not_learned():
    book = SpeakerBook()
    for _ in range(3):
        book.fix("Briam: Yo.")

    assert book.fix("Brian: Yo.") == "Brian: Yo."
    assert "Briam" not in book.known


def test_unreadable_learned_names_file_starts_empty(tmp_path):
    path = tmp_path / LEARNED_NAMES_FILE
    path.write_text("{broken", encoding="utf-8")

    assert SpeakerBook(path).fix("Raut: Go!") == "Raul: Go!"
