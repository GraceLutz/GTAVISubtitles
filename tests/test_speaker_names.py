"""A félreolvasott beszélőnevek javítása és az új nevek megtanulása."""

import json

import pytest

from gta_felirat.config import load_config, save_settings
from gta_felirat.speaker_names import LEARN_AFTER_READS, SAVE_FIX_COUNT_EVERY, SpeakerBook


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

def test_new_name_read_many_times_is_learned_and_its_misreads_fixed():
    saved = []
    book = SpeakerBook(save=saved.append)
    for _ in range(LEARN_AFTER_READS):
        book.fix("Brian: Yo.")

    assert book.fix("Briam: Yo.") == "Brian: Yo."
    # A következő indításkor a mentett állapotból is tudja
    assert SpeakerBook(saved[-1]).fix("Briam: Yo.") == "Brian: Yo."


def test_name_read_only_a_few_times_is_not_learned():
    book = SpeakerBook()
    for _ in range(3):
        book.fix("Briam: Yo.")

    assert book.fix("Brian: Yo.") == "Brian: Yo."
    assert "Briam" not in book.known


def test_fixes_are_saved_with_their_count():
    saved = []
    book = SpeakerBook(save=saved.append)
    for _ in range(SAVE_FIX_COUNT_EVERY):
        book.fix("Raut: Go!")

    assert saved[-1] == {"learned": [], "fixes": {"Raut": {"name": "Raul", "count": SAVE_FIX_COUNT_EVERY}}}


def test_a_known_fix_is_not_saved_on_every_line():
    saved = []
    book = SpeakerBook(save=saved.append)
    for _ in range(SAVE_FIX_COUNT_EVERY - 1):
        book.fix("Raut: Go!")

    assert len(saved) == 1  # csak az első, új javításkor


def test_saved_counts_continue_after_restart():
    book = SpeakerBook({"learned": [], "fixes": {"Raut": {"name": "Raul", "count": 4}}})
    book.fix("Raut: Go!")

    assert book.data["fixes"]["Raut"]["count"] == 5


@pytest.mark.parametrize("data", [
    "broken",
    {"learned": "Brian", "fixes": []},
    {"learned": [3, "Brian"], "fixes": {"Raut": {"name": 1}, "Locat": {"name": "Local", "count": 2}}},
])
def test_malformed_saved_names_are_skipped_not_fatal(data):
    book = SpeakerBook(data)

    assert book.fix("Raut: Go!") == "Raul: Go!"


def test_partly_malformed_data_keeps_the_valid_part():
    book = SpeakerBook({"learned": [3, "Brian"], "fixes": {"Raut": {"name": 1}}})

    assert "Brian" in book.known
    assert book.data["fixes"] == {}


def test_learned_names_round_trip_through_config_json(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")
    book = SpeakerBook(save=lambda data: save_settings(path, {"speaker_names": data}))
    book.fix("Raut: Go!")

    config = load_config(path)

    assert config.speaker_names == {"learned": [], "fixes": {"Raut": {"name": "Raul", "count": 1}}}
    assert json.loads(path.read_text(encoding="utf-8"))["speaker_names"] == config.speaker_names


def test_save_failure_does_not_break_the_fix():
    def failing_save(_data):
        raise OSError("lemez megtelt")

    assert SpeakerBook(save=failing_save).fix("Raut: Go!") == "Raul: Go!"
