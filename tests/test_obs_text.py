"""A kiírt felirat szövegfájlba az OBS szövegforrásának."""

from gta_felirat.layer_plan import Shown, ShownLine
from gta_felirat.obs_text import ObsTextFile, shown_text
from gta_felirat.ocr import Box

ANCHOR = Box(100, 900, 800, 990)


def two_lines() -> Shown:
    lines = (ShownLine("Lucia: Gyere, menjünk.", Box(100, 900, 800, 940)),
             ShownLine("Jason: Jó, megyek.", Box(100, 950, 800, 990)))
    return Shown("Lucia: Gyere, menjünk.\nJason: Jó, megyek.", ANCHOR, lines)


def test_translated_lines_go_one_per_line():
    assert shown_text(two_lines()) == "Lucia: Gyere, menjünk.\nJason: Jó, megyek."


def test_nothing_shown_is_empty_text():
    assert shown_text(None) == ""


def test_only_covered_english_lines_give_no_text():
    covering_only = Shown("", ANCHOR, (), hidden=(ANCHOR,))

    assert shown_text(covering_only) == ""


def test_streamer_mode_masks_slurs():
    shown = Shown("Local: Nigga, mi van?", ANCHOR)

    assert "Nigga" not in shown_text(shown, streamer_mode=True)
    assert "Nigga" in shown_text(shown)


def test_file_holds_the_current_text(tmp_path):
    output = ObsTextFile(tmp_path / "obs_felirat.txt")

    output.update("Lucia: Szia!")
    output.update("Jason: Helló!")

    assert output.path.read_text(encoding="utf-8") == "Jason: Helló!"
    assert not list(tmp_path.glob("*.new"))


def test_clear_empties_the_file(tmp_path):
    output = ObsTextFile(tmp_path / "obs_felirat.txt")
    output.update("Lucia: Szia!")

    output.clear()

    assert output.path.read_text(encoding="utf-8") == ""


def test_clear_does_not_create_a_file(tmp_path):
    ObsTextFile(tmp_path / "obs_felirat.txt").clear()

    assert not (tmp_path / "obs_felirat.txt").exists()


def test_unwritable_file_is_retried_later(tmp_path):
    blocked = tmp_path / "obs_felirat.txt"
    blocked.mkdir()  # könyvtárra nem cserélhető a fájl: az írás hibát ad, a program nem áll le
    output = ObsTextFile(blocked)

    output.update("Lucia: Szia!")
    blocked.rmdir()
    output.update("Lucia: Szia!")

    assert blocked.read_text(encoding="utf-8") == "Lucia: Szia!"
