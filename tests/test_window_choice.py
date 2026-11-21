import pytest

from gta_felirat.window_capture import VisibleWindow, WindowNotFoundError, choose_window

REMOTE_PLAY = VisibleWindow(hwnd=11, title="PS Remote Play", area=1920 * 1080)
REMOTE_PLAY_DIALOG = VisibleWindow(hwnd=12, title="PS Remote Play", area=360 * 97)
VEYRA = VisibleWindow(hwnd=21, title="Veyra — 采集卡 · LIVE", area=1920 * 1080)
BROWSER = VisibleWindow(hwnd=31, title="Likely7/Veyra-NRVideo - Google Chrome", area=2560 * 1440)
VEYRA_PLAIN = VisibleWindow(hwnd=22, title="Veyra 2.2.0", area=1920 * 1080)
REPO_TAB = VisibleWindow(hwnd=32, title="Veyra-NRVideo/README.md at main - Brave", area=2560 * 1440)


def test_window_with_the_exact_title_is_chosen():
    assert choose_window("PS Remote Play", [BROWSER, REMOTE_PLAY]) == REMOTE_PLAY.hwnd


def test_largest_of_several_windows_with_the_same_title_is_chosen():
    # A Remote Play egy kis párbeszédablakot is nyithat ugyanazzal a címmel
    assert choose_window("PS Remote Play", [REMOTE_PLAY_DIALOG, REMOTE_PLAY]) == REMOTE_PLAY.hwnd


def test_title_can_be_given_by_its_beginning():
    # A Veyra címe kínai betűket is tartalmaz: elég az elejét megadni
    assert choose_window("Veyra", [BROWSER, VEYRA], by_start=True) == VEYRA.hwnd


def test_beginning_of_the_title_ignores_letter_case_and_spaces_around_it():
    assert choose_window("  veyra ", [VEYRA], by_start=True) == VEYRA.hwnd


def test_exact_title_wins_over_a_longer_title_with_the_same_beginning():
    plain = VisibleWindow(hwnd=41, title="Xbox", area=800 * 600)
    longer = VisibleWindow(hwnd=42, title="Xbox Game Bar", area=1920 * 1080)

    assert choose_window("Xbox", [longer, plain], by_start=True) == plain.hwnd


def test_title_found_only_in_the_middle_does_not_count():
    with pytest.raises(WindowNotFoundError):
        choose_window("Veyra", [BROWSER], by_start=True)


def test_empty_title_matches_nothing():
    with pytest.raises(WindowNotFoundError):
        choose_window("  ", [VEYRA], by_start=True)


def test_built_in_title_is_not_matched_by_its_beginning():
    # Ha a Remote Play nem fut, egy „PS Remote Play | PlayStation” böngészőlapot ne olvassunk és fordítsunk
    browser_tab = VisibleWindow(hwnd=51, title="PS Remote Play | PlayStation - Google Chrome", area=2560 * 1440)

    with pytest.raises(WindowNotFoundError):
        choose_window("PS Remote Play", [browser_tab])


def test_title_with_the_version_after_the_name_is_found_by_its_beginning():
    # A Veyra 2.2.0 ablakának címe csak a név és a verzió
    assert choose_window("Veyra", [REPO_TAB, VEYRA_PLAIN], by_start=True) == VEYRA_PLAIN.hwnd


def test_title_that_only_continues_the_word_does_not_count():
    # Egy „Veyra-NRVideo …” című böngészőlap nem a Veyra ablaka
    with pytest.raises(WindowNotFoundError):
        choose_window("Veyra", [REPO_TAB], by_start=True)
