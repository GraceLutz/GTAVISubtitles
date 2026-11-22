import pytest

from gta_felirat.subtitle_tracker import NewSubtitle, SubtitleCleared, SubtitleTracker, normalize


def make_tracker(stable_reads=2, hold=0.5):
    return SubtitleTracker(stable_reads=stable_reads, similarity_threshold=0.85, hold_seconds=hold)


def test_commits_text_only_after_stable_reads():
    tracker = make_tracker()
    assert tracker.update("We go now.", 0.0) is None
    assert tracker.update("We go now.", 0.25) == NewSubtitle("We go now.")


def test_single_read_commits_immediately_when_stable_reads_is_one():
    tracker = make_tracker(stable_reads=1)
    assert tracker.update("Hello there", 0.0) == NewSubtitle("Hello there")


def test_ocr_jitter_on_committed_text_does_not_retrigger():
    tracker = make_tracker()
    tracker.update("We need to get out of here, now.", 0.0)
    tracker.update("We need to get out of here, now.", 0.25)
    assert tracker.update("We need to get out of here, n0w.", 0.5) is None
    assert tracker.current == "We need to get out of here, now."


def test_jittery_candidate_reads_still_count_as_stable_and_keep_longer_variant():
    tracker = make_tracker()
    tracker.update("Get in the car", 0.0)
    assert tracker.update("Get in the car!", 0.25) == NewSubtitle("Get in the car!")


def test_fuller_read_of_a_name_cut_on_the_left_replaces_the_committed_text():
    tracker = make_tracker()
    tracker.update("ul: We gotta clear these fuckers", 0.0)
    tracker.update("ul: We gotta clear these fuckers", 0.1)

    assert tracker.update("Raul: We gotta clear these fuckers", 0.2) is None  # egy olvasás még kevés
    assert tracker.update("Raul: We gotta clear these fuckers", 0.3) == NewSubtitle(
        "Raul: We gotta clear these fuckers")


def test_fuller_confirmed_read_of_a_word_cut_on_the_right_replaces_the_committed_text():
    tracker = make_tracker()
    tracker.update("Raul: Just get there, mamit", 0.0, confirmed=True)

    assert tracker.update("Raul: Just get there, mamita.", 0.1, confirmed=True) == NewSubtitle(
        "Raul: Just get there, mamita.")


def test_read_adding_only_punctuation_does_not_retrigger():
    tracker = make_tracker()
    tracker.update("Raul: That's it. Good", 0.0, confirmed=True)

    assert tracker.update("Raul: That's it. Good.", 0.1, confirmed=True) is None


def test_shorter_read_after_the_fuller_one_does_not_retrigger():
    tracker = make_tracker()
    tracker.update("Raul: We gotta clear these fuckers", 0.0, confirmed=True)

    assert tracker.update("ul: We gotta clear these fuckers", 0.1, confirmed=True) is None


def test_shorter_read_missing_a_space_does_not_retrigger():
    # Valós hiba: a hosszú sor csonka olvasata („this?Hey, look,”) újra lefordult, aztán megint a teljes
    tracker = make_tracker()
    full = "Dre'Quan: The fuck kinda party is this? Hey, look, I ain't finna go in nobody crib I don't know like that"
    tracker.update(full, 0.0, confirmed=True)

    assert tracker.update("Dre'Quan: The fuck kinda party is this?Hey, look,", 0.1, confirmed=True) is None
    assert tracker.update(full + ".", 0.2, confirmed=True) is None


def test_reads_that_alternately_miss_the_upper_line_commit_only_once():
    # Valós hiba: a két beszélő sorát hol együtt, hol csak az alsót olvastuk, és mindkettőt újra meg újra kiírtuk
    tracker = make_tracker()
    lower = "Dre'Quan: Get your hand out my fries."
    both = "Lucia: Here...\n" + lower
    tracker.update(lower, 0.0, confirmed=True)

    assert tracker.update(both, 0.1, confirmed=True) == NewSubtitle(both)  # a kimaradt felső sor is kell
    for step in range(2, 8):
        assert tracker.update(lower if step % 2 else both, step / 10, confirmed=True) is None
    assert tracker.current == both


@pytest.mark.parametrize("cut", ["eed y'all 'to find ya favorite", ", I'mma need you to shake",
                                 "s, I'mma need you to shake", "ucia: I see hir", "you feel me?",
                                 ". oh, sorry, uh, Dre wanted me to com", "'cause you're doing too much."])
def test_read_cut_at_the_start_is_never_committed(cut):
    tracker = make_tracker()

    assert tracker.update(cut, 0.0, confirmed=True) is None
    assert tracker.update(cut, 0.1, confirmed=True) is None
    assert tracker.current == ""


def test_read_starting_with_an_apostrophe_word_is_committed():
    tracker = make_tracker()

    assert tracker.update("'Ey, Pooch. Bruh, shut up.", 0.0, confirmed=True) == NewSubtitle(
        "'Ey, Pooch. Bruh, shut up.")


def test_partial_read_after_the_full_line_does_not_retrigger():
    tracker = make_tracker()
    tracker.update("Bae-Luxe: Yeah, Dre'Quan, you not even worried about where she gone?", 0.0, confirmed=True)

    assert tracker.update("Bae-Luxe: Yeah, Dre'Quan,", 0.1, confirmed=True) is None
    assert tracker.current == "Bae-Luxe: Yeah, Dre'Quan, you not even worried about where she gone?"


def test_cut_read_keeps_the_committed_subtitle_alive():
    tracker = make_tracker(hold=0.5)
    tracker.update("Raul: We go now.", 0.0, confirmed=True)

    tracker.update("e go now.", 0.4, confirmed=True)

    assert tracker.update("", 0.6) is None  # 0,4-kor még látszott: nem tűnt el


def test_different_text_replaces_committed_after_stable_reads():
    tracker = make_tracker()
    tracker.update("First line here.", 0.0)
    tracker.update("First line here.", 0.25)
    assert tracker.update("Something completely different.", 0.5) is None
    assert tracker.update("Something completely different.", 0.75) == NewSubtitle(
        "Something completely different.")


def test_clears_only_after_hold_time():
    tracker = make_tracker(hold=0.5)
    tracker.update("Line", 0.0)
    tracker.update("Line", 0.25)
    assert tracker.update("", 0.5) is None
    assert tracker.update("", 0.8) == SubtitleCleared()
    assert tracker.current == ""


def test_empty_reads_without_subtitle_emit_nothing():
    tracker = make_tracker()
    assert tracker.update("", 0.0) is None
    assert tracker.update("   ", 5.0) is None


def test_normalize_collapses_whitespace_within_a_line():
    assert normalize("  We   go\tnow ") == "We go now"


def test_short_new_line_under_a_long_one_is_a_new_subtitle():
    # Valós eset: a rövid új sor az egész szöveghez mérve alig változtat, mégis új megszólalás
    tracker = make_tracker()
    long_line = "You have no idea who you're messing with."
    tracker.update(long_line, 0.0)
    tracker.update(long_line, 0.1)
    both = long_line + "\nYou're good."

    events = [tracker.update(both, 0.2), tracker.update(both, 0.3)]

    assert NewSubtitle(both) in events


# --- megerősítés a képpontokból (soralapú út): nem kell második olvasás -----------

def test_confirmed_text_is_committed_on_its_first_read():
    tracker = make_tracker()

    assert tracker.update("We go now.", 0.0, confirmed=True) == NewSubtitle("We go now.")
    assert tracker.current == "We go now."


def test_confirmed_read_of_the_committed_text_does_not_retrigger():
    tracker = make_tracker()
    tracker.update("We go now.", 0.0, confirmed=True)

    assert tracker.update("We go n0w.", 0.1, confirmed=True) is None


def test_confirmed_text_wins_over_an_unfinished_candidate():
    tracker = make_tracker()
    tracker.update("First line here.", 0.0)

    event = tracker.update("Something completely different.", 0.1, confirmed=True)

    assert event == NewSubtitle("Something completely different.")
    assert tracker.update("First line here.", 0.2) is None  # a régi jelölt számlálása elölről indul


def test_confirmed_empty_text_still_waits_for_the_hold_time():
    tracker = make_tracker(hold=0.5)
    tracker.update("Line", 0.0, confirmed=True)

    assert tracker.update("", 0.2, confirmed=True) is None
    assert tracker.update("", 0.6, confirmed=True) == SubtitleCleared()


def test_touch_keeps_the_subtitle_alive_while_its_pixels_are_on_screen():
    tracker = make_tracker(hold=0.5)
    tracker.update("Line", 0.0, confirmed=True)

    tracker.touch(5.0)

    assert tracker.update("", 5.3) is None  # csak 0,3 mp-e tűnt el: még várunk
    assert tracker.update("", 5.6) == SubtitleCleared()


def test_touch_without_a_subtitle_changes_nothing():
    tracker = make_tracker(hold=0.5)

    tracker.touch(5.0)

    assert tracker.current == ""
    assert tracker.update("", 9.0) is None
