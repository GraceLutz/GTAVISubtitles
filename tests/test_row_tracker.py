from gta_felirat.ocr import Box
from gta_felirat.row_tracker import (GONE_GRACE_CAPTURES, MAX_CAPTURE_GAP_SECONDS, SETTLE_SECONDS, DetectedRow,
                                     RowText, RowTracker, is_same_place, overlaps)

UPPER = Box(700, 913, 1220, 950)
LOWER = Box(780, 957, 1140, 994)
CAPTURE_SECONDS = 0.033


def same_pixels(first, second):
    """Hamis aláírás-összehasonlítás: a tesztekben az aláírás maga a sor „szövege”."""
    return first == second


def detected(box, signature, sure=True, whiteness=0.1):
    return DetectedRow(box, signature, whiteness, sure)


# --- ugyanazok a képpontok = ugyanaz a sor ------------------------------------------

def test_capture_without_rows_tracks_nothing():
    assert RowTracker(same_pixels).update([], 1.0) == ()


def test_same_pixels_keep_their_row_id_and_become_stable():
    tracker = RowTracker(same_pixels)
    first = tracker.update([detected(LOWER, "a")], 1.0)

    second = tracker.update([detected(LOWER, "a")], 1.0 + CAPTURE_SECONDS)

    assert second[0].row_id == first[0].row_id
    assert (first[0].stable_captures, second[0].stable_captures) == (1, 2)
    assert second[0].first_seen == 1.0


def test_row_that_the_game_scrolls_up_keeps_its_id():
    tracker = RowTracker(same_pixels)
    before = tracker.update([detected(LOWER, "a")], 1.0)

    after = tracker.update([detected(UPPER, "a"), detected(LOWER, "b")], 1.0 + CAPTURE_SECONDS)

    assert (after[0].row_id, after[0].box) == (before[0].row_id, UPPER)
    assert after[1].row_id != before[0].row_id and after[1].stable_captures == 1  # csak az alsó sor új


def test_tracked_row_takes_the_latest_box_and_whiteness():
    tracker = RowTracker(same_pixels)
    tracker.update([detected(LOWER, "a", whiteness=0.1)], 1.0)
    moved = Box(LOWER.left + 2, LOWER.top - 1, LOWER.right + 1, LOWER.bottom)

    rows = tracker.update([detected(moved, "a", whiteness=0.2)], 1.0 + CAPTURE_SECONDS)

    assert (rows[0].box, rows[0].whiteness) == (moved, 0.2)


def test_other_text_in_the_same_slot_is_a_new_row():
    tracker = RowTracker(same_pixels)
    before = tracker.update([detected(LOWER, "a")], 1.0)

    after = tracker.update([detected(LOWER, "b")], 1.0 + CAPTURE_SECONDS)

    assert len(after) == 1  # a régi sor türelmi idő nélkül eltűnik: más áll a helyén
    assert after[0].row_id != before[0].row_id and after[0].stable_captures == 1


def test_two_rows_with_the_same_pixels_keep_separate_ids():
    tracker = RowTracker(same_pixels)
    before = tracker.update([detected(UPPER, "no"), detected(LOWER, "no")], 1.0)

    after = tracker.update([detected(UPPER, "no"), detected(LOWER, "no")], 1.0 + CAPTURE_SECONDS)

    assert [row.row_id for row in after] == [row.row_id for row in before]
    assert before[0].row_id != before[1].row_id


def test_rows_are_listed_from_top_to_bottom():
    tracker = RowTracker(same_pixels)

    rows = tracker.update([detected(LOWER, "b"), detected(UPPER, "a")], 1.0)

    assert [row.box for row in rows] == [UPPER, LOWER]


def test_row_the_detector_was_sure_about_once_stays_sure():
    tracker = RowTracker(same_pixels)
    tracker.update([detected(LOWER, "a", sure=True)], 1.0)

    rows = tracker.update([detected(LOWER, "a", sure=False)], 1.0 + CAPTURE_SECONDS)

    assert rows[0].sure


def test_row_becomes_sure_when_a_later_capture_is_sure():
    tracker = RowTracker(same_pixels)
    first = tracker.update([detected(LOWER, "a", sure=False)], 1.0)

    second = tracker.update([detected(LOWER, "a", sure=True)], 1.0 + CAPTURE_SECONDS)

    assert (first[0].sure, second[0].sure) == (False, True)


# --- egy rossz képkocka nem tüntet el egy sort ------------------------------------

def test_row_survives_a_short_dropout_and_then_goes():
    tracker = RowTracker(same_pixels)
    seen = tracker.update([detected(LOWER, "a")], 1.0)

    during = [tracker.update([], 1.0 + CAPTURE_SECONDS * (step + 1)) for step in range(GONE_GRACE_CAPTURES)]
    after = tracker.update([], 1.0 + CAPTURE_SECONDS * (GONE_GRACE_CAPTURES + 1))

    assert [[(row.row_id, row.missing) for row in rows] for rows in during] == [
        [(seen[0].row_id, step + 1)] for step in range(GONE_GRACE_CAPTURES)]
    assert after == ()


def test_row_that_returns_from_a_dropout_keeps_its_id():
    tracker = RowTracker(same_pixels)
    seen = tracker.update([detected(LOWER, "a")], 1.0)
    tracker.update([], 1.0 + CAPTURE_SECONDS)

    back = tracker.update([detected(LOWER, "a")], 1.0 + 2 * CAPTURE_SECONDS)

    assert (back[0].row_id, back[0].missing, back[0].stable_captures) == (seen[0].row_id, 0, 2)


def test_missing_row_does_not_linger_where_another_row_stands_now():
    tracker = RowTracker(same_pixels)
    tracker.update([detected(UPPER, "a"), detected(LOWER, "b")], 1.0)

    rows = tracker.update([detected(UPPER, "b"), detected(LOWER, "c")], 1.0 + CAPTURE_SECONDS)

    assert [row.box for row in rows] == [UPPER, LOWER]  # az „a” sor helyére a „b” görgött: nincs türelmi idő
    assert all(row.missing == 0 for row in rows)


# --- az eltakarás órája (first_seen) -----------------------------------------------

def test_row_that_is_still_fading_in_keeps_its_first_seen():
    tracker = RowTracker(same_pixels)
    tracker.update([detected(LOWER, "half visible")], 1.0)

    rows = tracker.update([detected(LOWER, "fully visible")], 1.0 + CAPTURE_SECONDS)

    assert rows[0].first_seen == 1.0  # új azonosító, de az eltakarás órája nem indul újra


def test_row_that_replaces_a_settled_one_starts_its_own_clock():
    tracker = RowTracker(same_pixels)
    tracker.update([detected(LOWER, "a")], 1.0)
    replaced_at = 1.0 + SETTLE_SECONDS + CAPTURE_SECONDS

    rows = tracker.update([detected(LOWER, "b")], replaced_at)

    assert rows[0].first_seen == replaced_at


def test_new_row_elsewhere_does_not_inherit_a_young_rows_clock():
    tracker = RowTracker(same_pixels)
    tracker.update([detected(UPPER, "a")], 1.0)

    rows = tracker.update([detected(LOWER, "b")], 1.0 + CAPTURE_SECONDS)

    assert [row.first_seen for row in rows if row.box == LOWER] == [1.0 + CAPTURE_SECONDS]


# --- új ablakméret: minden elölről -----------------------------------------------

def test_reset_forgets_the_rows_and_never_reuses_an_id():
    tracker = RowTracker(same_pixels)
    before = tracker.update([detected(LOWER, "a")], 1.0)
    tracker.reset()

    after = tracker.update([detected(LOWER, "a")], 1.0 + CAPTURE_SECONDS)

    assert after[0].row_id > before[0].row_id and after[0].stable_captures == 1


def test_row_seen_before_a_long_pause_counts_as_new_afterwards():
    tracker = RowTracker(same_pixels)
    before = tracker.update([detected(LOWER, "a")], 1.0)

    after = tracker.update([detected(LOWER, "a")], 1.0 + MAX_CAPTURE_GAP_SECONDS + 0.1)

    assert after[0].row_id != before[0].row_id  # ami a szünetben történt, nem láttuk: újra el kell olvasni
    assert (after[0].stable_captures, after[0].first_seen) == (1, 1.0 + MAX_CAPTURE_GAP_SECONDS + 0.1)


def test_row_that_vanished_during_a_long_pause_gets_no_grace_period():
    tracker = RowTracker(same_pixels)
    tracker.update([detected(LOWER, "a")], 1.0)

    assert tracker.update([], 1.0 + MAX_CAPTURE_GAP_SECONDS + 0.1) == ()


def test_short_pause_between_captures_keeps_the_rows():
    tracker = RowTracker(same_pixels)
    before = tracker.update([detected(LOWER, "a")], 1.0)

    after = tracker.update([detected(LOWER, "a")], 1.0 + MAX_CAPTURE_GAP_SECONDS - 0.1)

    assert after[0].row_id == before[0].row_id


# --- dobozok összevetése ------------------------------------------------------------

def test_boxes_that_only_touch_do_not_overlap():
    assert not overlaps(Box(0, 0, 10, 10), Box(10, 0, 20, 10))
    assert overlaps(Box(0, 0, 10, 10), Box(9, 9, 20, 20))


def test_nearly_identical_boxes_are_the_same_place():
    jittered = Box(LOWER.left + 3, LOWER.top - 2, LOWER.right + 4, LOWER.bottom + 1)

    assert is_same_place(LOWER, jittered)


def test_row_measured_taller_against_a_moving_bright_background_is_the_same_place():
    blurred = Box(LOWER.left - 6, LOWER.top - 6, LOWER.right + 5, LOWER.bottom + 4)  # elmosódott betűszélek

    assert is_same_place(LOWER, blurred)


def test_slightly_wider_box_of_another_text_is_not_the_same_place():
    wider = Box(LOWER.left - 60, LOWER.top, LOWER.right + 60, LOWER.bottom)

    assert not is_same_place(LOWER, wider)


def test_same_slot_with_a_clearly_different_width_is_not_the_same_place():
    shorter = Box(LOWER.left + 80, LOWER.top, LOWER.right - 80, LOWER.bottom)

    assert not is_same_place(LOWER, shorter)


def test_rows_above_each_other_are_not_the_same_place():
    assert not is_same_place(UPPER, LOWER)


# --- olvasható-e a felismert szöveg ---------------------------------------------------

def test_text_is_legible_when_it_is_not_empty_and_the_reader_is_confident():
    assert RowText(1, "Go.", 0.9).is_legible(0.6)


def test_empty_or_uncertain_text_is_not_legible():
    assert not RowText(1, "  ", 0.9).is_legible(0.6)
    assert not RowText(1, "Go.", 0.5).is_legible(0.6)
