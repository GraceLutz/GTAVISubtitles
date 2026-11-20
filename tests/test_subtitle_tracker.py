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


def test_normalize_collapses_whitespace():
    assert normalize("  We   go\nnow ") == "We go now"
