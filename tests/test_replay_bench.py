"""A tools/replay_bench.py olyan részei, amelyekhez nem kellenek rögzített képkockák."""

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from gta_felirat.ocr import Box

BENCH_PATH = Path(__file__).resolve().parent.parent / "tools" / "replay_bench.py"


def _load_bench():
    # Útvonalból töltjük be: a "tools" nem csomag, és egy telepített "tools" csomag sem árnyékolhatja
    spec = importlib.util.spec_from_file_location("replay_bench", BENCH_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["replay_bench"] = module
    spec.loader.exec_module(module)
    return module


bench = _load_bench()


# --- fordító ---------------------------------------------------------------------------------------
def _store_file(tmp_path, entries):
    path = tmp_path / "store.json"
    path.write_text(json.dumps({"version": 1, "entries": entries}), encoding="utf-8")
    return path


def test_saved_translation_is_returned_from_the_store(tmp_path):
    path = _store_file(tmp_path, {"deepl:hu": {"go to hell!": "Menj a pokolba!"}})
    translator, backend = bench.make_offline_translator(path, "deepl:hu")

    assert translator.translate("Go to hell!") == "Menj a pokolba!"
    assert backend.calls == 0


def test_unknown_line_gets_the_placeholder(tmp_path):
    translator, backend = bench.make_offline_translator(_store_file(tmp_path, {"deepl:hu": {}}), "deepl:hu")

    assert translator.translate("Hello there") == "[HU] Hello there"
    assert backend.calls == 1


def test_speaker_name_is_kept_and_only_the_spoken_text_is_translated(tmp_path):
    translator, _ = bench.make_offline_translator(_store_file(tmp_path, {"deepl:hu": {}}), "deepl:hu")

    assert translator.translate("Lucia: Stop talking.") == "Lucia: [HU] Stop talking."


def test_other_namespace_is_searched_when_the_requested_one_misses(tmp_path):
    path = _store_file(tmp_path, {"google:hu": {"hi": "Szia"}, "deepl:hu": {}})
    translator, backend = bench.make_offline_translator(path, "deepl:hu")

    assert translator.translate("Hi") == "Szia"
    assert backend.calls == 0


def test_placeholder_translation_is_never_written_to_disk(tmp_path):
    path = _store_file(tmp_path, {"deepl:hu": {"a": "b"}})
    before = path.read_bytes()
    translator, _ = bench.make_offline_translator(path, "deepl:hu")

    translator.translate("A brand new line")

    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


# --- idő -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text, seconds", [("06:09:15", 22155), ("5:10", 18600), ("00:00:00", 0),
                                           ("23:59:59.5", 86399.5)])
def test_clock_text_is_parsed_to_seconds_of_day(text, seconds):
    assert bench.parse_clock(text) == seconds


@pytest.mark.parametrize("text", ["", "6", "25:00:00", "10:61", "10:00:61", "ab:cd:ef", "10-00-00"])
def test_invalid_clock_text_is_rejected(text):
    with pytest.raises(ValueError):
        bench.parse_clock(text)


def test_frame_timestamp_comes_from_the_raw_file_name_only():
    assert bench.frame_timestamp("1795318473.22_raw.jpg") == 1795318473.22
    assert bench.frame_timestamp("1795318473.22_screen.jpg") is None
    assert bench.frame_timestamp("notes.txt") is None


def test_range_selection_uses_local_time_with_the_utc_offset():
    # 2026-11-22 00:00 helyi (UTC+1) = 2026-11-21 23:00 UTC
    midnight = bench.local_midnight(1795318473.22, 1.0)
    frames = [bench.FrameFile(midnight + seconds, Path(f"{seconds}.jpg")) for seconds in (22154.0, 22155.0, 22160.5,
                                                                                             22170.0, 22171.0)]

    picked = bench.select_range(frames, midnight + bench.parse_clock("06:09:15"),
                                midnight + bench.parse_clock("06:09:30"))

    assert [frame.timestamp - midnight for frame in picked] == [22155.0, 22160.5, 22170.0]


def test_local_midnight_matches_the_recording_date():
    # 1795318473 = 2026-11-22 04:34:33 helyi idő (UTC+1)
    assert bench.local_midnight(1795318473.22, 1.0) == 1795318473.22 - (4 * 3600 + 34 * 60 + 33.22)


def test_clock_is_formatted_for_file_names():
    assert bench.format_clock(1795318473.22, 1.0) == "04-34-33.22"


def test_frames_are_listed_in_time_order_and_only_raw_ones(tmp_path):
    for name in ("200.50_raw.jpg", "100.25_raw.jpg", "100.25_screen.jpg", "readme.txt"):
        (tmp_path / name).write_bytes(b"")

    frames = bench.list_frames(tmp_path)

    assert [frame.timestamp for frame in frames] == [100.25, 200.5]


# --- szivárgásmérés ---------------------------------------------------------------------------------
def _frame_with_thin_text(width=300, height=60):
    """Sötét háttér, rajta vékony fehér "betűk" (a körvonal a sötét háttér): 3 pixel széles függőleges vonalak."""
    frame = np.full((height, width, 3), 20, np.uint8)
    for x in range(30, width - 30, 12):
        frame[15:45, x:x + 3] = 255
    return frame


def test_thin_white_strokes_are_text_pixels():
    letters = bench.text_pixels(_frame_with_thin_text())

    assert letters.sum() > 0


def test_a_solid_bright_area_is_not_text():
    frame = np.full((60, 300, 3), 255, np.uint8)  # napos fal: nincs körvonal

    assert bench.text_pixels(frame).sum() == 0


def test_thick_coloured_blob_is_not_a_name():
    frame = np.full((80, 120, 3), 20, np.uint8)
    frame[10:70, 10:100] = (40, 40, 230)  # vastag piros folt (hátsó lámpa)

    assert bench.text_pixels(frame).sum() == 0


def test_thin_coloured_stroke_is_a_name_letter():
    frame = np.full((80, 120, 3), 20, np.uint8)
    frame[20:60, 50:53] = (40, 200, 230)  # vékony sárga vonal

    assert bench.text_pixels(frame).sum() > 0


def test_uncovered_text_leaks_and_is_split_into_thirds():
    frame = _frame_with_thin_text()
    cover = np.zeros(frame.shape[:2], np.uint8)
    box = Box(0, 0, 300, 60)

    counts = bench.leak_counts(frame, cover, [box])

    assert counts.left > 0 and counts.middle > 0 and counts.right > 0
    assert counts.total == counts.text


def test_fully_covered_text_does_not_leak():
    frame = _frame_with_thin_text()
    cover = np.full(frame.shape[:2], 255, np.uint8)

    counts = bench.leak_counts(frame, cover, [Box(0, 0, 300, 60)])

    assert counts.total == 0
    assert counts.text > 0


def test_only_the_uncovered_right_part_counts_as_right_leak():
    frame = _frame_with_thin_text()
    cover = np.full(frame.shape[:2], 255, np.uint8)
    cover[:, 200:] = 0  # a sor jobb harmada nincs takarva

    counts = bench.leak_counts(frame, cover, [Box(0, 0, 300, 60)])

    assert counts.left == 0 and counts.middle == 0
    assert counts.right > 0


def test_text_outside_the_line_boxes_is_ignored():
    frame = _frame_with_thin_text()
    cover = np.zeros(frame.shape[:2], np.uint8)

    counts = bench.leak_counts(frame, cover, [Box(0, 50, 300, 60)])  # a betűk (15..45) fölött

    assert counts.total == 0


def test_box_beyond_the_frame_is_clipped_not_an_error():
    frame = _frame_with_thin_text()
    cover = np.zeros(frame.shape[:2], np.uint8)

    counts = bench.leak_counts(frame, cover, [Box(-20, -20, 400, 100)])

    assert counts.total > 0


def test_duplicate_and_nested_boxes_are_counted_once():
    boxes = [Box(0, 0, 100, 20), Box(0, 0, 100, 20), Box(2, 2, 98, 18), Box(0, 40, 100, 60)]

    assert len(bench.distinct_boxes(boxes)) == 2


def test_overlay_is_composited_over_the_frame_with_premultiplied_alpha():
    frame = np.full((2, 2, 3), 200, np.uint8)
    layer = np.zeros((2, 2, 4), np.uint8)
    layer[0, 0] = (100, 100, 100, 255)  # teljesen fedett
    layer[0, 1] = (50, 50, 50, 128)  # félig fedett, előszorzott
    out = bench.composite_premultiplied(frame, layer)

    assert tuple(out[0, 0]) == (100, 100, 100)
    assert all(abs(int(v) - (50 + 200 * 127 // 255)) <= 2 for v in out[0, 1])
    assert tuple(out[1, 1]) == (200, 200, 200)  # átlátszó: a kép marad


# --- hamis óra és végrehajtó --------------------------------------------------------------------------
def test_network_translation_arrives_after_the_delay_but_a_cached_one_is_immediate():
    clock = bench.BenchClock()
    calls = {"n": 0}

    def fetch(text):
        if text == "new":
            calls["n"] += 1
        return text

    executor = bench.ManualExecutor(clock, 0.5, lambda: calls["n"])
    clock.now = 10.0
    slow, fast = executor.submit(fetch, "new"), executor.submit(fetch, "known")
    fired = []
    slow.add_done_callback(lambda future: fired.append(clock.now))

    assert fast.done() and not slow.done()
    clock.now = 10.4
    executor.advance()
    assert not fired
    clock.now = 10.5
    executor.advance()
    assert slow.done() and fired == [10.5]


def test_second_request_for_the_same_text_waits_for_the_first_not_longer():
    clock = bench.BenchClock()
    calls = {"n": 0}

    def fetch(text):
        calls["n"] += 1
        return text

    executor = bench.ManualExecutor(clock, 0.5, lambda: calls["n"])
    clock.now = 1.0
    prefetched = executor.submit(fetch, "line")  # az előre indított fordítás
    prefetched.done()
    clock.now = 1.3
    confirmed = executor.submit(fetch, "line")  # a megerősített fordítás ugyanarra a szövegre
    confirmed.done()

    assert confirmed.ready_at == 1.5


def test_failed_task_raises_from_the_future_result():
    clock = bench.BenchClock()

    def broken(text):
        raise RuntimeError("boom")

    future = bench.ManualExecutor(clock, 0.0, lambda: 0).submit(broken, "x")

    with pytest.raises(RuntimeError):
        future.result()


def test_clocked_time_gives_the_fake_monotonic_and_the_real_rest():
    clock = bench.BenchClock()
    clock.now = 1234.5
    shim = bench.ClockedTime(clock)

    assert shim.monotonic() == 1234.5
    assert shim.perf_counter() > 0


def test_recognised_texts_reach_the_mailbox_only_after_the_recognition_delay():
    clock = bench.BenchClock()
    inbox = bench.DelayedInbox(lambda: None, clock, 0.2)
    clock.now = 5.0
    inbox.put_texts(["text"])

    inbox.release()
    assert inbox.drain().texts == ()
    clock.now = 5.2
    inbox.release()
    assert inbox.drain().texts == ("text",)


def test_simulated_recognizer_reads_the_latest_job_and_is_busy_for_its_delay():
    clock = bench.BenchClock()
    box = bench.PollableBox()
    handled = []
    recognizer = bench.SimulatedRecognizer(clock, 0.2)
    recognizer.attach(box, handled.append)

    clock.now = 1.0
    box.put("old")
    box.put("new")  # a LatestBox-ban csak a legfrissebb marad
    recognizer.service()
    box.put("next")
    clock.now = 1.1
    recognizer.service()  # még foglalt
    clock.now = 1.2
    recognizer.service()

    assert handled == ["new", "next"]


# --- kimenet ---------------------------------------------------------------------------------------
def test_metrics_csv_has_one_row_per_record_and_the_documented_columns(tmp_path):
    leak = bench.LeakCounts(left=3, middle=0, right=9, text=100)
    record = bench.FrameRecord("06-09-15.10", 1.0, 0.1, "[HU] szia", "Hi @1,2,3,4", 1, "1,2,3,4", 1, "5,6,7,8", 0,
                               leak, image_name="x_bench.jpg", flagged=True)
    path = tmp_path / "metrics.csv"

    bench.write_metrics(path, [record])

    rows = path.read_text(encoding="utf-8-sig").splitlines()
    assert rows[0].split(",") == list(bench.CSV_COLUMNS)
    assert len(rows) == 2 and ",12,1,x_bench.jpg" in rows[1]
