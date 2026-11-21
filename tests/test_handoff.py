import codecs
import threading
import time
import tkinter
from dataclasses import dataclass

import numpy as np
import pytest

from gta_felirat.config import AppConfig, Region
from gta_felirat.handoff import (READ_ATTEMPTS, RETRY_CAPTURES, BandWatcher, Inbox, LatestBox, RowRecognizer,
                                 TkWaker, start_worker, tk_scheduler)
from gta_felirat.layer_plan import ShownLine
from gta_felirat.ocr import Box, region_to_box
from gta_felirat.row_pipeline import RowPipeline
from gta_felirat.row_tracker import (EMPTY_MAIL, GONE_GRACE_CAPTURES, RECOGNIZER, WATCHER, BandFrame, BandSnapshot,
                                     DetectedRow, RowFinder, RowObs, RowText, WorkerFailed)
from gta_felirat.translation_service import TranslationService
from gta_felirat.translator import CachedTranslator, TranslatorChoice

FRAME = (1920, 1080)
SMALL_FRAME = (1280, 720)
AREA = region_to_box(Region(), *FRAME)
BAND = np.zeros((AREA.height, AREA.width, 3), np.uint8)
UPPER = Box(700, 913, 1220, 950)
LOWER = Box(780, 957, 1140, 994)
TROUBLE = "We ain't lookin' for trouble, okay?"
HOW = "How about you, how you doin'?"
WAIT_SECONDS = 5  # felső határ egy másik szálra várva; a tesztek ezredmásodpercek alatt végeznek
QUIET_SECONDS = 0.05  # ennyit várunk, amikor azt nézzük, hogy valami NEM történik meg
MIN_SCORE = AppConfig().min_text_score


def snapshot_of(*rows, at=1.0):
    return BandSnapshot(BAND, AREA, FRAME, at, tuple(rows))


def seen_row(row_id, box, missing=0):
    return RowObs(row_id, box, 0.1, 1.0, 2, True, missing)


class Desk:
    """A munkaszálak kimenete egy helyen: a postaláda, az ébresztések száma és a felismerő feladatai."""

    def __init__(self):
        self.wakes = 0
        self.inbox = Inbox(self._count_wake)
        self.jobs = []

    def _count_wake(self):
        self.wakes += 1

    def put(self, job):
        """A LatestBox helyett: a tesztben minden feladat megmarad, hogy megnézhessük."""
        self.jobs.append(job)


# --- postaláda a Tk-szálnak ---------------------------------------------------------

def test_inbox_keeps_only_the_latest_snapshot_and_every_recognised_text():
    inbox = Inbox(lambda: None)
    old, new = snapshot_of(at=1.0), snapshot_of(at=1.033)
    first, second = RowText(1, TROUBLE, 0.99), RowText(2, HOW, 0.98)

    inbox.put_snapshot(old, urgent=True)
    inbox.put_texts([first])
    inbox.put_snapshot(new, urgent=False)
    inbox.put_texts([second])
    mail = inbox.drain()

    assert mail.snapshot is new
    assert mail.texts == (first, second)


def test_inbox_hands_everything_over_only_once():
    inbox = Inbox(lambda: None)
    inbox.put_snapshot(snapshot_of(), urgent=True)
    inbox.put_failure(WorkerFailed(WATCHER, "boom"))
    inbox.drain()

    assert inbox.drain() is EMPTY_MAIL


def test_inbox_wakes_the_tk_thread_only_when_there_is_something_to_do():
    desk = Desk()

    desk.inbox.put_snapshot(snapshot_of(), urgent=False)
    after_quiet_snapshot = desk.wakes
    desk.inbox.put_snapshot(snapshot_of(), urgent=True)
    desk.inbox.put_texts([RowText(1, TROUBLE, 0.99)])
    desk.inbox.put_failure(WorkerFailed(RECOGNIZER, "boom"))

    assert after_quiet_snapshot == 0
    assert desk.wakes == 3


def test_inbox_loses_nothing_when_several_threads_post_at_once():
    inbox = Inbox(lambda: None)
    posts_per_thread = 200

    def post(thread_number):
        for index in range(posts_per_thread):
            inbox.put_texts([RowText(thread_number * posts_per_thread + index, TROUBLE, 0.99)])

    threads = [threading.Thread(target=post, args=(number,)) for number in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=WAIT_SECONDS)

    assert sorted(text.row_id for text in inbox.drain().texts) == list(range(4 * posts_per_thread))


# --- egyelemű postaláda és a szálhurka ------------------------------------------------

def test_latest_box_never_queues_more_than_one_job():
    box = LatestBox()
    box.put("old")
    box.put("new")

    assert box.take() == "new"


def test_closed_box_hands_out_nothing_even_when_a_job_is_waiting():
    box = LatestBox()
    box.put("job")

    box.close()

    assert box.take() is None


def test_worker_handles_jobs_on_its_own_thread_and_leaves_when_the_box_is_closed():
    box = LatestBox()
    handled = threading.Event()
    seen = []

    def handle(job):
        seen.append((job, threading.get_ident()))
        handled.set()

    worker = start_worker("test-worker", box, handle)
    box.put("job")
    was_handled = handled.wait(timeout=WAIT_SECONDS)
    box.close()
    worker.join(timeout=WAIT_SECONDS)

    assert was_handled and not worker.is_alive()
    assert seen == [("job", worker.ident)] and worker.ident != threading.get_ident()


# --- a Tk-szál ébresztése -----------------------------------------------------------

class FakeScheduler:
    """A tk_scheduler hamisítványa: feljegyzi, melyik szál hívta; kérésre vár, vagy hibát dob."""

    def __init__(self, error=None):
        self.error = error
        self.threads = []
        self.entered = threading.Semaphore(0)
        self.gate = threading.Event()
        self.gate.set()

    def __call__(self):
        self.threads.append(threading.get_ident())
        self.entered.release()
        self.gate.wait(timeout=WAIT_SECONDS)
        if self.error is not None:
            raise self.error


@pytest.fixture
def wakers():
    """A tesztben létrehozott ébresztők; a végén mind leáll, és megvárjuk a száluk végét."""
    made = []

    def make(scheduler):
        made.append(TkWaker(scheduler))
        return made[-1]

    yield make
    for waker in made:
        waker.close()
        waker.join(timeout=WAIT_SECONDS)


def test_wake_schedules_the_callback_from_the_waker_thread_not_from_the_caller(wakers):
    scheduler = FakeScheduler()
    waker = wakers(scheduler)

    waker.wake()
    was_scheduled = scheduler.entered.acquire(timeout=WAIT_SECONDS)

    assert was_scheduled
    assert scheduler.threads != [threading.get_ident()]  # a hívó (munkaszál) sosem vár a Tk-szálra


def test_wakes_that_arrive_while_one_is_being_delivered_merge_into_one(wakers):
    scheduler = FakeScheduler()
    scheduler.gate.clear()
    waker = wakers(scheduler)
    waker.wake()
    scheduler.entered.acquire(timeout=WAIT_SECONDS)  # az első ébresztés éppen úton van

    for _ in range(5):
        waker.wake()
    scheduler.gate.set()
    scheduler.entered.acquire(timeout=WAIT_SECONDS)
    waker.close()
    waker.join(timeout=WAIT_SECONDS)

    assert len(scheduler.threads) == 2


def test_wake_without_a_running_main_loop_is_a_no_op_and_the_next_one_still_works(wakers):
    scheduler = FakeScheduler(RuntimeError("main thread is not in main loop"))
    waker = wakers(scheduler)

    waker.wake()
    first = scheduler.entered.acquire(timeout=WAIT_SECONDS)
    waker.wake()
    second = scheduler.entered.acquire(timeout=WAIT_SECONDS)

    assert first and second


def test_waker_stops_for_good_when_the_application_is_gone(wakers):
    scheduler = FakeScheduler(tkinter.TclError('can\'t invoke "after" command: application has been destroyed'))
    waker = wakers(scheduler)
    waker.wake()
    scheduler.entered.acquire(timeout=WAIT_SECONDS)
    waker.join(timeout=WAIT_SECONDS)

    waker.wake()

    assert not scheduler.entered.acquire(timeout=QUIET_SECONDS)


def test_closed_waker_schedules_nothing(wakers):
    scheduler = FakeScheduler()
    waker = wakers(scheduler)
    waker.close()
    waker.join(timeout=WAIT_SECONDS)

    waker.wake()

    assert not scheduler.entered.acquire(timeout=QUIET_SECONDS)


def test_waker_runs_the_callback_on_the_tcl_main_loop_thread(wakers):
    interp = tkinter.Tcl()  # csupasz Tcl-értelmező: nincs ablak
    seen = []

    def callback():
        seen.append(threading.get_ident())
        interp.quit()

    waker = wakers(tk_scheduler(interp, callback))
    interp.after(2000, interp.quit)  # ha az ébresztés nem érkezne meg
    interp.after(0, lambda: threading.Thread(target=waker.wake).start())  # munkaszál ébreszt, már futó huroknál
    started = time.monotonic()
    interp.mainloop(-1)
    elapsed = time.monotonic() - started
    waker.close()
    waker.join(timeout=WAIT_SECONDS)

    assert seen == [threading.get_ident()]
    assert elapsed < 1.0


# --- figyelő: sáv → sorok → pillanatkép ---------------------------------------------

class FakeDetector:
    """A sorkereső hamisítványa: minden sávon a beállított sorokat „találja”; az aláírás maga a szöveg."""

    def __init__(self):
        self.rows = ()
        self.error = None
        self.calls = []

    def __call__(self, band, area, frame_size):
        self.calls.append((band, area, frame_size))
        if self.error is not None:
            raise self.error
        return self.rows


def same_pixels(first, second):
    return first == second


def found(box, signature):
    return DetectedRow(box, signature, 0.1, True)


class Watch:
    """Egy figyelő a tesztben: a hamis sorkereső sorait látja a sorban érkező sávokon."""

    def __init__(self):
        self.desk = Desk()
        self.detector = FakeDetector()
        self.watcher = BandWatcher(RowFinder(self.detector, same_pixels), self.desk.inbox, self.desk)
        self.now = 1.0

    def capture(self, *rows, frame_size=FRAME, area=AREA):
        self.detector.rows = rows
        self.now += 0.033
        self.watcher.submit(BandFrame(BAND, area, frame_size, self.now))
        return self.desk.inbox.drain()


def test_watcher_publishes_the_rows_together_with_the_band_they_were_found_in():
    watch = Watch()

    mail = watch.capture(found(LOWER, "a"))

    assert [row.box for row in mail.snapshot.rows] == [LOWER]
    assert mail.snapshot.image is BAND and (mail.snapshot.area, mail.snapshot.frame_size) == (AREA, FRAME)
    assert mail.snapshot.captured_at == watch.now
    assert watch.detector.calls == [(BAND, AREA, FRAME)]


def test_watcher_takes_the_capture_threads_own_band_object_as_it_is():
    @dataclass(frozen=True, eq=False)
    class CapturedBand:  # a rögzítő szál (band_feed) ilyen mezőkkel adja a sávot
        image: np.ndarray
        area: Box
        frame_size: tuple
        captured_at: float
        sequence: int
        source: str

    watch = Watch()
    watch.detector.rows = (found(LOWER, "a"),)

    watch.watcher.submit(CapturedBand(BAND, AREA, FRAME, 2.5, 7, "wgc"))
    mail = watch.desk.inbox.drain()

    assert mail.failures == () and mail.snapshot.image is BAND
    assert (mail.snapshot.captured_at, mail.snapshot.rows[0].first_seen) == (2.5, 2.5)


def test_new_row_wakes_the_tk_thread_and_goes_to_the_recognizer():
    watch = Watch()
    watch.capture()
    wakes_before = watch.desk.wakes

    mail = watch.capture(found(LOWER, "a"))

    assert watch.desk.wakes == wakes_before + 1
    assert watch.desk.jobs == [mail.snapshot]


def test_row_is_announced_until_its_pixels_are_stable_and_then_the_tk_thread_is_left_alone():
    watch = Watch()
    watch.capture()
    wakes_before = watch.desk.wakes

    for _ in range(6):
        watch.capture(found(LOWER, "a"))

    assert watch.desk.wakes == wakes_before + 2  # a megjelenésekor, és amikor a képpontjai stabilak lettek
    assert len(watch.desk.jobs) == 1  # felismerésre csak egyszer megy


def test_row_that_scrolls_up_wakes_the_tk_thread_but_only_the_new_row_is_a_new_job():
    watch = Watch()
    for _ in range(4):
        watch.capture(found(LOWER, "a"))
    wakes_before = watch.desk.wakes

    mail = watch.capture(found(UPPER, "a"), found(LOWER, "b"))

    assert watch.desk.wakes == wakes_before + 1
    assert watch.desk.jobs[-1] is mail.snapshot and len(watch.desk.jobs) == 2


def test_row_that_disappears_wakes_the_tk_thread_without_a_job_for_the_recognizer():
    watch = Watch()
    for _ in range(4):
        watch.capture(found(LOWER, "a"))
    wakes_before = watch.desk.wakes

    watch.capture()

    assert watch.desk.wakes == wakes_before + 1
    assert len(watch.desk.jobs) == 1


def test_row_that_returns_after_a_dropout_is_offered_to_the_recognizer_again():
    watch = Watch()
    watch.capture(found(LOWER, "a"))
    watch.capture()

    mail = watch.capture(found(LOWER, "a"))

    assert watch.desk.jobs[-1] is mail.snapshot and len(watch.desk.jobs) == 2
    assert mail.snapshot.rows[0].missing == 0


def test_recognizer_gets_the_picture_again_from_time_to_time_while_a_row_is_visible():
    watch = Watch()

    for _ in range(1 + 2 * RETRY_CAPTURES):
        watch.capture(found(LOWER, "a"))

    assert len(watch.desk.jobs) == 3  # a megjelenésekor, aztán időnként: hátha most el tudja olvasni


def test_picture_without_rows_is_never_offered_to_the_recognizer():
    watch = Watch()

    for _ in range(3 * RETRY_CAPTURES):
        watch.capture()

    assert watch.desk.jobs == []


def test_new_window_size_starts_with_new_rows_and_is_announced_at_once():
    watch = Watch()
    for _ in range(4):
        before = watch.capture(found(LOWER, "a"))
    wakes_before = watch.desk.wakes

    after = watch.capture(found(LOWER, "a"), frame_size=SMALL_FRAME)
    emptied = watch.capture(frame_size=FRAME)

    assert after.snapshot.rows[0].row_id != before.snapshot.rows[0].row_id
    assert after.snapshot.rows[0].stable_captures == 1 and after.snapshot.frame_size == SMALL_FRAME
    assert emptied.snapshot.rows == () and watch.desk.wakes == wakes_before + 2


def test_switching_the_row_finder_starts_new_rows_without_reusing_an_old_id():
    watch = Watch()
    before = [watch.capture(found(UPPER, "a"), found(LOWER, "b")) for _ in range(3)][-1]
    old_ids = {row.row_id for row in before.snapshot.rows}

    watch.watcher.set_finder(RowFinder(watch.detector, same_pixels))
    after = watch.capture(found(LOWER, "b"))

    (row,) = after.snapshot.rows
    assert row.row_id not in old_ids and row.stable_captures == 1


def test_detector_failure_is_reported_once_and_the_watcher_keeps_working():
    watch = Watch()
    watch.detector.error = ValueError("band must be an 8-bit BGR image")

    failed = [watch.capture(found(LOWER, "a")) for _ in range(3)]
    watch.detector.error = None
    recovered = watch.capture(found(LOWER, "a"))

    assert [len(mail.failures) for mail in failed] == [1, 0, 0]  # hibasorozatból csak az első megy a naplóba
    assert failed[0].failures[0].worker == WATCHER and "8-bit BGR" in failed[0].failures[0].error
    assert all(mail.snapshot is None for mail in failed)
    assert [row.box for row in recovered.snapshot.rows] == [LOWER]


def test_second_streak_of_detector_failures_is_reported_again():
    watch = Watch()
    watch.detector.error = ValueError("first")
    watch.capture()
    watch.detector.error = None
    watch.capture()

    watch.detector.error = ValueError("second")
    mail = watch.capture()

    assert [failure.error for failure in mail.failures] == [repr(ValueError("second"))]


# --- felismerő: csak az olvasatlan sorok ----------------------------------------------

class FakeReader:
    """A SubtitleReader.read_rows hamisítványa: dobozonként a megadott szöveget „olvassa”."""

    def __init__(self, texts, score=0.9):
        self.texts = texts
        self.score = score
        self.error = None
        self.calls = []

    def read_rows(self, crop, area, rows):
        self.calls.append((crop, area, list(rows)))
        if self.error is not None:
            raise self.error
        return [(self.texts[box], self.score) for box in rows]


def recognizer_with(reader):
    desk = Desk()
    return RowRecognizer(reader.read_rows, desk.inbox, MIN_SCORE), desk


def test_recognizer_reads_a_new_row_from_the_band_of_its_snapshot():
    reader = FakeReader({LOWER: TROUBLE})
    recognizer, desk = recognizer_with(reader)

    recognizer.submit(snapshot_of(seen_row(1, LOWER)))

    assert reader.calls == [(BAND, AREA, [LOWER])]
    assert desk.inbox.drain().texts == (RowText(1, TROUBLE, 0.9),)
    assert desk.wakes == 1


def test_row_is_read_only_once_even_when_it_scrolls():
    reader = FakeReader({LOWER: TROUBLE, UPPER: TROUBLE})
    recognizer, desk = recognizer_with(reader)
    recognizer.submit(snapshot_of(seen_row(1, LOWER)))
    reader.texts = {UPPER: TROUBLE, LOWER: HOW}

    recognizer.submit(snapshot_of(seen_row(1, UPPER), seen_row(2, LOWER)))

    assert reader.calls[-1] == (BAND, AREA, [LOWER])  # a feljebb görgött sort nem olvassuk újra
    assert desk.inbox.drain().texts == (RowText(1, TROUBLE, 0.9), RowText(2, HOW, 0.9))
    assert recognizer.recognitions == 2


def test_nothing_is_read_or_posted_when_every_row_was_read_before():
    reader = FakeReader({LOWER: TROUBLE})
    recognizer, desk = recognizer_with(reader)
    recognizer.submit(snapshot_of(seen_row(1, LOWER)))
    desk.inbox.drain()

    recognizer.submit(snapshot_of(seen_row(1, LOWER)))

    assert len(reader.calls) == 1
    assert desk.inbox.drain() is EMPTY_MAIL and desk.wakes == 1


def test_row_that_is_missing_from_the_picture_is_read_when_it_is_back():
    reader = FakeReader({LOWER: TROUBLE})
    recognizer, desk = recognizer_with(reader)

    recognizer.submit(snapshot_of(seen_row(1, LOWER, missing=GONE_GRACE_CAPTURES)))
    while_missing = list(reader.calls)
    recognizer.submit(snapshot_of(seen_row(1, LOWER)))

    assert while_missing == []  # nincs a képen: nincs mit olvasni róla
    assert desk.inbox.drain().texts == (RowText(1, TROUBLE, 0.9),)


def test_row_that_could_not_be_read_is_tried_again_until_it_is_legible():
    reader = FakeReader({LOWER: "We ain't lookin'"}, score=MIN_SCORE - 0.2)
    recognizer, desk = recognizer_with(reader)
    recognizer.submit(snapshot_of(seen_row(1, LOWER)))
    reader.texts, reader.score = {LOWER: TROUBLE}, 0.9

    for _ in range(3):
        recognizer.submit(snapshot_of(seen_row(1, LOWER)))

    assert len(reader.calls) == 2  # a jó olvasat után nem olvassuk többször
    assert desk.inbox.drain().texts[-1] == RowText(1, TROUBLE, 0.9)


def test_row_that_stays_unreadable_is_given_up_after_a_few_attempts():
    reader = FakeReader({LOWER: ""}, score=0.0)
    recognizer, _desk = recognizer_with(reader)

    for _ in range(READ_ATTEMPTS + 3):
        recognizer.submit(snapshot_of(seen_row(1, LOWER)))

    assert len(reader.calls) == READ_ATTEMPTS


def test_reader_failure_is_reported_and_the_row_is_read_with_the_next_job():
    reader = FakeReader({LOWER: TROUBLE})
    recognizer, desk = recognizer_with(reader)
    reader.error = RuntimeError("onnxruntime failed")

    recognizer.submit(snapshot_of(seen_row(1, LOWER)))
    failed = desk.inbox.drain()
    reader.error = None
    recognizer.submit(snapshot_of(seen_row(1, LOWER)))

    assert failed.texts == () and [failure.worker for failure in failed.failures] == [RECOGNIZER]
    assert desk.inbox.drain().texts == (RowText(1, TROUBLE, 0.9),)


def test_reader_that_answers_for_the_wrong_number_of_rows_is_a_failure():
    desk = Desk()
    recognizer = RowRecognizer(lambda crop, area, rows: [], desk.inbox, MIN_SCORE)

    recognizer.submit(snapshot_of(seen_row(1, LOWER)))
    mail = desk.inbox.drain()

    assert mail.texts == () and [failure.worker for failure in mail.failures] == [RECOGNIZER]


# --- az egész út valódi szálakkal -----------------------------------------------------

def hu(text):
    return "HU<" + codecs.encode(text, "rot13") + ">"


class GatedBackend:
    """Hamis fordító: addig nem válaszol, amíg a teszt el nem engedi."""

    def __init__(self):
        self.release = threading.Event()

    def translate(self, text):
        self.release.wait(timeout=WAIT_SECONDS)
        return hu(text)


class Stage:
    """Az egész soralapú út valódi szálakkal: hamis sorkereső, olvasó és fordító, Tcl-eseményhurok ablak nélkül."""

    def __init__(self):
        self.interp = tkinter.Tcl()
        self.backend = GatedBackend()
        self.presented = []
        self.waker = TkWaker(tk_scheduler(self.interp, self._on_mail))
        self.inbox, self.jobs = Inbox(self.waker.wake), LatestBox()
        recognizer = RowRecognizer(FakeReader({LOWER: TROUBLE}).read_rows, self.inbox, MIN_SCORE)
        self.recognizer = start_worker("row-recognizer", self.jobs, recognizer.submit)
        service = TranslationService(TranslatorChoice("deepl", "hu", "", ""), CachedTranslator(self.backend))
        self.pipeline = RowPipeline(AppConfig(only_spoken_lines=False), None, service, wake=self.waker.wake)
        self.capturer = threading.Thread(target=self._capture_three_bands)

    def _on_mail(self):
        """A Tk-szálon fut, valahányszor a postaláda vagy a fordító ébreszt."""
        mail = self.inbox.drain()
        if self.pipeline.receive(mail.snapshot, mail.texts, time.monotonic()):
            self.presented.append((threading.get_ident(), self.pipeline.shown))
        shown = self.pipeline.shown
        if shown is not None and shown.hidden:
            self.backend.release.set()  # a sor már takarva van: jöhet a fordítás
        if shown is not None and shown.lines:
            self.interp.quit()

    def _capture_three_bands(self):
        """A „rögzítő szál”: ugyanaz a sor három egymás utáni sávon."""
        detector = FakeDetector()
        detector.rows = (found(LOWER, "a"),)
        watcher = BandWatcher(RowFinder(detector, same_pixels), self.inbox, self.jobs)
        for _ in range(3):
            watcher.submit(BandFrame(BAND, AREA, FRAME, time.monotonic()))
            time.sleep(0.01)

    def play(self):
        self.interp.after(0, self.capturer.start)  # a rögzítés már futó eseményhuroknál indul
        self.interp.after(3000, self.interp.quit)  # ha valami elakadna
        self.interp.mainloop(-1)

    def close(self):
        self.backend.release.set()
        self.jobs.close()
        self.waker.close()
        started = [thread for thread in (self.capturer, self.recognizer) if thread.ident is not None]
        for thread in (*started, self.waker):
            thread.join(timeout=WAIT_SECONDS)
        self.pipeline.shutdown()


def test_new_line_goes_from_capture_to_translation_across_the_threads():
    stage = Stage()

    try:
        stage.play()
    finally:
        stage.close()

    first, last = stage.presented[0][1], stage.presented[-1][1]
    assert first.hidden == (LOWER,) and first.lines == ()  # előbb csak eltakartuk az angol sort
    assert last.lines == (ShownLine(hu(TROUBLE), LOWER),)
    assert {thread for thread, _shown in stage.presented} == {threading.get_ident()}  # csak a Tk-szál ír
