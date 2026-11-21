import threading
import time

import numpy as np
import pytest

from gta_felirat.band_capture import GrabbedBand
from gta_felirat.band_feed import (MAX_FRUITLESS_STARTS, PRINTWINDOW_MAX_FPS, RETRY_SECONDS, SOURCE_NONE,
                                   SOURCE_PRINTWINDOW, SOURCE_WGC, STALE_SECONDS, BandFeed, FeedRuntime, SourceSpec,
                                   default_sources)
from gta_felirat.ocr import Box

HWND = 4242  # hamis ablakazonosító: a hamis forrásokkal sosem jut el a Windowshoz
CLIENT_SIZE = (1920, 1080)
AREA = Box(230, 842, 1689, 1058)
PERIOD = 0.01
START_TIME = 1000.0
WAIT_SECONDS = 2.0  # legfeljebb ennyit várunk a rögzítő szálra (rendesen néhány ezredmásodperc)
SPIN_PAUSE_SECONDS = 0.001  # a forgatókönyv végén a hamis forrás ennyit pihen, hogy a szál ne pörögjön üresen


def image(value):
    return np.full((AREA.height, AREA.width, 3), value, np.uint8)


def area_of(_width, _height):
    return AREA


def wait_until(condition, timeout=WAIT_SECONDS):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(SPIN_PAUSE_SECONDS)
    return False


class FakeClock:
    """Kézzel járó óra: a várakozás nem telik valódi idővel, csak előrébb teszi az órát."""

    def __init__(self):
        self.now = START_TIME
        self.sleeps = []
        self.on_sleep = None
        self.frozen = False  # a forgatókönyv végén megáll az idő: a legutóbbi sáv nem „öregszik” tovább

    def time(self):
        return self.now

    def sleep(self, seconds):
        if self.frozen:
            return
        if self.on_sleep is not None:
            self.on_sleep()
        self.sleeps.append(seconds)
        self.now += seconds


class FakeWindow:
    """Hamis ablak: megadott számú kérdés után „bezárul”; megjegyzi, melyik szál kérdezett rá (a rögzítő szál)."""

    def __init__(self, closes_after=None):
        self._answers_left = closes_after
        self._asked_by = set()

    def exists(self, _hwnd):
        self._asked_by.add(threading.get_ident())
        if self._answers_left is None:
            return True
        self._answers_left -= 1
        return self._answers_left >= 0

    def capture_thread_ended(self):
        return bool(self._asked_by) and not any(thread.ident in self._asked_by for thread in threading.enumerate())


class ScriptedSource:
    """Hamis képforrás: sorban adja a lépéseit (kép: új sáv, None: nincs új kép, kivétel: hiba)."""

    def __init__(self, clock, steps, read_seconds=0.0, period=PERIOD, band_period=None):
        self._clock = clock
        self._steps = list(steps)
        self._read_seconds = read_seconds
        self.poll_period = period
        self.band_period = period if band_period is None else band_period
        self.read_times = []
        self.read_threads = set()
        self.close_threads = []
        self.finished = threading.Event()  # a forgatókönyv elfogyott

    def read(self, area_of):
        self.read_times.append(self._clock.now)
        self.read_threads.add(threading.get_ident())
        if not self._steps:
            self._clock.frozen = True
            self.finished.set()
            time.sleep(SPIN_PAUSE_SECONDS)
            return None
        self._clock.now += self._read_seconds
        step = self._steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return None if step is None else GrabbedBand(step, area_of(*CLIENT_SIZE), CLIENT_SIZE)

    def close(self):
        self.close_threads.append(threading.get_ident())


class Opener:
    """Megnyitó: hívásonként a következő kimenetét adja (forrás, vagy kivétel: nem nyitható meg).

    Az utolsó kimenet ismétlődik.
    """

    def __init__(self, *outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        outcome = self._outcomes.pop(0) if len(self._outcomes) > 1 else self._outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def spec(name, *outcomes):
    """Forrásleírás a megadott megnyitóval, vagy a megnyitás egymás utáni kimeneteivel."""
    return SourceSpec(name, outcomes[0] if isinstance(outcomes[0], Opener) else Opener(*outcomes))


def make_feed(clock, *specs, on_band=None, window=None, area=area_of):
    runtime = FeedRuntime(clock=clock.time, sleep=clock.sleep, window_exists=(window or FakeWindow()).exists)
    return BandFeed(HWND, area, on_band=on_band, sources=specs, runtime=runtime)


@pytest.fixture
def run():
    """Elindítja a kapott feedet, és a teszt végén mindenképp leállítja."""
    feeds = []

    def start(feed):
        feeds.append(feed)
        feed.start()
        return feed

    yield start
    for feed in feeds:
        feed.stop()


# --- a latest() és a sorszám --------------------------------------------------

def test_there_is_no_band_and_no_source_before_the_feed_is_started():
    clock = FakeClock()
    feed = make_feed(clock, spec("fake", ScriptedSource(clock, [image(1)])))

    assert feed.latest() is None
    assert feed.source == SOURCE_NONE


def test_latest_returns_the_newest_band_with_where_and_when_it_was_captured(run):
    clock = FakeClock()
    source = ScriptedSource(clock, [image(10), image(20), image(30)])
    feed = run(make_feed(clock, spec("fake", source)))

    assert source.finished.wait(WAIT_SECONDS)
    band = feed.latest()

    assert band.image[0, 0].tolist() == [30, 30, 30]
    assert (band.area, band.frame_size, band.source, band.sequence) == (AREA, CLIENT_SIZE, "fake", 3)
    assert band.captured_at == pytest.approx(START_TIME + 2 * PERIOD)
    assert feed.source == "fake"


def test_every_new_band_is_announced_once_with_the_next_sequence_number(run):
    clock = FakeClock()
    announced = []
    source = ScriptedSource(clock, [image(1), None, image(2), None, None, image(3)])
    run(make_feed(clock, spec("fake", source), on_band=announced.append))

    assert source.finished.wait(WAIT_SECONDS)

    assert [(band.sequence, int(band.image[0, 0, 0])) for band in announced] == [(1, 1), (2, 2), (3, 3)]


def test_listener_runs_on_the_capture_thread(run):
    clock = FakeClock()
    listener_threads = []
    source = ScriptedSource(clock, [image(1), image(2)])
    run(make_feed(clock, spec("fake", source), on_band=lambda _band: listener_threads.append(threading.get_ident())))

    assert source.finished.wait(WAIT_SECONDS)

    assert set(listener_threads) == source.read_threads
    assert threading.get_ident() not in listener_threads


def test_published_image_cannot_be_modified(run):
    clock = FakeClock()
    source = ScriptedSource(clock, [image(5)])
    feed = run(make_feed(clock, spec("fake", source)))

    assert source.finished.wait(WAIT_SECONDS)

    with pytest.raises(ValueError):
        feed.latest().image[0, 0] = 0


def test_band_is_not_returned_once_the_source_has_been_silent_for_too_long(run):
    clock = FakeClock()
    silent_reads = [None] * (int(STALE_SECONDS / PERIOD) + 5)
    source = ScriptedSource(clock, [image(1), *silent_reads])
    feed = run(make_feed(clock, spec("fake", source)))

    assert source.finished.wait(WAIT_SECONDS)

    assert feed.latest() is None
    assert feed.source == "fake"


def test_band_is_returned_again_when_the_silent_source_delivers(run):
    clock = FakeClock()
    silent_reads = [None] * (int(STALE_SECONDS / PERIOD) + 5)
    source = ScriptedSource(clock, [image(1), *silent_reads, image(2)])
    feed = run(make_feed(clock, spec("fake", source)))

    assert source.finished.wait(WAIT_SECONDS)

    assert feed.latest().sequence == 2


# --- leállítás ----------------------------------------------------------------

def test_stop_closes_the_source_on_the_capture_thread(run):
    clock = FakeClock()
    source = ScriptedSource(clock, [image(1)])
    feed = run(make_feed(clock, spec("fake", source)))
    assert source.finished.wait(WAIT_SECONDS)

    feed.stop()

    assert source.close_threads == list(source.read_threads)
    assert threading.get_ident() not in source.close_threads


def test_there_is_no_band_and_no_source_after_stop(run):
    clock = FakeClock()
    source = ScriptedSource(clock, [image(1)])
    feed = run(make_feed(clock, spec("fake", source)))
    assert source.finished.wait(WAIT_SECONDS)

    feed.stop()

    assert feed.latest() is None
    assert feed.source == SOURCE_NONE


def test_stop_is_harmless_before_start_and_when_repeated(run):
    clock = FakeClock()
    never_started = make_feed(clock, spec("fake", ScriptedSource(clock, [])))
    source = ScriptedSource(clock, [image(1)])
    started = run(make_feed(clock, spec("fake", source)))
    assert source.finished.wait(WAIT_SECONDS)

    never_started.stop()
    started.stop()
    started.stop()

    assert len(source.close_threads) == 1


def test_feed_stops_and_releases_the_source_when_the_window_is_closed(run):
    clock = FakeClock()
    source = ScriptedSource(clock, [image(1), image(2), image(3)])
    window = FakeWindow(closes_after=2)  # a harmadik kérdésre az ablak már nincs meg
    feed = run(make_feed(clock, spec("fake", source), window=window))

    assert wait_until(window.capture_thread_ended)

    assert len(source.read_times) == 2
    assert len(source.close_threads) == 1
    assert feed.latest() is None and feed.source == SOURCE_NONE


# --- ütemezés -----------------------------------------------------------------

def test_reads_start_one_period_apart_however_long_a_read_takes(run):
    clock = FakeClock()
    source = ScriptedSource(clock, [image(1)] * 4, read_seconds=0.03, period=0.1)
    feed = run(make_feed(clock, spec("slow", source)))
    assert source.finished.wait(WAIT_SECONDS)
    feed.stop()

    assert np.diff(source.read_times[:4]) == pytest.approx([0.1, 0.1, 0.1])


def test_read_longer_than_the_period_is_followed_by_the_next_one_at_once(run):
    clock = FakeClock()
    source = ScriptedSource(clock, [image(1)] * 4, read_seconds=0.15, period=0.1)
    feed = run(make_feed(clock, spec("slow", source)))
    assert source.finished.wait(WAIT_SECONDS)
    feed.stop()

    assert np.diff(source.read_times[:4]) == pytest.approx([0.15, 0.15, 0.15])
    assert clock.sleeps == []


def test_source_rests_longer_after_a_band_than_after_an_empty_read(run):
    clock = FakeClock()
    source = ScriptedSource(clock, [image(1), None, None, image(2), None], period=0.004, band_period=0.03)
    feed = run(make_feed(clock, spec("frames", source)))
    assert source.finished.wait(WAIT_SECONDS)
    feed.stop()

    assert np.diff(source.read_times[:6]) == pytest.approx([0.03, 0.004, 0.004, 0.03, 0.004])


def test_window_capture_is_preferred_to_printwindow():
    names = [source.name for source in default_sources(HWND, max_fps=30.0)]

    assert names == [SOURCE_WGC, SOURCE_PRINTWINDOW]


def test_printwindow_is_read_at_most_fifteen_times_a_second():
    printwindow = default_sources(HWND, max_fps=30.0)[1].open()

    assert printwindow.poll_period == printwindow.band_period == pytest.approx(1 / PRINTWINDOW_MAX_FPS)


def test_printwindow_follows_a_frame_rate_limit_below_its_own():
    printwindow = default_sources(HWND, max_fps=5.0)[1].open()

    assert printwindow.band_period == pytest.approx(0.2)


# --- tartalékra váltás ----------------------------------------------------------

def test_next_source_is_used_when_the_first_cannot_be_opened(run):
    clock = FakeClock()
    fallback = ScriptedSource(clock, [image(7)])
    feed = run(make_feed(clock, spec("first", OSError("unsupported")), spec("second", fallback)))

    assert fallback.finished.wait(WAIT_SECONDS)

    assert feed.source == "second"
    assert feed.latest().source == "second"


def test_error_while_reading_switches_to_the_fallback_source(run):
    clock = FakeClock()
    announced = []
    failing = ScriptedSource(clock, [image(1), OSError("device lost")])
    fallback = ScriptedSource(clock, [image(2), image(3)])
    run(make_feed(clock, spec("first", failing), spec("second", fallback), on_band=announced.append))

    assert fallback.finished.wait(WAIT_SECONDS)

    assert [(band.sequence, band.source) for band in announced] == [(1, "first"), (2, "second"), (3, "second")]
    assert len(failing.close_threads) == 1


def test_failed_source_is_tried_again_after_the_retry_delay(run):
    clock = FakeClock()
    announced = []
    period = RETRY_SECONDS / 2  # a tartalék két olvasása alatt telik le a pihenő
    before_failure = ScriptedSource(clock, [image(1), OSError("device lost")], period=period)
    after_retry = ScriptedSource(clock, [image(9)], period=period)
    opener = Opener(before_failure, after_retry)
    fallback = ScriptedSource(clock, [image(2)] * 5, period=period)
    run(make_feed(clock, spec("first", opener), spec("second", fallback), on_band=announced.append))

    assert after_retry.finished.wait(WAIT_SECONDS)

    assert [band.source for band in announced] == ["first", "second", "second", "first"]
    assert opener.calls == 2
    assert len(fallback.close_threads) == 1


def test_source_that_never_opens_is_given_up_while_the_fallback_keeps_running(run):
    clock = FakeClock()
    opener = Opener(OSError("unsupported"))
    fallback = ScriptedSource(clock, [image(1)] * 8, period=RETRY_SECONDS)
    run(make_feed(clock, spec("first", opener), spec("second", fallback)))

    assert fallback.finished.wait(WAIT_SECONDS)

    assert opener.calls == MAX_FRUITLESS_STARTS
    assert fallback.close_threads == []


def test_feed_ends_without_a_source_when_every_source_has_failed(run):
    clock = FakeClock()
    first, second = Opener(OSError("unsupported")), Opener(OSError("no picture"))
    window = FakeWindow()
    feed = run(make_feed(clock, spec("first", first), spec("second", second), window=window))

    assert wait_until(window.capture_thread_ended)

    assert (first.calls, second.calls) == (MAX_FRUITLESS_STARTS, MAX_FRUITLESS_STARTS)
    assert feed.latest() is None and feed.source == SOURCE_NONE


def test_latest_is_cleared_while_no_source_is_usable(run):
    clock = FakeClock()
    seen_while_waiting = []
    before_failure = ScriptedSource(clock, [image(1), OSError("device lost")])
    after_retry = ScriptedSource(clock, [image(2)])
    feed = make_feed(clock, spec("only", before_failure, after_retry))
    clock.on_sleep = lambda: seen_while_waiting.append((feed.source, feed.latest()))
    run(feed)

    assert after_retry.finished.wait(WAIT_SECONDS)

    assert (SOURCE_NONE, None) in seen_while_waiting
    assert (feed.source, feed.latest().sequence) == ("only", 2)


def test_error_in_the_listener_stops_the_feed_and_releases_the_source(run, caplog):
    clock = FakeClock()
    source = ScriptedSource(clock, [image(1), image(2)])

    def broken_listener(_band):
        raise RuntimeError("listener bug")

    window = FakeWindow()
    feed = run(make_feed(clock, spec("fake", source), on_band=broken_listener, window=window))

    assert wait_until(window.capture_thread_ended)

    assert len(source.close_threads) == 1
    assert feed.latest() is None and feed.source == SOURCE_NONE
    assert "listener bug" in caplog.text


def test_error_in_the_area_callback_stops_the_feed_and_releases_the_source(run):
    clock = FakeClock()
    source = ScriptedSource(clock, [image(1)])

    def broken_area(_width, _height):
        raise KeyError("missing setting")

    window = FakeWindow()
    feed = run(make_feed(clock, spec("fake", source), area=broken_area, window=window))

    assert wait_until(window.capture_thread_ended)

    assert len(source.close_threads) == 1
    assert feed.latest() is None and feed.source == SOURCE_NONE
