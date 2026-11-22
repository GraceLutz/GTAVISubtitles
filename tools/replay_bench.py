"""Offline visszajátszó tesztpad: a rögzített játékképkockák a valódi Remote Play csővezetéken át.

Futtatás (a gta_felirat csomag a PYTHONPATH-on):

    PYTHONPATH=/c/gta_felirat /c/gta_felirat/.venv/Scripts/python.exe -X utf8 tools/replay_bench.py \
        --frames <mappa> --from 06:09:15 --to 06:09:30 --out <mappa>

A képkockák (`<unix_ts>_raw.jpg`: az ablak képe a fedőréteg nélkül) ugyanazokon az osztályokon mennek át, mint
az alkalmazásban (BandWatcher → RowRecognizer → WakingRowPipeline → SubtitleLayer.plan → LayerCompositor),
hamis órával, amelyet a képkockák időbélyege hajt. A kimenet az, amit a néző látna: a nyers kép, rajta a
feliratréteg (a satírozott takarás és a magyar szöveg).

Ami hamis: az óra, a végrehajtók (fordítás és felismerés késleltetése a hamis órán), és a fordító (a
mentett fordításokból olvas, különben "[HU] " + szöveg). Hálózatot, API-kulcsot és a Windows
hitelesítőadat-tárat a tesztpad nem érinti, a config.json-t és a mentett fordításokat csak olvassa.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import ctypes
import dataclasses
import datetime as dt
import logging
import os
import re
import statistics
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from gta_felirat import row_path as row_path_module
from gta_felirat import speaker_names
from gta_felirat import subtitle_pipeline as subtitle_pipeline_module
from gta_felirat.config import AppConfig, ConfigError, load_config
from gta_felirat.frosted_cover import Backdrop
from gta_felirat.handoff import Inbox, LatestBox
from gta_felirat.i18n import set_language
from gta_felirat.layer_compositor import LayerCompositor
from gta_felirat.layer_plan import CoverJob, LayerPlan, Shown
from gta_felirat.ocr import Box, SubtitleReader
from gta_felirat.remote_play import effective_hide_original
from gta_felirat.row_path import RowPath, WakingRowPipeline
from gta_felirat.band_feed import SOURCE_WGC, CapturedBand
from gta_felirat.subtitle_layer import SubtitleLayer
from gta_felirat.translation_service import TranslationService, choice_from
from gta_felirat.translation_store import TranslationStore
from gta_felirat.translator import CachedTranslator

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_DIR / "config.json"
DEFAULT_STORE = PROJECT_DIR / "mentett_forditasok.json"
RAW_SUFFIX = "_raw.jpg"
SCREEN_SUFFIX = "_screen.jpg"
PLACEHOLDER_PREFIX = "[HU] "
BELOW_NORMAL_PRIORITY_CLASS = 0x4000
DEFAULT_TRANSLATION_DELAY = 0.3  # hamis órán mért másodperc; a gyorsítótárból érkező fordításnak nincs késése
DEFAULT_OCR_DELAY = 0.2  # a sorfelismerő szál egy olvasásának ideje (becsült; a kód szerint terhelten 0,5-1 mp)
TICK_SECONDS = 0.015  # az alkalmazás rendes időzítőjének üteme (remote_play.TICK_MS)
FEED_STALE_SECONDS = 1.0  # ennyi csend után a folyamatos rögzítésnek nincs élő sávja (band_feed.STALE_SECONDS)
MAX_IDLE_TICK_SECONDS = 3.0  # ennél hosszabb kihagyott szakaszt nem léptetünk végig 15 ms-onként
DEFAULT_WARMUP_SECONDS = 4.0
DEFAULT_UTC_OFFSET_HOURS = 1.0
OUTPUT_BOTTOM_RATIO = 0.25  # a kimeneti képen a kép alsó negyede látszik
JPEG_QUALITY = 90
DEFAULT_WORST_COUNT = 15

# --- szivárgásmérés: a felirat betűi a nyers képen, a tesztpad saját (a takarástól független) mércéjével ---
WHITE_LEVEL = 190  # a hófehér betű mindhárom csatornája legalább ennyi
NAME_LEVEL = 120  # a színes beszélőnév legvilágosabb csatornája legalább ennyi
NAME_SPREAD = 30  # és a csatornái legalább ennyire eltérnek
NAME_MAX_STROKE_PX = 7  # ennél vastagabb színes folt nem betű (1080p-n)
OUTLINE_LEVEL = 150  # a betűt sötét körvonal veszi körül: a közelében van ennél sötétebb pixel
OUTLINE_REACH_PX = 4
COVER_ALPHA_MIN = 128  # ennél áttetszőbb takarás alatt a betű "kilátszik"
DEFAULT_LEAK_THRESHOLD = 12  # ennél több kilátszó betűpixelnél jelöljük a képkockát

logger = logging.getLogger("replay_bench")


# =====================================================================================================
# Időkezelés és képkockák
# =====================================================================================================
@dataclass(frozen=True)
class FrameFile:
    timestamp: float  # unix másodperc, a fájlnévből
    path: Path


def parse_clock(text: str) -> float:
    """"ÓÓ:PP", "ÓÓ:PP:MM" vagy "ÓÓ:PP:MM.tt" → a nap eleje óta eltelt másodpercek."""
    match = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2})(\.\d+)?)?", text.strip())
    if match is None:
        raise ValueError(f"érvénytelen időpont: {text!r} (várt alak: ÓÓ:PP:MM[.tt])")
    hours, minutes, seconds, fraction = match.groups()
    total = int(hours) * 3600 + int(minutes) * 60 + int(seconds or 0) + float(fraction or 0)
    if int(hours) > 23 or int(minutes) > 59 or int(seconds or 0) > 59:
        raise ValueError(f"érvénytelen időpont: {text!r}")
    return total


def frame_timestamp(name: str) -> float | None:
    """A "<unix_ts>_raw.jpg" fájlnév időbélyege; None, ha nem ilyen a név."""
    if not name.endswith(RAW_SUFFIX):
        return None
    try:
        return float(name[:-len(RAW_SUFFIX)])
    except ValueError:
        return None


def list_frames(folder: Path) -> list[FrameFile]:
    """A mappa nyers képkockái időrendben (egyszeri listázás: a mappában több tízezer fájl van)."""
    frames = []
    with os.scandir(folder) as entries:
        for entry in entries:
            timestamp = frame_timestamp(entry.name)
            if timestamp is not None:
                frames.append(FrameFile(timestamp, Path(entry.path)))
    frames.sort(key=lambda frame: frame.timestamp)
    return frames


def local_midnight(timestamp: float, utc_offset_hours: float, date: dt.date | None = None) -> float:
    """A helyi nap kezdete (unix másodperc); a dátum nélkül az időbélyeg napja."""
    zone = dt.timezone(dt.timedelta(hours=utc_offset_hours))
    day = date or dt.datetime.fromtimestamp(timestamp, zone).date()
    return dt.datetime(day.year, day.month, day.day, tzinfo=zone).timestamp()


def select_range(frames: list[FrameFile], start: float, end: float) -> list[FrameFile]:
    """A [start, end] időbélyegű képkockák (a lista időrendben van)."""
    stamps = [frame.timestamp for frame in frames]
    return frames[bisect.bisect_left(stamps, start):bisect.bisect_right(stamps, end)]


def format_clock(timestamp: float, utc_offset_hours: float) -> str:
    """Helyi idő "ÓÓ-PP-MM.tt" alakban (fájlnévbe is jó)."""
    zone = dt.timezone(dt.timedelta(hours=utc_offset_hours))
    moment = dt.datetime.fromtimestamp(timestamp, zone)
    return f"{moment:%H-%M-%S}.{int(moment.microsecond / 10000):02d}"


# =====================================================================================================
# Offline fordító
# =====================================================================================================
class ReadOnlyStore(TranslationStore):
    """A mentett fordítások csak olvasva: a lemezre semmit sem ír. Névtértől függetlenül is keres.

    Elsőként a kért névtérben (pl. "deepl:hu"), utána a többiben, a bejegyzések számának sorrendjében.
    """

    def put_many(self, namespace, items) -> None:
        """Szándékosan nem ír: sem a memóriába, sem a lemezre."""

    def get(self, namespace: str, key: str) -> str | None:
        lookup = super().get
        found = lookup(namespace, key)
        if found is not None:
            return found
        others = sorted(self._entries, key=lambda name: -len(self._entries[name]))
        return next((value for name in others if (value := lookup(name, key)) is not None), None)


class PlaceholderBackend:
    """Hálózat nélküli "szolgáltatás": a szöveg elé teszi a "[HU] " jelölést. Számolja a hívásokat."""

    def __init__(self) -> None:
        self.calls = 0

    def translate(self, text: str) -> str:
        self.calls += 1
        return PLACEHOLDER_PREFIX + text


def make_offline_translator(store_path: Path, namespace: str) -> tuple[CachedTranslator, PlaceholderBackend]:
    """A valódi CachedTranslator (beszélőnév-megtartással, soronkénti gyorsítótárral), offline háttérrel."""
    backend = PlaceholderBackend()
    store = ReadOnlyStore(store_path)
    return CachedTranslator(backend, store=store, namespace=namespace), backend


# =====================================================================================================
# Hamis óra és végrehajtók
# =====================================================================================================
class BenchClock:
    """A hamis óra: a pipeline `now` értékével azonos időalap (a képkockák időbélyege)."""

    def __init__(self) -> None:
        self.now = 0.0


class ClockedTime:
    """A `time` modul helyett a pipeline-ban: a monotonic() a hamis órát adja, a többi a valódi modulból."""

    def __init__(self, clock: BenchClock) -> None:
        self._clock = clock

    def monotonic(self) -> float:
        return self._clock.now

    def __getattr__(self, name: str):
        return getattr(time, name)


class ManualFuture:
    """Késleltetve, első használatkor lefutó feladat: a hamis órán csak `ready_at` után "kész" (done).

    A feladat nem a submit()-ban fut (a hívó ott még zárat foghat, amelyet a feladat is kérne), hanem az
    első done()/result()/advance()-nál; a késés ekkor dől el: hálózati hívásnál a beadás + késés.
    """

    def __init__(self, executor: ManualExecutor, fn, args, kwargs) -> None:
        self._executor = executor
        self._task = (fn, args, kwargs)
        self.submitted_at = executor.clock.now
        self.ready_at = float("inf")
        self._ran = False
        self._value = None
        self._error: BaseException | None = None
        self._callbacks: list[Callable[[ManualFuture], None]] = []
        self.fired = False

    def _run(self) -> None:
        if self._ran:
            return
        self._ran = True
        fn, args, kwargs = self._task
        self.ready_at = self._executor.run_task(self, fn, args, kwargs)

    def done(self) -> bool:
        self._run()
        return self._executor.clock.now >= self.ready_at

    def result(self, timeout: float | None = None):
        self._run()
        if self._error is not None:
            raise self._error
        return self._value

    def exception(self, timeout: float | None = None) -> BaseException | None:
        self._run()
        return self._error

    def cancel(self) -> bool:
        return False

    def add_done_callback(self, callback: Callable[[ManualFuture], None]) -> None:
        if self.fired:
            callback(self)
        else:
            self._callbacks.append(callback)

    def fire_if_ready(self) -> None:
        if self.fired or not self.done():
            return
        self.fired = True
        callbacks, self._callbacks = self._callbacks, []
        for callback in callbacks:
            callback(self)


class ManualExecutor:
    """Szálak nélküli végrehajtó: a feladat késleltetve, a hívó szálán fut, az eredmény a hamis órán késve "érkezik".

    A késés csak hálózati hívásnál van (a PlaceholderBackend hívásszámából látszik): a mentett vagy
    gyorsítótárból jövő fordítás azonnal kész. Ugyanarra a szövegre indított második kérés (az előre
    indított fordítás, utána a megerősített) az első érkezésére vár, mint az alkalmazásban.
    """

    def __init__(self, clock: BenchClock, delay: float, network_calls: Callable[[], int]) -> None:
        self.clock = clock
        self._delay = delay
        self._network_calls = network_calls
        self._ready_by_text: dict[str, float] = {}
        self._futures: list[ManualFuture] = []

    def submit(self, fn, *args, **kwargs) -> ManualFuture:
        future = ManualFuture(self, fn, args, kwargs)
        self._futures.append(future)
        return future

    def run_task(self, future: ManualFuture, fn, args, kwargs) -> float:
        """Lefuttatja a feladatot; a visszaadott érték: mikor érkezik meg az eredménye."""
        before = self._network_calls()
        try:
            future._value = fn(*args, **kwargs)
        except Exception as err:  # a valódi Future is a hibát tárolja, és az eredménynél dobja
            future._error = err
        used_network = self._network_calls() != before
        ready_at = future.submitted_at + (self._delay if used_network else 0.0)
        text = args[0] if args and isinstance(args[0], str) else None
        return ready_at if text is None else self._ready_by_text.setdefault(text, ready_at)

    def advance(self) -> None:
        """A hamis óra lépett: a most kész futurek visszahívásai lefutnak."""
        for future in list(self._futures):
            future.fire_if_ready()
        self._futures = [future for future in self._futures if not future.fired]

    def shutdown(self, wait: bool = False, cancel_futures: bool = False) -> None:
        self._futures.clear()


@contextmanager
def patched_pipeline_environment(clock: BenchClock, executors: list[ManualExecutor]) -> Iterator[None]:
    """A pipeline szálak és valódi idő nélkül fusson: hamis végrehajtók és hamis monotonic().

    A subtitle_pipeline modulban cseréli a ThreadPoolExecutor-t és a time-ot; kilépéskor visszaállítja.
    """
    module = subtitle_pipeline_module
    original_pool, original_time = module.ThreadPoolExecutor, module.time
    module.ThreadPoolExecutor = lambda *args, **kwargs: executors.pop(0)
    module.time = ClockedTime(clock)
    try:
        yield
    finally:
        module.ThreadPoolExecutor, module.time = original_pool, original_time


# =====================================================================================================
# A felismerő szál és a postaláda modellje
# =====================================================================================================
class DelayedInbox(Inbox):
    """Inbox, amelybe a felismert szövegek a felismerés idejével késve kerülnek (a pillanatkép azonnal)."""

    def __init__(self, wake: Callable[[], None], clock: BenchClock, delay: float) -> None:
        super().__init__(wake)
        self._clock = clock
        self._delay = delay
        self._due: list[tuple[float, tuple]] = []

    def put_texts(self, texts) -> None:
        self._due.append((self._clock.now + self._delay, tuple(texts)))

    def release(self) -> None:
        """A hamis órán már megérkezett szövegek a postaládába kerülnek."""
        ready = [texts for moment, texts in self._due if moment <= self._clock.now]
        self._due = [(moment, texts) for moment, texts in self._due if moment > self._clock.now]
        for texts in ready:
            super().put_texts(texts)


class PollableBox(LatestBox):
    """LatestBox várakozás nélküli kivétellel: a hamis felismerő szál innen veszi a legfrissebb feladatot."""

    def pop(self):
        with self._changed:
            item, self._item = self._item, None
            return item


class SimulatedRecognizer:
    """Egyetlen felismerő szál: a legfrissebb pillanatképet olvassa, olvasásonként `delay` hamis másodpercig."""

    def __init__(self, clock: BenchClock, delay: float) -> None:
        self._clock = clock
        self._delay = delay
        self._free_at = -float("inf")
        self._box: PollableBox | None = None
        self._handle: Callable | None = None
        self.reads = 0
        self.wall_seconds = 0.0  # a valódi OCR összidő (tájékoztatás: mennyibe kerül egy olvasás)

    def attach(self, box: PollableBox, handle: Callable) -> None:
        self._box, self._handle = box, handle

    def service(self) -> None:
        """Ha a szál szabad és van feladat, azonnal elkezdi: az eredmény a postaládában késik."""
        if self._box is None or self._clock.now < self._free_at:
            return
        job = self._box.pop()
        if job is None:
            return
        started = time.perf_counter()
        self._handle(job)
        self.wall_seconds += time.perf_counter() - started
        self.reads += 1
        self._free_at = self._clock.now + self._delay


@contextmanager
def patched_row_path(recognizer: SimulatedRecognizer, clock: BenchClock, delay: float) -> Iterator[None]:
    """A RowPath szálak nélkül épüljön fel: saját postaláda, nem blokkoló feladatláda, hamis felismerő szál."""
    module = row_path_module
    saved = module.Inbox, module.LatestBox, module.start_worker

    def start_worker(name, box, handle):
        recognizer.attach(box, handle)
        return None

    module.Inbox = lambda wake: DelayedInbox(wake, clock, delay)
    module.LatestBox = PollableBox
    module.start_worker = start_worker
    try:
        yield
    finally:
        module.Inbox, module.LatestBox, module.start_worker = saved


# =====================================================================================================
# A csővezeték
# =====================================================================================================
@dataclass
class PipelineState:
    """A csővezeték állása egy lépés után: amit a megjelenítés és a mérés lát."""

    shown: Shown | None
    backdrop: Backdrop | None
    live_lines: tuple
    early_boxes: tuple[Box, ...]  # a még olvasatlan, feliratszerű sorok dobozai


class ReplayPipeline:
    """A valódi RowPath + WakingRowPipeline, szálak nélkül, hamis órával."""

    def __init__(self, config: AppConfig, store_path: Path, *, delay: float, ocr_delay: float,
                 tick_seconds: float = TICK_SECONDS) -> None:
        self.config = config
        self.clock = BenchClock()
        self.tick_seconds = tick_seconds
        namespace = f"{config.translator}:{config.target_lang}"
        translator, self.backend = make_offline_translator(store_path, namespace)
        # Kulcs, hitelesítőadat-tár és lemezes tároló nélkül: a fordító már kész, a szolgáltatás csak fut vele
        service = TranslationService(choice_from(config), translator, store=None, vault=None)
        translate = ManualExecutor(self.clock, delay, lambda: self.backend.calls)
        ocr = ManualExecutor(self.clock, 0.0, lambda: 0)
        self.translate_executor = translate
        speaker_names.use_config(config.speaker_names, None)  # mint az alkalmazás, de mentés nélkül
        reader = SubtitleReader(config.subtitle_region, config.min_text_score)
        self.recognizer = SimulatedRecognizer(self.clock, ocr_delay)
        # A hamis óra és végrehajtók a csővezeték élettartamára cserélődnek (a close() állítja vissza);
        # a SubtitlePipeline először az OCR-, aztán a fordítókészletet kéri
        self._patches = ExitStack()
        self._patches.enter_context(patched_pipeline_environment(self.clock, [ocr, translate]))
        with patched_row_path(self.recognizer, self.clock, ocr_delay):
            self.pipeline = WakingRowPipeline(config, reader, service, wake=lambda: None)
            self.row_path = RowPath(self.pipeline, reader.read_rows, config.min_text_score, lambda: None,
                                    whole_band=config.whole_band_text)
        self._sequence = 0
        self._last_frame_at: float | None = None
        self._frame_size: tuple[int, int] | None = None

    def close(self) -> None:
        self.row_path.close()
        self.pipeline.shutdown()
        self._patches.close()

    def feed(self, frame: np.ndarray, timestamp: float) -> PipelineState:
        """Egy rögzített képkocka: a közbeeső időzítő-lépések, majd a képkocka sávja és a lépés."""
        height, width = frame.shape[:2]
        self._frame_size = (width, height)
        if not self.row_path.can_read(width):
            raise RuntimeError(f"a soralapú olvasás nem használható {width}x{height} képen")
        self._run_ticks_until(timestamp)
        self.clock.now = timestamp
        self.row_path.on_band(self._band(frame, timestamp))
        self._last_frame_at = timestamp
        return self._step()

    def _band(self, frame: np.ndarray, timestamp: float) -> CapturedBand:
        height, width = frame.shape[:2]
        area = self.pipeline.band_area(width, height)
        image = np.ascontiguousarray(frame[area.top:area.bottom, area.left:area.right]).copy()
        image.setflags(write=False)
        self._sequence += 1
        return CapturedBand(image, area, (width, height), timestamp, self._sequence, SOURCE_WGC)

    def _run_ticks_until(self, timestamp: float) -> None:
        """Az alkalmazás rendes időzítője a két képkocka között (lejáró takarások, késő fordítás)."""
        last = self._last_frame_at
        if last is None:
            return
        if timestamp - last > FEED_STALE_SECONDS:
            self.row_path.suspend()  # nem jött kép: a folyamatos rögzítésnek nincs élő sávja, a sorok elévülnek
            self.clock.now = timestamp - self.tick_seconds
            return
        moment = last + self.tick_seconds
        while moment < timestamp:
            self.clock.now = moment
            self._step()
            moment += self.tick_seconds

    def _step(self) -> PipelineState:
        """Az alkalmazás _step()-je a soralapú úton: a munkaszálak eredménye, a pipeline, a kirajzolás állapota."""
        recognizer = self.recognizer
        recognizer.service()
        self.row_path._inbox.release()
        self.translate_executor.advance()
        self.row_path.pump(self.clock.now, self._frame_size)
        self.pipeline.poll(self.clock.now)
        early = tuple(row.box for row in self.pipeline._rows if row.row_id not in self.pipeline._texts
                      and self.pipeline._can_hide_early(row))
        return PipelineState(self.pipeline.shown, self.row_path.backdrop(), self.pipeline._live_lines, early)


# =====================================================================================================
# Kirajzolás
# =====================================================================================================
class FrameRenderer:
    """A feliratréteg kirajzolása ugyanazzal a tervezővel és festővel, mint az átlátszó ablak.

    Két festő fut: az egyik a teljes réteget rajzolja (ez kerül a képre), a másik csak a takarásokat
    (ennek az áttetszősége mondja meg, hol takarja az angol betűket valami).
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._layer = SubtitleLayer(config, on_change=lambda *_: None)
        self._full = LayerCompositor()
        self._covers = LayerCompositor()
        self._full_surface: np.ndarray | None = None
        self._cover_surface: np.ndarray | None = None

    def render(self, frame: np.ndarray, state: PipelineState) -> tuple[np.ndarray, np.ndarray, LayerPlan | None]:
        """(a kép a réteggel, a takarások áttetszősége H×W, a terv)."""
        height, width = frame.shape[:2]
        if self._full_surface is None or self._full_surface.shape[:2] != (height, width):
            self._full_surface = np.zeros((height, width, 4), np.uint8)
            self._cover_surface = np.zeros((height, width, 4), np.uint8)
            self._full.reset()
            self._covers.reset()
        hide = effective_hide_original(self._config, ocr_sees_overlay=False)
        plan = self._layer.plan((width, height), state.shown, hide, None, state.backdrop)
        self._full.render(self._full_surface, plan, state.backdrop)
        covers_only = dataclasses.replace(plan, panel=None, texts=(), outline=None) if plan is not None else None
        self._covers.render(self._cover_surface, covers_only, state.backdrop)
        return composite_premultiplied(frame, self._full_surface), self._cover_surface[:, :, 3], plan


def composite_premultiplied(frame: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """BGR kép + előszorzott BGRA réteg: kimenet = réteg + kép · (255 − alfa) / 255."""
    keep = cv2.bitwise_not(layer[:, :, 3])
    keep3 = cv2.merge((keep, keep, keep))
    return cv2.add(cv2.multiply(frame, keep3, scale=1.0 / 255.0), np.ascontiguousarray(layer[:, :, :3]))


# =====================================================================================================
# Szivárgásmérés
# =====================================================================================================
@dataclass(frozen=True)
class LeakCounts:
    left: int = 0
    middle: int = 0
    right: int = 0
    text: int = 0  # az összes betűpixel a dobozokban (takarással együtt)

    @property
    def total(self) -> int:
        return self.left + self.middle + self.right

    def __add__(self, other: LeakCounts) -> LeakCounts:
        return LeakCounts(self.left + other.left, self.middle + other.middle, self.right + other.right,
                          self.text + other.text)


def _disc(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


_OUTLINE_KERNEL = _disc(OUTLINE_REACH_PX)


def text_pixels(image: np.ndarray) -> np.ndarray:
    """A feliratbetűnek látszó pixelek (bool H×W): fehér vagy telített színes, sötét körvonallal a közelben.

    A tesztpad saját mércéje, nem a takarás betűmaszkja: így nem körkörös (amit a takarás kihagy, azt is látja).
    A vastag színes folt (hátsó lámpa, neon) nem betű.
    """
    darkest = image.min(axis=2)
    brightest = image.max(axis=2)
    has_dark_nearby = cv2.erode(brightest, _OUTLINE_KERNEL) <= OUTLINE_LEVEL
    white = darkest >= WHITE_LEVEL
    spread = brightest.astype(np.int16) - darkest.astype(np.int16)
    coloured = ((brightest >= NAME_LEVEL) & (spread >= NAME_SPREAD)).astype(np.uint8)
    stroke = max(NAME_MAX_STROKE_PX, image.shape[0] // 11) | 1
    thick = cv2.morphologyEx(coloured, cv2.MORPH_OPEN, _disc(stroke // 2))
    name = (coloured > 0) & (cv2.dilate(thick, _disc(stroke)) == 0)
    return (white | name) & has_dark_nearby


def leak_counts(frame: np.ndarray, cover_alpha: np.ndarray, boxes: list[Box]) -> LeakCounts:
    """Az angol sorok dobozaiban a nem takart betűpixelek, soronként bal/közép/jobb harmadra bontva."""
    height, width = frame.shape[:2]
    total = LeakCounts()
    for box in boxes:
        left, top = max(box.left, 0), max(box.top, 0)
        right, bottom = min(box.right, width), min(box.bottom, height)
        if right - left < 3 or bottom - top < 1:
            continue
        pad = OUTLINE_REACH_PX
        crop_top, crop_left = max(top - pad, 0), max(left - pad, 0)
        crop = frame[crop_top:min(bottom + pad, height), crop_left:min(right + pad, width)]
        letters = text_pixels(crop)[top - crop_top:bottom - crop_top, left - crop_left:right - crop_left]
        uncovered = letters & (cover_alpha[top:bottom, left:right] < COVER_ALPHA_MIN)
        edges = [(right - left) * part // 3 for part in range(4)]
        left_px, middle_px, right_px = (int(uncovered[:, edges[i]:edges[i + 1]].sum()) for i in range(3))
        total += LeakCounts(left_px, middle_px, right_px, int(letters.sum()))
    return total


def distinct_boxes(boxes: list[Box]) -> list[Box]:
    """A (közel) azonos vagy egymásba foglalt dobozok csak egyszer: ne számoljunk kétszer egy sort."""
    kept: list[Box] = []
    for box in sorted(boxes, key=lambda b: -(b.width * b.height)):
        if not any(_mostly_inside(box, other) for other in kept):
            kept.append(box)
    return kept


def _mostly_inside(inner: Box, outer: Box, ratio: float = 0.8) -> bool:
    overlap_w = min(inner.right, outer.right) - max(inner.left, outer.left)
    overlap_h = min(inner.bottom, outer.bottom) - max(inner.top, outer.top)
    if overlap_w <= 0 or overlap_h <= 0:
        return False
    return overlap_w * overlap_h >= ratio * inner.width * inner.height


def line_boxes(state: PipelineState) -> list[Box]:
    """Az angol sorok, amelyeket a csővezeték lát: a felismert feliratsorok és a még olvasatlan, feliratszerű sorok."""
    return distinct_boxes([line.box for line in state.live_lines] + list(state.early_boxes))


# =====================================================================================================
# Mérések és kimenet
# =====================================================================================================
@dataclass
class FrameRecord:
    clock: str
    timestamp: float
    gap_seconds: float
    shown_text: str
    english: str
    cover_count: int
    covers: str
    text_count: int
    texts: str
    hidden_count: int
    leak: LeakCounts
    image_name: str = ""
    flagged: bool = False


CSV_COLUMNS = ("time", "timestamp", "gap_s", "shown_text", "english_lines", "cover_count", "covers", "text_count",
               "text_boxes", "hidden_count", "text_px", "leak_left", "leak_middle", "leak_right", "leak_total",
               "flagged", "image")


def box_text(box: Box) -> str:
    return f"{box.left},{box.top},{box.right},{box.bottom}"


def describe_plan(plan: LayerPlan | None) -> tuple[int, str, int, str]:
    """(takarások száma, dobozaik, szövegek száma, dobozaik)."""
    if plan is None:
        return 0, "", 0, ""
    covers = [job.area if isinstance(job, CoverJob) else job.box for job in plan.covers]
    texts = [f"{job.x},{job.y},{job.x + job.rendered.width},{job.y + job.rendered.height}" for job in plan.texts]
    return len(covers), ";".join(box_text(box) for box in covers), len(texts), ";".join(texts)


def make_record(clock: str, timestamp: float, gap: float, state: PipelineState, plan: LayerPlan | None,
                leak: LeakCounts, threshold: int) -> FrameRecord:
    cover_count, covers, text_count, texts = describe_plan(plan)
    english = " | ".join(f"{line.text.replace(chr(10), ' ')} @{box_text(line.box)}" for line in state.live_lines)
    shown = state.shown
    return FrameRecord(clock, timestamp, gap, (shown.text if shown else "").replace("\n", " / "), english,
                       cover_count, covers, text_count, texts, len(shown.hidden) if shown else 0, leak,
                       flagged=leak.total > threshold)


def record_row(record: FrameRecord) -> list:
    leak = record.leak
    return [record.clock, f"{record.timestamp:.2f}", f"{record.gap_seconds:.2f}", record.shown_text, record.english,
            record.cover_count, record.covers, record.text_count, record.texts, record.hidden_count, leak.text,
            leak.left, leak.middle, leak.right, leak.total, int(record.flagged), record.image_name]


def write_metrics(path: Path, records: list[FrameRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(record_row(record) for record in records)


def write_summary(path: Path, records: list[FrameRecord], args: argparse.Namespace, extra: list[str]) -> str:
    """A summary.txt tartalma: a futás adatai, az alapszint és a legrosszabb képkockák."""
    lines = [f"Replay bench: {args.from_time} - {args.to_time}, {len(records)} képkocka, leak-küszöb {args.leak_threshold}",
             f"fordítási késés {args.delay} s, felismerési késés {args.ocr_delay} s (hamis órán)", *extra]
    with_lines = [r for r in records if r.english]
    covered = [r for r in with_lines if r.cover_count]
    lines.append(f"angol sort látó képkocka: {len(with_lines)}, ebből takarással: {len(covered)}, "
                 f"takarás nélkül: {len(with_lines) - len(covered)}")
    if with_lines:
        leaks = [r.leak.total for r in with_lines]
        lines.append(f"szivárgás (angol sort látó képkockák): medián {statistics.median(leaks):.0f}, "
                     f"90. percentilis {np.percentile(leaks, 90):.0f}, max {max(leaks)}")
    flagged = [r for r in records if r.flagged]
    lines.append(f"jelölt képkockák (szivárgás > {args.leak_threshold}): {len(flagged)}")
    lines.append("")
    lines.append(f"A {args.worst} legrosszabb képkocka (bal/közép/jobb betűpixel takarás nélkül):")
    for record in sorted(records, key=lambda r: -r.leak.total)[:args.worst]:
        leak = record.leak
        lines.append(f"  {record.clock}  összes {leak.total:4d} (b {leak.left}, k {leak.middle}, j {leak.right}) "
                     f"takarás {record.cover_count}  {record.shown_text[:60]!r}  {record.image_name}")
    content = "\n".join(lines) + "\n"
    path.write_text(content, encoding="utf-8")
    return content


# =====================================================================================================
# Futtatás
# =====================================================================================================
def lower_process_priority() -> None:
    """Alacsony prioritás: a felhasználó gépe közben használatban lehet."""
    if sys.platform != "win32":
        return
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline visszajátszó tesztpad a feliratréteghez")
    parser.add_argument("--frames", type=Path, required=True, help="a <unix_ts>_raw.jpg képkockák mappája")
    parser.add_argument("--from", dest="from_time", required=True, help="helyi idő ÓÓ:PP:MM")
    parser.add_argument("--to", dest="to_time", required=True, help="helyi idő ÓÓ:PP:MM")
    parser.add_argument("--out", type=Path, required=True, help="kimeneti mappa")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="config.json (csak olvassuk)")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE, help="mentett fordítások (csak olvassuk)")
    parser.add_argument("--delay", type=float, default=DEFAULT_TRANSLATION_DELAY,
                        help="a fordítás késése hamis órán, másodperc")
    parser.add_argument("--ocr-delay", type=float, default=DEFAULT_OCR_DELAY,
                        help="egy sorolvasás ideje hamis órán, másodperc")
    parser.add_argument("--warmup", type=float, default=DEFAULT_WARMUP_SECONDS,
                        help="ennyi másodperc képkockát a --from előtt is feldolgozunk, kimenet nélkül")
    parser.add_argument("--leak-threshold", type=int, default=DEFAULT_LEAK_THRESHOLD)
    parser.add_argument("--utc-offset", type=float, default=DEFAULT_UTC_OFFSET_HOURS, help="a helyi idő eltolása UTC-hez")
    parser.add_argument("--date", type=dt.date.fromisoformat, help="a nap (ÉÉÉÉ-HH-NN); alapból az első képkockáé")
    parser.add_argument("--worst", type=int, default=DEFAULT_WORST_COUNT, help="a summary.txt-ben listázott képkockák")
    parser.add_argument("--compare", action="store_true", help="_cmp.jpg: a tesztpad képe alatt a valódi _screen kép")
    parser.add_argument("--no-images", action="store_true", help="csak metrics.csv és summary.txt")
    return parser.parse_args(argv)


def comparison_image(bench_crop: np.ndarray, frame_path: Path) -> np.ndarray | None:
    """A tesztpad kivágata alatt a valódi, rögzített képernyő (a _screen.jpg alsó negyede); None, ha nincs."""
    screen = cv2.imread(str(frame_path.with_name(frame_path.name[:-len(RAW_SUFFIX)] + SCREEN_SUFFIX)))
    if screen is None:
        return None
    rows = bench_crop.shape[0]
    real = screen[max(screen.shape[0] - rows, 0):]
    real = cv2.resize(real, (bench_crop.shape[1], rows)) if real.shape[1] != bench_crop.shape[1] else real
    return np.vstack([bench_crop, real])


def process_frame(pipeline: ReplayPipeline, renderer: FrameRenderer, frame_file: FrameFile, frame: np.ndarray):
    """(állapot, kép a réteggel, takarások alfája, terv) egy képkockára."""
    state = pipeline.feed(frame, frame_file.timestamp)
    composite, cover_alpha, plan = renderer.render(frame, state)
    return state, composite, cover_alpha, plan


def run(args: argparse.Namespace) -> int:
    frames = list_frames(args.frames)
    if not frames:
        print(f"nincs *{RAW_SUFFIX} képkocka itt: {args.frames}", file=sys.stderr)
        return 2
    midnight = local_midnight(frames[0].timestamp, args.utc_offset, args.date)
    start, end = midnight + parse_clock(args.from_time), midnight + parse_clock(args.to_time)
    selected = select_range(frames, start - args.warmup, end)
    if not any(f.timestamp >= start for f in selected):
        print("a megadott időszakban nincs képkocka", file=sys.stderr)
        return 2
    config = load_config(args.config)  # csak olvasás
    set_language(config.ui_language)
    args.out.mkdir(parents=True, exist_ok=True)
    pipeline = ReplayPipeline(config, args.store, delay=args.delay, ocr_delay=args.ocr_delay)
    renderer = FrameRenderer(config)
    records: list[FrameRecord] = []
    notes: list[str] = []
    previous = None
    try:
        for index, frame_file in enumerate(selected):
            frame = cv2.imread(str(frame_file.path))
            if frame is None:
                notes.append(f"nem olvasható képkocka kihagyva: {frame_file.path.name}")
                continue
            gap = 0.0 if previous is None else frame_file.timestamp - previous
            previous = frame_file.timestamp
            state, composite, cover_alpha, plan = process_frame(pipeline, renderer, frame_file, frame)
            if frame_file.timestamp < start:
                continue  # bemelegítés: a csővezeték állapota felépül, de nem mérünk
            if gap > FEED_STALE_SECONDS:
                notes.append(f"{format_clock(frame_file.timestamp, args.utc_offset)}: {gap:.1f} s hézag a felvételben")
            leak = leak_counts(frame, cover_alpha, line_boxes(state))
            clock = format_clock(frame_file.timestamp, args.utc_offset)
            record = make_record(clock, frame_file.timestamp, gap, state, plan, leak, args.leak_threshold)
            if not args.no_images:
                crop = composite[int(composite.shape[0] * (1 - OUTPUT_BOTTOM_RATIO)):]
                record.image_name = f"{clock}_bench.jpg"
                cv2.imwrite(str(args.out / record.image_name), crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if args.compare and (both := comparison_image(crop, frame_file.path)) is not None:
                    cv2.imwrite(str(args.out / f"{clock}_cmp.jpg"), both, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            records.append(record)
            if len(records) % 50 == 0:
                print(f"  {len(records)} képkocka kész ({clock})", flush=True)
        recognizer = pipeline.recognizer
        notes.append(f"sorfelismerés: {recognizer.reads} olvasás, összesen {recognizer.wall_seconds:.1f} s valódi idő; "
                     f"fordítószolgáltatás-hívás (placeholder): {pipeline.backend.calls}")
    finally:
        pipeline.close()
    write_metrics(args.out / "metrics.csv", records)
    print(write_summary(args.out / "summary.txt", records, args, notes))
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    lower_process_priority()
    cv2.setNumThreads(1)  # mint az alkalmazás: ezekhez a kis képekhez a munkaszálak csak processzort égetnek
    try:
        return run(args)
    except (ConfigError, ValueError) as err:
        print(f"hiba: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
