"""A feliratsáv folyamatos rögzítése külön szálon: a latest() mindig a legfrissebb sávot adja, várakozás nélkül.

Források, ebben a sorrendben:
1. Windows.Graphics.Capture (window_stream.py): ~1,4 ms sávonként, másodpercenként ~27 sáv, ~5% egy magból.
2. PrintWindow egy tartós DIB-be (band_capture.PrintWindowSource), legfeljebb ~15 sáv másodpercenként:
   Windows 10-en (ott a sárga rögzítési keret nem tüntethető el) és a WGC bármely hibájánál ez lép be.
3. Ha ez sem megy, a forrás "none" és a latest() None: a hívó marad a régi úton (band_capture.BandCapture).
Egyik forrás képében sincs benne más ablak, így a saját feliratrétegünk sem.

Szálszabály: a start(), a latest(), a stop() és a source bármely szálról (a Tk-szálról) hívható; minden más
a rögzítő szálon fut – a források megnyitása, olvasása és lezárása, az area_of és az on_band hívása is.
"""

import ctypes
import ctypes.wintypes as wintypes
import logging
import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import numpy as np

from .band_capture import AreaOf, GrabbedBand, PrintWindowSource, fits
from .ocr import Box
from .window_capture import client_rect, window_exists
from .window_stream import WindowStream

SOURCE_WGC, SOURCE_PRINTWINDOW, SOURCE_NONE = "wgc", "printwindow", "none"
DEFAULT_MAX_FPS = 30.0  # a WGC-vel ez ~5% egy magból; a kétszerese kétszer annyi, ~9 ms-mal frissebb képért
PRINTWINDOW_MAX_FPS = 15.0  # egy PrintWindow ~9 ms processzoridő és 20–40 ms várakozás: sűrűbben túl drága
WGC_POLL_SECONDS = 0.004  # ilyen sűrűn kérdezzük meg, jött-e új képkocka (~0,1 ms processzoridő kérdésenként)
EARLY_FRAME_SLACK_SECONDS = 1.5 * WGC_POLL_SECONDS  # ennyivel korábban még esedékes: a kérdezés ideje ingadozik
IDLE_SECONDS = 0.1  # amíg egy forrás sem használható, ilyen sűrűn nézünk rá újra
RETRY_SECONDS = 5.0  # a hibázó forrást ennyi idő múlva próbáljuk újra
MAX_FRUITLESS_STARTS = 2  # ha egy forrás ennyiszer egymás után egy sávot sem ad, végleg kiesik (pl. Windows 10)
STALE_SECONDS = 1.0  # ha a forrás ennyi ideje hallgat (elakadt, megállt a kép, lekicsinyítették), nincs élő sáv
STOP_TIMEOUT_SECONDS = 2.0
DWMWA_EXTENDED_FRAME_BOUNDS = 9  # az ablak látható kerete (a láthatatlan átméretező szegély nélkül)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class CapturedBand:
    """Egy rögzített feliratsáv. A kép csak olvasható, mert több szál is ugyanazt a tömböt kapja."""

    image: np.ndarray  # BGR, (area.height, area.width, 3), saját memóriával
    area: Box  # a sáv helye az ablak kliensterületén
    frame_size: tuple[int, int]  # a kliensterület (szélesség, magasság), amelyre az area készült
    captured_at: float  # time.monotonic(), amikor a pixelek megérkeztek
    sequence: int  # sávonként eggyel nő
    source: str  # SOURCE_WGC vagy SOURCE_PRINTWINDOW


class BandSource(Protocol):
    """Egy megnyitott képforrás; minden metódusa a rögzítő szálon fut."""

    poll_period: float  # ha nem jött kép, az olvasás kezdete után ennyi másodperccel próbáljuk újra
    band_period: float  # ha jött, a következő olvasás ennyivel az előző kezdete után indul

    def read(self, area_of: AreaOf) -> GrabbedBand | None:
        """A legújabb sáv; None, ha most nincs új kép (ez nem hiba). Tartós hibánál OSError."""

    def close(self) -> None: ...


@dataclass(frozen=True)
class SourceSpec:
    """Egy lehetséges képforrás: a neve és a megnyitója (hibánál OSError)."""

    name: str
    open: Callable[[], BandSource]


@dataclass(frozen=True)
class FeedRuntime:
    """A rögzítő szál külvilága; a tesztek hamis órát, várakozást és ablakot adnak."""

    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep  # nem Event.wait: az Windowson 15,6 ms-os lépcsőkben vár (mérve)
    window_exists: Callable[[int], bool] = window_exists


class FramePacer:
    """Saját ritkítás, ha a rendszer nem tudja korlátozni a képkockák ütemét (Windows 11 24H2 előtt)."""

    def __init__(self, interval: float) -> None:
        self._interval = interval
        self._next_due = -math.inf

    def accept(self, now: float) -> bool:
        """Igaz, ha ezt a képkockát be kell olvasni; a túl korán érkezőt a hívó eldobja."""
        if now < self._next_due - EARLY_FRAME_SLACK_SECONDS:
            return False
        self._next_due = now + self._interval
        return True


def band_in_frame(client: Box, area: Box, frame_size: tuple[int, int]) -> Box | None:
    """A sáv a WGC-képkocka koordinátáiban; None, ha üres, vagy kilóg a kliensterületről, illetve a képkockából.

    client: a kliensterület helye a képkockán belül (a képkockában a címsor és a szegély is benne van).
    """
    if not fits(area, client.width, client.height):
        return None
    band = Box(client.left + area.left, client.top + area.top, client.left + area.right, client.top + area.bottom)
    return band if fits(band, *frame_size) else None


@lru_cache(maxsize=None)
def _dwmapi() -> ctypes.WinDLL:
    # Saját példány pontos típusokkal: a 64 bites ablakleíró csonkolás nélkül megy át
    dwmapi = ctypes.WinDLL("dwmapi")
    dwmapi.DwmGetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
    dwmapi.DwmGetWindowAttribute.restype = ctypes.c_long
    return dwmapi


def client_box_in_frame(hwnd: int) -> Box | None:
    """A kliensterület a WGC-képkocka koordinátáiban (az a látható ablakkeret bal felső sarkánál kezdődik).

    None, ha az ablak le van kicsinyítve vagy bezárták.
    """
    client, bounds = client_rect(hwnd), wintypes.RECT()
    if client is None or _dwmapi().DwmGetWindowAttribute(hwnd, DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(bounds),
                                                         ctypes.sizeof(bounds)) < 0:
        return None
    left, top = client.left - bounds.left, client.top - bounds.top
    return Box(left, top, left + client.width, top + client.height)


class StreamSource:
    """Elsődleges forrás: Windows.Graphics.Capture; a sávot a képkockán belül a kliensterülethez igazítja."""

    def __init__(self, hwnd: int, min_interval: float) -> None:
        self._hwnd = hwnd
        self._stream = WindowStream(hwnd, min_interval)
        self.poll_period = self.band_period = WGC_POLL_SECONDS
        self._pacer: FramePacer | None = None
        if self._stream.limits_rate:  # egy képkocka után legkorábban egy képkocka-köz múlva jön új: addig kár kérdezni
            self.band_period = max(WGC_POLL_SECONDS, min_interval - WGC_POLL_SECONDS)
        else:  # mi ritkítunk: a közben érkezőket is el kell kérni, különben a következő sáv régi kép lenne
            self._pacer = FramePacer(min_interval)

    def read(self, area_of: AreaOf) -> GrabbedBand | None:
        with self._stream.frame() as frame_size:
            if frame_size is None or (self._pacer is not None and not self._pacer.accept(time.monotonic())):
                return None
            client = client_box_in_frame(self._hwnd)
            if client is None:
                return None
            area = area_of(client.width, client.height)
            box = band_in_frame(client, area, frame_size)
            if box is None:
                return None
            return GrabbedBand(self._stream.read_box(box), area, (client.width, client.height))

    def close(self) -> None:
        self._stream.close()


def default_sources(hwnd: int, max_fps: float) -> tuple[SourceSpec, ...]:
    """A valódi források, előnyben részesítési sorrendben."""
    interval = 1.0 / max_fps
    slow = max(interval, 1.0 / PRINTWINDOW_MAX_FPS)
    return (SourceSpec(SOURCE_WGC, lambda: StreamSource(hwnd, interval)),
            SourceSpec(SOURCE_PRINTWINDOW, lambda: PrintWindowSource(hwnd, slow)))


class BandFeed:
    """A rögzítő szál gazdája: mindig a legfrissebb feliratsávot tartja.

        feed = BandFeed(hwnd, lambda width, height: region_to_box(region, width, height), on_band=wake)
        feed.start()
        band = feed.latest()  # CapturedBand | None; sosem vár
        feed.stop()

    area_of: a kliensterület méretéből adja a sáv helyét. A rögzítő szálon fut, ezért legyen gyors, és
    megváltoztathatatlan pillanatképből olvasson (a beállított feliratsáv futás közben változhat).
    on_band: minden új sáv után hívjuk a rögzítő szálon; legyen rövid (pl. a Tk-szál ébresztése).
    Ha az area_of vagy az on_band kivételt dob, a szál naplózza és leáll: a forrás "none", a latest() None.
    """

    def __init__(self, hwnd: int, area_of: AreaOf, *, max_fps: float = DEFAULT_MAX_FPS,
                 on_band: Callable[[CapturedBand], None] | None = None,
                 sources: Sequence[SourceSpec] | None = None, runtime: FeedRuntime = FeedRuntime()) -> None:
        self._hwnd = hwnd
        self._area_of = area_of
        self._on_band = on_band
        self._specs = tuple(default_sources(hwnd, max_fps) if sources is None else sources)
        self._runtime = runtime
        self._latest: CapturedBand | None = None
        self._source_name = SOURCE_NONE
        self._stopping = threading.Event()
        self._thread = threading.Thread(target=self._run, name="band-feed", daemon=True)
        # Innen lefelé: csak a rögzítő szál írja és olvassa
        self._active: tuple[SourceSpec, BandSource] | None = None
        self._produced = False  # az aktív forrás adott-e már sávot a megnyitása óta
        self._sequence = 0
        self._blocked_until = {spec.name: -math.inf for spec in self._specs}
        self._fruitless = {spec.name: 0 for spec in self._specs}

    def start(self) -> None:
        """Elindítja a rögzítő szálat (egyszer hívható)."""
        self._thread.start()

    def stop(self) -> None:
        """Leállítja a szálat, és megvárja, míg az a saját szálán mindent elenged. Többször is hívható."""
        self._stopping.set()
        if self._thread.is_alive():
            self._thread.join(STOP_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            logger.debug("band feed: the capture thread did not stop within %s s", STOP_TIMEOUT_SECONDS)

    def latest(self) -> CapturedBand | None:
        """A legfrissebb sáv; None, ha nincs élő kép: nincs működő forrás, vagy az STALE_SECONDS óta hallgat."""
        band = self._latest  # egyetlen hivatkozás olvasása: zár nélkül is egységes
        is_live = band is not None and self._runtime.clock() - band.captured_at <= STALE_SECONDS
        return band if is_live else None

    @property
    def source(self) -> str:
        """Az éppen használt forrás: SOURCE_WGC, SOURCE_PRINTWINDOW vagy SOURCE_NONE."""
        return self._source_name

    def _run(self) -> None:
        try:
            self._capture_until_stopped()
        except Exception:  # a szál nem halhat meg csendben; a hívó a "none" forrásból látja, hogy a régi út kell
            logger.exception("band feed: the capture thread stopped on an unexpected error")
        finally:
            self._latest = None
            self._activate(None)

    def _capture_until_stopped(self) -> None:
        while not self._stopping.is_set() and self._runtime.window_exists(self._hwnd):
            started = self._runtime.clock()
            spec = self._pick(started)
            if spec is None and all(until == math.inf for until in self._blocked_until.values()):
                return  # minden forrás végleg kiesett
            try:
                band = self._read_from(spec)
            except OSError as err:
                logger.debug("band feed: the %s source failed: %s", spec.name, err)
                self._block(spec, self._runtime.clock())
                continue
            period = IDLE_SECONDS if self._active is None else self._active[1].poll_period
            if band is not None:
                self._publish(band, spec.name)
                period = self._active[1].band_period
            remaining = started + period - self._runtime.clock()
            if remaining > 0:
                self._runtime.sleep(remaining)

    def _pick(self, now: float) -> SourceSpec | None:
        """Az első forrás a sorban, amelyik éppen nem pihen."""
        return next((spec for spec in self._specs if self._blocked_until[spec.name] <= now), None)

    def _read_from(self, spec: SourceSpec | None) -> GrabbedBand | None:
        """Olvas a megadott forrásból; ha nem az az aktív, előbb arra vált."""
        if spec is None:
            self._activate(None)
            self._latest = None  # nincs működő forrás: a hívó a régi úton rögzít
            return None
        if self._active is None or self._active[0] is not spec:
            self._activate((spec, spec.open()))  # előbb nyitunk: ha nem megy, a régi forrás marad
        return self._active[1].read(self._area_of)

    def _block(self, spec: SourceSpec, now: float) -> None:
        """A hibázó forrás pihen; ha a megnyitása óta egy sávot sem adott, néhány próba után végleg kiesik."""
        was_active = self._active is not None and self._active[0] is spec
        self._fruitless[spec.name] = 0 if was_active and self._produced else self._fruitless[spec.name] + 1
        gave_up = self._fruitless[spec.name] >= MAX_FRUITLESS_STARTS
        self._blocked_until[spec.name] = math.inf if gave_up else now + RETRY_SECONDS
        if was_active:
            self._activate(None)

    def _activate(self, active: tuple[SourceSpec, BandSource] | None) -> None:
        """Az aktív forrás cseréje (None: egyik sem); az előzőt csak az új beállítása után zárja le."""
        previous, self._active, self._produced = self._active, active, False
        self._source_name = SOURCE_NONE if active is None else active[0].name
        if previous is not None:
            try:
                previous[1].close()
            except OSError as err:
                logger.debug("band feed: closing the %s source failed: %s", previous[0].name, err)

    def _publish(self, band: GrabbedBand, source: str) -> None:
        band.image.setflags(write=False)
        self._sequence += 1
        captured = CapturedBand(band.image, band.area, band.frame_size, self._runtime.clock(), self._sequence, source)
        self._produced = True
        self._latest = captured
        if self._on_band is not None:
            self._on_band(captured)
