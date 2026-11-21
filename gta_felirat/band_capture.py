"""A feliratsáv rögzítése közvetlenül a Remote Play ablakból (PrintWindow), a fölötte lévő ablakok nélkül.

Így a feliratrétegünk (fordítás, az angol szöveg takarása) nem kerül bele abba, amit az OCR
olvas, miközben a Medal/OBS felvételen látszik. Ha az ablak tartalma nem rögzíthető (a
PrintWindow hibát ad, vagy tartósan fekete képet, miközben a képernyőn látszik a játék),
a képernyőről rögzítünk tovább (mss), mint korábban.
"""

import ctypes
import ctypes.wintypes as wintypes
import logging
import weakref
from collections.abc import Callable
from typing import NamedTuple, Protocol

import cv2
import numpy as np

from .i18n import t
from .ocr import Box
from .window_capture import ScreenRect, client_rect

PW_CLIENTONLY = 0x1
PW_RENDERFULLCONTENT = 0x2  # a hardveresen gyorsított (videó) tartalom is kerüljön bele
DIB_RGB_COLORS = 0
BYTES_PER_PIXEL = 4
FAILED_GRABS_BEFORE_FALLBACK = 3
BLACK_FRAMES_BEFORE_FALLBACK = 30  # ennyi fekete sáv után (~3 mp) vetjük össze az ablak képét a képernyővel
# Ennyi egymás utáni összevetésnek kell ugyanazt mondania: egy sötét jelenet vége nem eshet ennyiszer egybe vele
BLACK_CHECKS_BEFORE_FALLBACK = 3
BLACK_LEVEL = 8  # ennél sötétebb sávban nincs látható játékkép
# A képernyősáv ekkora része legyen nem fekete, hogy játékképnek számítson (egy értesítés vagy felirat kevesebb)
MIN_PICTURE_SHARE = 0.2
SINGLE_THREAD_PIXELS = 60_000  # ~65 ezer pixel fölött az OpenCV a munkaszálaira osztja a színkonverziót

logger = logging.getLogger(__name__)
AreaOf = Callable[[int, int], Box]  # (kliensszélesség, kliensmagasság) -> a sáv helye a kliensterületen


class WindowGrabError(OSError):
    """A PrintWindow nem ad képet erről az ablakról."""


class GrabbedBand(NamedTuple):
    """Egy rögzített sáv: a kép (BGR), a helye a kliensterületen és a kliensterület (szélesség, magasság)."""

    image: np.ndarray
    area: Box
    frame_size: tuple[int, int]


class _BitmapInfoHeader(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG), ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD),
    ]


_libs: tuple[ctypes.WinDLL, ctypes.WinDLL] | None = None


def _gdi() -> tuple[ctypes.WinDLL, ctypes.WinDLL]:
    # Saját példányok pontos típusokkal: 64 bites leírók csonkolás nélkül
    global _libs
    if _libs is None:
        user32, gdi32 = ctypes.WinDLL("user32"), ctypes.WinDLL("gdi32")
        handle = wintypes.HANDLE
        user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        user32.PrintWindow.argtypes, user32.PrintWindow.restype = [wintypes.HWND, handle, wintypes.UINT], wintypes.BOOL
        gdi32.CreateCompatibleDC.argtypes, gdi32.CreateCompatibleDC.restype = [handle], handle
        gdi32.CreateDIBSection.argtypes = [handle, ctypes.c_void_p, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p),
                                           handle, wintypes.DWORD]
        gdi32.CreateDIBSection.restype = handle
        gdi32.SelectObject.argtypes, gdi32.SelectObject.restype = [handle, handle], handle
        gdi32.DeleteObject.argtypes = [handle]
        gdi32.DeleteDC.argtypes = [handle]
        gdi32.GdiFlush.argtypes = []
        _libs = (user32, gdi32)
    return _libs


def fits(area: Box, width: int, height: int) -> bool:
    """Igaz, ha a sáv nem üres, és teljes egészében a megadott méretű területen belül van."""
    return 0 <= area.left < area.right <= width and 0 <= area.top < area.bottom <= height


def bgrx_to_bgr(pixels: np.ndarray) -> np.ndarray:
    """BGRX képből (lehet nézet is) saját BGR-tömb, a hívó szálán.

    Egyben konvertálva a cv2.cvtColor felébreszti az OpenCV munkaszálait: sávonként ~0,6 helyett ~2,6 ms
    processzoridő (mérve). Kisebb csíkokban az OpenCV nem osztja szét a munkát, és ugyanaz a kép készül.
    """
    height, width = pixels.shape[:2]
    image = np.empty((height, width, 3), np.uint8)
    rows = max(1, SINGLE_THREAD_PIXELS // max(width, 1))
    for top in range(0, height, rows):
        cv2.cvtColor(pixels[top:top + rows], cv2.COLOR_BGRA2BGR, dst=image[top:top + rows])
    return image


def cut_band(pixels: np.ndarray, area: Box) -> np.ndarray:
    """A sáv saját BGR-tömbként a BGRX rajzfelületből; ami a felületen kívülre esne, az fekete."""
    height, width = pixels.shape[:2]
    if fits(area, width, height):
        return bgrx_to_bgr(pixels[area.top:area.bottom, area.left:area.right])
    band = np.zeros((area.height, area.width, 3), np.uint8)
    left, top, right, bottom = max(area.left, 0), max(area.top, 0), min(area.right, width), min(area.bottom, height)
    if left < right and top < bottom:
        band[top - area.top:bottom - area.top, left - area.left:right - area.left] = pixels[top:bottom, left:right, :3]
    return band


def _delete_surface(dc: int, bitmap: int, previous: int) -> None:
    _, gdi32 = _gdi()
    gdi32.SelectObject(dc, previous)
    gdi32.DeleteObject(bitmap)
    gdi32.DeleteDC(dc)


class _DibSurface:
    """Memória-DC egy felülről lefelé tárolt, 32 bites DIB-bel; a pixels a DIB memóriájára néz (BGRX)."""

    def __init__(self, width: int, height: int) -> None:
        _, gdi32 = _gdi()
        self.size = (width, height)
        self.dc = gdi32.CreateCompatibleDC(None)
        if not self.dc:
            raise WindowGrabError("cannot create a memory DC for the capture surface")
        # Negatív magasság: felülről lefelé tárolt sorok, mint a numpy-tömbben
        header = _BitmapInfoHeader(ctypes.sizeof(_BitmapInfoHeader), width, -height, 1, 8 * BYTES_PER_PIXEL,
                                   0, 0, 0, 0, 0, 0)
        bits = ctypes.c_void_p()
        bitmap = gdi32.CreateDIBSection(self.dc, ctypes.byref(header), DIB_RGB_COLORS, ctypes.byref(bits), None, 0)
        if not bitmap or not bits.value:
            gdi32.DeleteDC(self.dc)
            raise WindowGrabError(f"cannot create a {width}x{height} capture surface")
        memory = (ctypes.c_uint8 * (width * height * BYTES_PER_PIXEL)).from_address(bits.value)
        self.pixels = np.ctypeslib.as_array(memory).reshape(height, width, BYTES_PER_PIXEL)
        # Akkor is felszabadul, ha a gazdája elfelejti lezárni (szemétgyűjtéskor vagy kilépéskor)
        self._finalizer = weakref.finalize(self, _delete_surface, self.dc, bitmap,
                                           gdi32.SelectObject(self.dc, bitmap))

    def close(self) -> None:
        self.pixels = np.zeros((0, 0, BYTES_PER_PIXEL), np.uint8)  # a DIB memóriája megszűnik
        self._finalizer()


class WindowContentGrabber:
    """Az ablak kliensterületének egy részét adja vissza BGR képként (None, ha nem sikerült).

    A PrintWindow egy tartós DIB-be rajzol, és a sávot közvetlenül annak a memóriájából vesszük ki: nem kell
    rögzítésenként DC-ket és bitképeket létrehozni (mérve: ~16 helyett ~7 ms processzoridő rögzítésenként).
    Egyszerre egy szál használhatja; a close() elengedi a felületet.
    """

    def __init__(self, hwnd: int) -> None:
        self._hwnd = hwnd
        self._surface: _DibSurface | None = None

    def grab(self, area: Box) -> np.ndarray | None:
        user32, gdi32 = _gdi()
        client = wintypes.RECT()
        if not user32.GetClientRect(self._hwnd, ctypes.byref(client)) or area.width <= 0 or area.height <= 0:
            return None
        # A PrintWindow mindig az ablak bal felső sarkától rajzol, és a nagyobb cél drágább:
        # a felület csak akkora, hogy a sáv beleférjen
        surface = self._surface_for(max(1, min(client.right, area.right)), max(1, min(client.bottom, area.bottom)))
        if surface is None or not user32.PrintWindow(self._hwnd, surface.dc, PW_CLIENTONLY | PW_RENDERFULLCONTENT):
            return None
        gdi32.GdiFlush()  # a DIB memóriáját közvetlenül olvassuk: előbb minden rajzolás érjen célba
        return cut_band(surface.pixels, area)

    def _surface_for(self, width: int, height: int) -> _DibSurface | None:
        """A tartós rajzfelület; új csak akkor készül, ha más méret kell (átméretezett ablak, másik sáv)."""
        if self._surface is None or self._surface.size != (width, height):
            self.close()
            try:
                self._surface = _DibSurface(width, height)
            except WindowGrabError as err:
                logger.debug("window grab: %s", err)
        return self._surface

    def close(self) -> None:
        """Elengedi a rajzfelületet; a következő grab újat készít."""
        surface, self._surface = self._surface, None
        if surface is not None:
            surface.close()


class PrintWindowSource:
    """A folyamatos rögzítés (band_feed.BandFeed) tartalék forrása: PrintWindow a tartós rajzfelületre.

    Néhány egymás utáni sikertelen rögzítés után WindowGrabError-ral feladja. period: két olvasás kezdete
    között ennyi másodperc teljen el (a PrintWindow drága, ezért ritkábban olvasunk, mint az elsődleges forrásból).
    """

    def __init__(self, hwnd: int, period: float) -> None:
        self._hwnd = hwnd
        self._grabber = WindowContentGrabber(hwnd)
        self._failures = 0
        self.poll_period = self.band_period = period

    def read(self, area_of: AreaOf) -> GrabbedBand | None:
        client = client_rect(self._hwnd)
        if client is None:  # lekicsinyítve nincs mit rögzíteni, de ez nem hiba
            return None
        area = area_of(client.width, client.height)
        if not fits(area, client.width, client.height):
            return None
        image = self._grabber.grab(area)
        if image is None:
            self._failures += 1
            if self._failures >= FAILED_GRABS_BEFORE_FALLBACK:
                raise WindowGrabError("PrintWindow keeps failing on this window")
            return None
        self._failures = 0
        return GrabbedBand(image, area, (client.width, client.height))

    def close(self) -> None:
        self._grabber.close()


class _WindowGrabber(Protocol):
    def grab(self, area: Box) -> np.ndarray | None: ...


class _ScreenGrabber(Protocol):
    def grab(self, rect: ScreenRect) -> np.ndarray: ...


class BandCapture:
    """Elsőként az ablak saját tartalmát rögzíti; ha az nem megy, véglegesen a képernyőre vált.

    A váltás az angol felirat takarását is kikapcsolhatja, ezért csak biztos jelre történik: a PrintWindow
    sorozatban hibázik, vagy az ablak sávja tartósan fekete, miközben a képernyőn ugyanott újra és újra
    a játék látszik (egy sötét jelenet ezt nem tudja előidézni).
    """

    def __init__(self, window: _WindowGrabber, screen: _ScreenGrabber,
                 can_judge_screen: Callable[[ScreenRect], bool] = lambda rect: True) -> None:
        """can_judge_screen: igaz, ha a képernyőn a sáv helyén most csak a játék látszhat (nem takarja más
        ablak, és a saját feliratrétegünk sem rajzol rá); az OCR-szálon hívjuk."""
        self._window = window
        self._screen = screen
        self._can_judge_screen = can_judge_screen
        self.uses_window_content = True  # amíg igaz, az OCR nem látja a feliratréteget
        self.last_from_window = False  # az utolsó sáv az ablakból jött (benne nincs a réteg)
        self._failures = 0
        self._black_frames = 0
        self._black_checks = 0  # ennyi egymás utáni összevetés szerint látszott a képernyőn a játék

    def grab(self, rect: ScreenRect, area: Box) -> np.ndarray:
        """rect: a sáv a képernyőn; area: ugyanaz a sáv az ablak kliensterületén belül."""
        self.last_from_window = False
        if not self.uses_window_content:
            return self._screen.grab(rect)
        band = self._window.grab(area)
        if band is None:
            self._failures += 1
            if self._failures >= FAILED_GRABS_BEFORE_FALLBACK:
                self._fall_back(t("capture.printwindow_failed"))
            return self._screen.grab(rect)
        self._failures = 0
        self.last_from_window = True
        if band.max() > BLACK_LEVEL:
            self._black_frames = self._black_checks = 0
            return band
        self._black_frames += 1
        if self._black_frames < BLACK_FRAMES_BEFORE_FALLBACK:
            return band
        self._black_frames = 0
        screen = self._screen_showing_the_game(rect)
        if screen is None:
            self._black_checks = 0
            return band  # sötét jelenet, vagy a képernyő most nem bizonyít semmit
        self._black_checks += 1
        if self._black_checks < BLACK_CHECKS_BEFORE_FALLBACK:
            return band
        self._fall_back(t("capture.window_black"))  # a képernyőn tartósan a játék látszik, az ablak képe fekete
        self.last_from_window = False
        return screen

    def _screen_showing_the_game(self, rect: ScreenRect) -> np.ndarray | None:
        """A képernyő sávja, ha most megítélhető, és valódi kép látszik rajta; különben None."""
        if not self._can_judge_screen(rect):
            return None
        screen = self._screen.grab(rect)
        lit_share = float((screen.max(axis=2) > BLACK_LEVEL).mean())
        return screen if lit_share >= MIN_PICTURE_SHARE else None

    def _fall_back(self, reason: str) -> None:
        self.uses_window_content = False
        logger.warning(t("capture.window_fallback", reason=reason))
