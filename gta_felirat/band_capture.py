"""A feliratsáv rögzítése közvetlenül a Remote Play ablakból (PrintWindow), a fölötte lévő ablakok nélkül.

Így a feliratrétegünk (fordítás, az angol szöveg takarása) nem kerül bele abba, amit az OCR
olvas, miközben a Medal/OBS felvételen látszik. Ha az ablak tartalma nem rögzíthető (a
PrintWindow hibát ad, vagy tartósan fekete képet, miközben a képernyőn látszik a játék),
a képernyőről rögzítünk tovább (mss), mint korábban.
"""

import ctypes
import ctypes.wintypes as wintypes
import logging
from typing import Protocol

import numpy as np

from .ocr import Box
from .window_capture import ScreenRect

PW_CLIENTONLY = 0x1
PW_RENDERFULLCONTENT = 0x2  # a hardveresen gyorsított (videó) tartalom is kerüljön bele
SRCCOPY = 0x00CC0020
DIB_RGB_COLORS = 0
BYTES_PER_PIXEL = 4
FAILED_GRABS_BEFORE_FALLBACK = 3
BLACK_FRAMES_BEFORE_FALLBACK = 30  # ~3 másodperc OCR-ütemben
BLACK_LEVEL = 8  # ennél sötétebb sávban nincs látható játékkép

logger = logging.getLogger(__name__)


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
        user32.GetDC.argtypes, user32.GetDC.restype = [wintypes.HWND], handle
        user32.ReleaseDC.argtypes = [wintypes.HWND, handle]
        user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        user32.PrintWindow.argtypes, user32.PrintWindow.restype = [wintypes.HWND, handle, wintypes.UINT], wintypes.BOOL
        gdi32.CreateCompatibleDC.argtypes, gdi32.CreateCompatibleDC.restype = [handle], handle
        gdi32.CreateCompatibleBitmap.argtypes = [handle, ctypes.c_int, ctypes.c_int]
        gdi32.CreateCompatibleBitmap.restype = handle
        gdi32.SelectObject.argtypes, gdi32.SelectObject.restype = [handle, handle], handle
        gdi32.BitBlt.argtypes = [handle, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 handle, ctypes.c_int, ctypes.c_int, wintypes.DWORD]
        gdi32.GetDIBits.argtypes = [handle, handle, wintypes.UINT, wintypes.UINT, ctypes.c_void_p,
                                    ctypes.c_void_p, wintypes.UINT]
        gdi32.DeleteObject.argtypes = [handle]
        gdi32.DeleteDC.argtypes = [handle]
        _libs = (user32, gdi32)
    return _libs


class WindowContentGrabber:
    """Az ablak kliensterületének egy részét adja vissza BGR képként (None, ha nem sikerült)."""

    def __init__(self, hwnd: int) -> None:
        self._hwnd = hwnd

    def grab(self, area: Box) -> np.ndarray | None:
        user32, gdi32 = _gdi()
        client = wintypes.RECT()
        if not user32.GetClientRect(self._hwnd, ctypes.byref(client)) or area.width <= 0 or area.height <= 0:
            return None
        window_dc = user32.GetDC(self._hwnd)
        if not window_dc:
            return None
        try:
            return self._grab_with(window_dc, client.right, client.bottom, area)
        finally:
            user32.ReleaseDC(self._hwnd, window_dc)

    def _grab_with(self, window_dc, client_width: int, client_height: int, area: Box) -> np.ndarray | None:
        user32, gdi32 = _gdi()
        full_dc, band_dc = gdi32.CreateCompatibleDC(window_dc), gdi32.CreateCompatibleDC(window_dc)
        full_bmp = gdi32.CreateCompatibleBitmap(window_dc, client_width, client_height)
        band_bmp = gdi32.CreateCompatibleBitmap(window_dc, area.width, area.height)
        try:
            old_full = gdi32.SelectObject(full_dc, full_bmp)
            old_band = gdi32.SelectObject(band_dc, band_bmp)
            printed = user32.PrintWindow(self._hwnd, full_dc, PW_CLIENTONLY | PW_RENDERFULLCONTENT)
            if printed:
                gdi32.BitBlt(band_dc, 0, 0, area.width, area.height, full_dc, area.left, area.top, SRCCOPY)
            gdi32.SelectObject(full_dc, old_full)
            gdi32.SelectObject(band_dc, old_band)  # a GetDIBits-hez a bitkép nem lehet DC-ben
            return self._read_bitmap(window_dc, band_bmp, area) if printed else None
        finally:
            for bitmap in (full_bmp, band_bmp):
                gdi32.DeleteObject(bitmap)
            for dc in (full_dc, band_dc):
                gdi32.DeleteDC(dc)

    @staticmethod
    def _read_bitmap(window_dc, bitmap, area: Box) -> np.ndarray | None:
        _, gdi32 = _gdi()
        # Negatív magasság: felülről lefelé tárolt sorok, mint a numpy-tömbben
        header = _BitmapInfoHeader(ctypes.sizeof(_BitmapInfoHeader), area.width, -area.height, 1,
                                   8 * BYTES_PER_PIXEL, 0, 0, 0, 0, 0, 0)
        buffer = np.empty((area.height, area.width, BYTES_PER_PIXEL), np.uint8)
        lines = gdi32.GetDIBits(window_dc, bitmap, 0, area.height, buffer.ctypes.data,
                                ctypes.byref(header), DIB_RGB_COLORS)
        return buffer[:, :, :3].copy() if lines == area.height else None


class _WindowGrabber(Protocol):
    def grab(self, area: Box) -> np.ndarray | None: ...


class _ScreenGrabber(Protocol):
    def grab(self, rect: ScreenRect) -> np.ndarray: ...


class BandCapture:
    """Elsőként az ablak saját tartalmát rögzíti; ha az nem megy, véglegesen a képernyőre vált."""

    def __init__(self, window: _WindowGrabber, screen: _ScreenGrabber) -> None:
        self._window = window
        self._screen = screen
        self.uses_window_content = True  # amíg igaz, az OCR nem látja a feliratréteget
        self.last_from_window = False  # az utolsó sáv az ablakból jött (benne nincs a réteg)
        self._failures = 0
        self._black_frames = 0

    def grab(self, rect: ScreenRect, area: Box) -> np.ndarray:
        """rect: a sáv a képernyőn; area: ugyanaz a sáv az ablak kliensterületén belül."""
        self.last_from_window = False
        if not self.uses_window_content:
            return self._screen.grab(rect)
        band = self._window.grab(area)
        if band is None:
            self._failures += 1
            if self._failures >= FAILED_GRABS_BEFORE_FALLBACK:
                self._fall_back("a PrintWindow nem működik ezen az ablakon")
            return self._screen.grab(rect)
        self._failures = 0
        self.last_from_window = True
        if band.max() > BLACK_LEVEL:
            self._black_frames = 0
            return band
        self._black_frames += 1
        if self._black_frames < BLACK_FRAMES_BEFORE_FALLBACK:
            return band
        self._black_frames = 0
        screen = self._screen.grab(rect)
        if screen.max() > BLACK_LEVEL:  # a képernyőn látszik a játék, csak a PrintWindow ad feketét
            self._fall_back("az ablak tartalma feketén rögzül")
            self.last_from_window = False
            return screen
        return band  # tényleg sötét a jelenet

    def _fall_back(self, reason: str) -> None:
        self.uses_window_content = False
        logger.warning("Az ablak tartalma nem rögzíthető (%s): a képernyőről olvasok, "
                       "felvételre látszó rétegnél az angol felirat nem takarható", reason)
