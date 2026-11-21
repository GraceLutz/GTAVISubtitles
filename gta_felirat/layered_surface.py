"""Pixelenként átlátszó ablakfelület (Windows UpdateLayeredWindow).

A réteg képe egy 32 bites DIB-ben van (BGRA, előszorzott alfa). Ahol az alfa 0, ott az
ablak teljesen átlátszó, és a kattintás is átmegy rajta a Remote Play ablakára; a köztes
értékeknél a réteg lágyan beleolvad az alatta futó játékképbe.
"""

import ctypes
from ctypes import wintypes

import numpy as np

BI_RGB = 0
DIB_RGB_COLORS = 0
AC_SRC_OVER = 0
AC_SRC_ALPHA = 1
ULW_ALPHA = 2
FULLY_OPAQUE = 255
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD)]


class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [("BlendOp", ctypes.c_ubyte), ("BlendFlags", ctypes.c_ubyte),
                ("SourceConstantAlpha", ctypes.c_ubyte), ("AlphaFormat", ctypes.c_ubyte)]


class UPDATELAYEREDWINDOWINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("hdcDst", wintypes.HDC), ("pptDst", ctypes.POINTER(wintypes.POINT)),
                ("psize", ctypes.POINTER(wintypes.SIZE)), ("hdcSrc", wintypes.HDC),
                ("pptSrc", ctypes.POINTER(wintypes.POINT)), ("crKey", wintypes.COLORREF),
                ("pblend", ctypes.POINTER(BLENDFUNCTION)), ("dwFlags", wintypes.DWORD),
                ("prcDirty", ctypes.POINTER(wintypes.RECT))]


class LayeredSurfaceError(OSError):
    """A pixelenként átlátszó réteg nem hozható létre vagy nem frissíthető."""


def _win32() -> tuple[ctypes.WinDLL, ctypes.WinDLL]:
    # Saját példány pontos argtypes-szal: a 64 bites leírókat a ctypes különben levágná
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
    gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
    gdi32.CreateCompatibleDC.restype = wintypes.HDC
    gdi32.CreateDIBSection.argtypes = [wintypes.HDC, ctypes.c_void_p, wintypes.UINT,
                                       ctypes.POINTER(ctypes.c_void_p), wintypes.HANDLE, wintypes.DWORD]
    gdi32.CreateDIBSection.restype = wintypes.HBITMAP
    gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    gdi32.SelectObject.restype = wintypes.HGDIOBJ
    gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    gdi32.DeleteDC.argtypes = [wintypes.HDC]
    user32.UpdateLayeredWindow.argtypes = [
        wintypes.HWND, wintypes.HDC, ctypes.POINTER(wintypes.POINT), ctypes.POINTER(wintypes.SIZE),
        wintypes.HDC, ctypes.POINTER(wintypes.POINT), wintypes.COLORREF, ctypes.POINTER(BLENDFUNCTION),
        wintypes.DWORD]
    user32.UpdateLayeredWindow.restype = wintypes.BOOL
    user32.UpdateLayeredWindowIndirect.argtypes = [wintypes.HWND, ctypes.POINTER(UPDATELAYEREDWINDOWINFO)]
    user32.UpdateLayeredWindowIndirect.restype = wintypes.BOOL
    user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
    user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
    user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
    return user32, gdi32


def premultiply(bgra: np.ndarray) -> np.ndarray:
    """Egyenes alfából előszorzott alfa (ezt várja az UpdateLayeredWindow)."""
    result = bgra.copy()
    alpha = bgra[:, :, 3:4].astype(np.uint16)
    result[:, :, :3] = ((bgra[:, :, :3].astype(np.uint16) * alpha + 127) // FULLY_OPAQUE).astype(np.uint8)
    return result


class LayeredSurface:
    def __init__(self, hwnd: int) -> None:
        self._hwnd = hwnd
        self._user32, self._gdi32 = _win32()
        self._dc = self._gdi32.CreateCompatibleDC(None)
        if not self._dc:
            raise LayeredSurfaceError(ctypes.get_last_error(), "CreateCompatibleDC")
        self._bitmap = None
        self._old_bitmap = None
        self._size = (0, 0)
        self._supports_dirty = True  # az UpdateLayeredWindowIndirect részleges frissítése működik-e
        self.pixels = np.zeros((0, 0, 4), np.uint8)  # a DIB memóriája: ide írjuk a képet
        style = self._user32.GetWindowLongPtrW(hwnd, GWL_EXSTYLE)
        self._user32.SetWindowLongPtrW(hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED)

    def resize(self, width: int, height: int) -> bool:
        """A felület mérete; True, ha új (üres) felület készült, False, ha a méret nem változott."""
        if (width, height) == self._size:
            return False
        self._release_bitmap()
        header = BITMAPINFOHEADER(biSize=ctypes.sizeof(BITMAPINFOHEADER), biWidth=width,
                                  biHeight=-height, biPlanes=1, biBitCount=32, biCompression=BI_RGB)
        bits = ctypes.c_void_p()
        self._bitmap = self._gdi32.CreateDIBSection(self._dc, ctypes.byref(header), DIB_RGB_COLORS,
                                                    ctypes.byref(bits), None, 0)
        if not self._bitmap or not bits.value:
            raise LayeredSurfaceError(ctypes.get_last_error(), "CreateDIBSection")
        self._old_bitmap = self._gdi32.SelectObject(self._dc, self._bitmap)
        buffer = (ctypes.c_uint8 * (width * height * 4)).from_address(bits.value)
        self.pixels = np.ctypeslib.as_array(buffer).reshape(height, width, 4)
        self.pixels[:] = 0
        self._size = (width, height)
        return True

    def present(self, left: int, top: int, dirty: tuple[int, int, int, int] | None = None) -> None:
        """A pixels tartalmát kiteszi a képernyőre a (left, top) ponttól.

        dirty: (bal, felső, jobb, alsó) – ha csak ez a rész változott, a rendszernek elég ennyit
        átvennie a 8 MB-os felület helyett. None: a teljes felület (méret- vagy helyváltás után).
        """
        if dirty is not None and self._supports_dirty and self._present_dirty(left, top, dirty):
            return
        width, height = self._size
        blend = BLENDFUNCTION(AC_SRC_OVER, 0, FULLY_OPAQUE, AC_SRC_ALPHA)
        ok = self._user32.UpdateLayeredWindow(
            self._hwnd, None, ctypes.byref(wintypes.POINT(left, top)), ctypes.byref(wintypes.SIZE(width, height)),
            self._dc, ctypes.byref(wintypes.POINT(0, 0)), 0, ctypes.byref(blend), ULW_ALPHA)
        if not ok:
            raise LayeredSurfaceError(ctypes.get_last_error(), "UpdateLayeredWindow")

    def _present_dirty(self, left: int, top: int, dirty: tuple[int, int, int, int]) -> bool:
        """Részleges frissítés; False, ha a rendszer nem fogadta el (ekkor a teljes felület megy ki)."""
        width, height = self._size
        blend = BLENDFUNCTION(AC_SRC_OVER, 0, FULLY_OPAQUE, AC_SRC_ALPHA)
        destination, size, source = wintypes.POINT(left, top), wintypes.SIZE(width, height), wintypes.POINT(0, 0)
        area = wintypes.RECT(*dirty)
        info = UPDATELAYEREDWINDOWINFO(
            ctypes.sizeof(UPDATELAYEREDWINDOWINFO), None, ctypes.pointer(destination), ctypes.pointer(size),
            self._dc, ctypes.pointer(source), 0, ctypes.pointer(blend), ULW_ALPHA, ctypes.pointer(area))
        if self._user32.UpdateLayeredWindowIndirect(self._hwnd, ctypes.byref(info)):
            return True
        self._supports_dirty = False  # ezen a rendszeren nem megy: a továbbiakban mindig a teljes felület
        return False

    def _release_bitmap(self) -> None:
        if self._bitmap:
            self._gdi32.SelectObject(self._dc, self._old_bitmap)
            self._gdi32.DeleteObject(self._bitmap)
            self._bitmap = None
        self.pixels = np.zeros((0, 0, 4), np.uint8)

    def close(self) -> None:
        self._release_bitmap()
        if self._dc:
            self._gdi32.DeleteDC(self._dc)
            self._dc = None
