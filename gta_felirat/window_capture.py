"""Egy ablak (pl. PS Remote Play) megkeresése cím alapján és a képének rögzítése.

Windows API-t ctypes-szal hívunk, a képet az mss olvassa a képernyőről. A saját
feliratrétegünket a képrögzítésből kizárjuk (lásd overlay_window.py), így az
OCR mindig az eredeti angol feliratot látja.
"""

import ctypes
import ctypes.wintypes as wintypes
import logging
import sys
import threading
from collections.abc import Callable, Collection
from dataclasses import dataclass

import numpy as np

from .i18n import t
from .ocr import Box

DPI_AWARENESS_PER_MONITOR = 2
GA_ROOT = 2
DWMWA_CLOAKED = 14
# A cím elején megadott szó után ezek a szó folytatásai (pl. „Veyra-NRVideo”), nem új szó kezdete
WORD_CONTINUATION = "-_/."
# Ilyen sűrű rácson nézzük meg, mi van legfelül a feliratsávban (egy keskeny csík se csússzon át)
OCCLUSION_SAMPLE_SPACING_PX = 40
OCCLUSION_SAMPLE_ROWS = 3
MIN_SAMPLE_COLUMNS = 2

logger = logging.getLogger(__name__)


class WindowNotFoundError(RuntimeError):
    """Nincs ilyen című, látható ablak."""


@dataclass(frozen=True)
class ScreenRect:
    left: int
    top: int
    width: int
    height: int


def enable_dpi_awareness() -> None:
    """Fizikai pixelekben dolgozzunk, különben 125%-os nagyításnál elcsúszik a réteg."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(DPI_AWARENESS_PER_MONITOR)
    except OSError as err:
        logger.debug("DPI-beállítás nem sikerült: %s", err)


@dataclass(frozen=True)
class VisibleWindow:
    hwnd: int
    title: str
    area: int  # a tartalmi rész területe pixelben (0, ha le van kicsinyítve)


def find_window(title: str, *, by_start: bool = False) -> int:
    """A megadott című látható ablak azonosítója (HWND); lásd choose_window."""
    windows = _visible_windows()
    hwnd = choose_window(title, windows, by_start=by_start)
    chosen = next(window for window in windows if window.hwnd == hwnd)
    logger.info(t("capture.window_chosen", title=chosen.title))
    return hwnd


def has_window(title: str, *, by_start: bool = False) -> bool:
    """Van-e most ilyen című látható ablak (pl. fut-e a PS Remote Play); lásd choose_window."""
    try:
        choose_window(title, _visible_windows(), by_start=by_start)
    except WindowNotFoundError:
        return False
    return True


def choose_window(title: str, windows: Collection[VisibleWindow], *, by_start: bool = False) -> int:
    """A pontosan ilyen című ablak; by_start esetén, ha nincs ilyen, az, amelyiknek a címe így kezdődik
    (kis- és nagybetű mindegy).

    A cím elejével olyan ablak is megadható, amelynek a címe nehezen begépelhető (pl. „Veyra — 采集卡 · LIVE”);
    ezt csak a felhasználó által megadott címnél engedjük, a beépített Remote Play-címnél nem (különben egy
    „PS Remote Play” kezdetű böngészőlapot is megtalálna). Több találatnál a legnagyobb: a játékablak,
    nem egy azonos című kis párbeszédablak.
    """
    wanted = title.strip()
    exact = [window for window in windows if wanted and window.title == wanted]
    starting = [window for window in windows if by_start and wanted and _starts_with_word(window.title, wanted)]
    matches = exact or starting
    if not matches:
        raise WindowNotFoundError(t("capture.window_not_found", title=title))
    return max(matches, key=lambda window: window.area).hwnd


def _starts_with_word(title: str, wanted: str) -> bool:
    """A cím így kezdődik, és utána szóköz vagy írásjel jön, nem a szó folytatása (kis- és nagybetű mindegy).

    „Veyra 2.2.0” és „Veyra — 采集卡 · LIVE” igen, egy „Veyra-NRVideo …” című böngészőlap nem.
    """
    if not title.casefold().startswith(wanted.casefold()):
        return False
    rest = title[len(wanted):]
    return not rest or not (rest[0].isalnum() or rest[0] in WORD_CONTINUATION)


def _visible_windows() -> list[VisibleWindow]:
    """A látható, címmel rendelkező felső szintű ablakok; a saját folyamatunké és a rejtetteké
    (pl. másik virtuális asztalon álló, felfüggesztett alkalmazás) nélkül."""
    user32 = ctypes.windll.user32
    own_process = ctypes.windll.kernel32.GetCurrentProcessId()
    found: list[VisibleWindow] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd, _param):
        if not user32.IsWindowVisible(hwnd) or _is_cloaked(hwnd):
            return True
        process = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process))
        if process.value == own_process:
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        found.append(VisibleWindow(hwnd, buffer.value, _window_area(hwnd)))
        return True

    user32.EnumWindows(visit, 0)
    return found


def _is_cloaked(hwnd: int) -> bool:
    cloaked = wintypes.DWORD()
    result = ctypes.windll.dwmapi.DwmGetWindowAttribute(hwnd, DWMWA_CLOAKED, ctypes.byref(cloaked),
                                                        ctypes.sizeof(cloaked))
    return result == 0 and cloaked.value != 0


class _WindowPlacement(ctypes.Structure):
    _fields_ = [("length", wintypes.UINT), ("flags", wintypes.UINT), ("showCmd", wintypes.UINT),
                ("ptMinPosition", wintypes.POINT), ("ptMaxPosition", wintypes.POINT),
                ("rcNormalPosition", wintypes.RECT)]


def _window_area(hwnd: int) -> int:
    """A tartalmi rész területe; lekicsinyített ablaknál a visszaállított mérete (különben 0 lenne,
    és egy azonos című kis párbeszédablak nyerne a valódi játékablakkal szemben)."""
    rect = client_rect(hwnd)
    if rect is not None:
        return rect.width * rect.height
    placement = _WindowPlacement()
    placement.length = ctypes.sizeof(placement)
    if not ctypes.windll.user32.GetWindowPlacement(hwnd, ctypes.byref(placement)):
        return 0
    normal = placement.rcNormalPosition
    return max(0, normal.right - normal.left) * max(0, normal.bottom - normal.top)


def client_rect(hwnd: int) -> ScreenRect | None:
    """Az ablak tartalmi része képernyő-koordinátákban; None, ha le van kicsinyítve vagy bezárták."""
    user32 = ctypes.windll.user32
    if not user32.IsWindow(hwnd) or user32.IsIconic(hwnd):
        return None
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    origin = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(origin))
    if rect.right <= 0 or rect.bottom <= 0:
        return None
    return ScreenRect(origin.x, origin.y, rect.right, rect.bottom)


def window_exists(hwnd: int) -> bool:
    return bool(ctypes.windll.user32.IsWindow(hwnd))


def blank_out(band: np.ndarray, band_area: Box, box: Box, margin: int) -> np.ndarray:
    """A sávképen feketére festi a saját kirajzolt fordításunk helyét (árnyékkal, körvonallal együtt).

    Ha a réteg látszik a képrögzítésben (Medal, OBS), az OCR enélkül visszaolvasná a magyar szöveget.
    band_area és box ugyanabban (a Remote Play ablak) koordináta-rendszerben van. Új képet ad vissza.
    """
    left = max(box.left - margin, band_area.left) - band_area.left
    top = max(box.top - margin, band_area.top) - band_area.top
    right = min(box.right + margin, band_area.right) - band_area.left
    bottom = min(box.bottom + margin, band_area.bottom) - band_area.top
    if right <= left or bottom <= top:
        return band
    result = band.copy()
    result[top:bottom, left:right] = 0
    return result


def band_sample_points(rect: ScreenRect) -> list[tuple[int, int]]:
    """Egyenletes rács a sáv belsejében (a szélektől fél cellányira)."""
    columns = max(MIN_SAMPLE_COLUMNS, rect.width // OCCLUSION_SAMPLE_SPACING_PX)
    return [
        (rect.left + int((col + 0.5) * rect.width / columns),
         rect.top + int((row + 0.5) * rect.height / OCCLUSION_SAMPLE_ROWS))
        for row in range(OCCLUSION_SAMPLE_ROWS)
        for col in range(columns)
    ]


_user32: ctypes.WinDLL | None = None


def _user32_for_hit_test() -> ctypes.WinDLL:
    # Saját példány, hogy az argtypes beállítása ne hasson a ctypes.windll.user32 többi hívására
    global _user32
    if _user32 is None:
        _user32 = ctypes.WinDLL("user32")
        _user32.WindowFromPoint.argtypes = [wintypes.POINT]
        _user32.WindowFromPoint.restype = wintypes.HWND
        _user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        _user32.GetAncestor.restype = wintypes.HWND
    return _user32


def root_window_at(x: int, y: int) -> int:
    """A képernyő adott pontján legfelül látszó felső szintű ablak (0, ha nincs)."""
    user32 = _user32_for_hit_test()
    hit = user32.WindowFromPoint(wintypes.POINT(x, y))
    if not hit:
        return 0
    return user32.GetAncestor(hit, GA_ROOT) or 0


def is_unobstructed(rect: ScreenRect, allowed: Collection[int],
                    root_at: Callable[[int, int], int] = root_window_at) -> bool:
    """Igaz, ha a sávban mindenhol a Remote Play (vagy a saját feliratrétegünk) van legfelül.

    Ha egy böngésző vagy más ablak belóg, a rögzítés azt látná, és fordításra küldené.
    """
    return all(root_at(x, y) in allowed for x, y in band_sample_points(rect))


class ScreenGrabber:
    """A megadott képernyőterületet BGR képként adja vissza; szálanként saját rögzítővel."""

    def __init__(self) -> None:
        self._local = threading.local()

    def _instance(self):
        if getattr(self._local, "mss", None) is None:
            import mss

            self._local.mss = mss.mss()
        return self._local.mss

    def grab(self, rect: ScreenRect) -> np.ndarray:
        shot = self._instance().grab({"left": rect.left, "top": rect.top, "width": rect.width, "height": rect.height})
        return np.asarray(shot)[:, :, :3].copy()

    def close(self) -> None:
        instance = getattr(self._local, "mss", None)
        if instance is not None:
            instance.close()
