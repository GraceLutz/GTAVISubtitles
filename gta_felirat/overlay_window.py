"""Átlátszó, mindig felül lévő feliratréteg egy másik ablak (pl. PS Remote Play) fölött.

- Pixelenként átlátszó (UpdateLayeredWindow): ahol nincs semmi, ott a kattintás is átmegy
  a Remote Playre; az angol felirat takarásának széle lágyan beleolvad az élő játékképbe.
- Maga a feliratpanel megfogható: húzható, görgethető, jobb klikkel visszaáll.
- WS_EX_NOACTIVATE: a panelre kattintás nem veszi el a fókuszt a Remote Playtől.
- Alapból a réteg látszik a képrögzítésben (Medal, OBS); az OCR a saját fordításunk
  helyét kitakarja (window_capture.blank_out). Ha ki van kapcsolva, WDA_EXCLUDEFROMCAPTURE:
  ekkor a felvételeken sem látszik.
"""

import ctypes
import logging
import tkinter as tk
from ctypes import wintypes

import cv2

from .frosted_cover import Backdrop
from .i18n import t
from .layer_compositor import LayerCompositor
from .layer_plan import Shown
from .layered_surface import LayeredSurface
from .ocr import Box
from .subtitle_layer import SubtitleLayer
from .window_capture import ScreenRect

GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
HWND_TOPMOST = -1
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
WDA_NONE = 0x00000000
WDA_EXCLUDEFROMCAPTURE = 0x00000011
WHEEL_SHIFT = 16

logger = logging.getLogger(__name__)


class OverlayWindow:
    def __init__(self, root: tk.Tk, layer: SubtitleLayer, capture_visible: bool = True) -> None:
        self._layer = layer
        self._capture_visible = capture_visible
        self._rect: ScreenRect | None = None
        self._surface: LayeredSurface | None = None
        self._compositor = LayerCompositor()  # helyben rajzol az ablak képpontjaiba
        self._surface_is_new = True  # méretváltás után az első kirakás a teljes felület
        self._visible = False
        self.needs_redraw = True
        self._window = tk.Toplevel(root)
        self._window.overrideredirect(True)
        self._window.attributes("-topmost", True)
        # A vászon csak az egéreseményekért kell; a képet az UpdateLayeredWindow teszi ki
        self._canvas = tk.Canvas(self._window, bd=0, highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)
        self._bind_mouse()
        self._window.withdraw()

    # --- elhelyezés -------------------------------------------------------
    def place(self, rect: ScreenRect) -> None:
        if self._surface is None:
            # Még rejtve: így a réteg már átlátszóként jelenik meg, nem villan fel szürkén
            self._window.update_idletasks()
            self._apply_window_styles()
        if rect != self._rect:
            self._rect = rect
            self._window.geometry(f"{rect.width}x{rect.height}+{rect.left}+{rect.top}")
            if self._surface.resize(rect.width, rect.height):
                self._compositor.reset()  # az új felület üres: nincs rajta mit törölni
                self._surface_is_new = True
            self.needs_redraw = True
        if not self._visible:
            self._show_without_taking_focus()
            self._visible = True

    def _show_without_taking_focus(self) -> None:
        """Megjelenéskor a Tk előtérbe hozná a réteget: az addigi ablak kapja vissza a billentyűzetet."""
        user32 = ctypes.windll.user32
        user32.GetForegroundWindow.restype = wintypes.HWND
        user32.SetForegroundWindow.argtypes = [wintypes.HWND]
        previous = user32.GetForegroundWindow()
        self._window.deiconify()
        self._window.update_idletasks()
        if previous and user32.GetForegroundWindow() == self.hwnd:
            user32.SetForegroundWindow(previous)

    def hide(self) -> None:
        if self._visible:
            self._window.withdraw()
            self._visible = False

    @property
    def hwnd(self) -> int:
        """A réteg felső szintű ablakának azonosítója (a takarás-ellenőrzés ezt nem veszi takarásnak)."""
        return int(self._window.wm_frame(), 16)

    def keep_on_top(self) -> None:
        """A réteg maradjon a játék fölött, de ne kerüljön előtérbe.

        Nem a Tk lift(): az előtérbe hozná a réteget, valahányszor a beállítóablak az aktív,
        és másodpercenként elvenné tőle a billentyűzetet (nem lehetne a mezőibe írni).
        """
        if not self._visible:
            return
        set_window_pos = ctypes.windll.user32.SetWindowPos
        set_window_pos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, wintypes.UINT]
        set_window_pos(self.hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)

    def _apply_window_styles(self) -> None:
        user32 = ctypes.windll.user32
        hwnd = self.hwnd
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW)
        self._surface = LayeredSurface(hwnd)
        self._apply_capture_affinity()

    def set_capture_visible(self, visible: bool) -> None:
        """Látszódjon-e a réteg a felvételeken és streamen (Medal, OBS)."""
        if visible == self._capture_visible:
            return
        self._capture_visible = visible
        if self._surface is not None:
            self._apply_capture_affinity()

    def _apply_capture_affinity(self) -> None:
        affinity = WDA_NONE if self._capture_visible else WDA_EXCLUDEFROMCAPTURE
        if not ctypes.windll.user32.SetWindowDisplayAffinity(self.hwnd, affinity):
            # Régebbi Windowson nincs kizárás; ekkor a saját szövegünk helyét az OCR kitakarja
            logger.warning(t("overlay.capture_affinity_failed"))

    # --- rajzolás ---------------------------------------------------------
    def render(self, shown: Shown | None, hide_original: bool, preview_region: Box | None,
               backdrop: Backdrop | None = None) -> None:
        """Helyben rajzol az ablak képpontjaiba, és csak a megváltozott részt teszi ki a képernyőre.

        preview_region: a figyelt sáv kerete (a beállítóablak „Minta” kapcsolója).
        """
        if self._surface is None or self._rect is None:
            return
        self.needs_redraw = False
        plan = self._layer.plan((self._rect.width, self._rect.height), shown, hide_original, preview_region, backdrop)
        outlines = (preview_region,) if preview_region is not None else ()
        changed = self._compositor.render(self._surface.pixels, plan, backdrop, outlines)
        if changed is None and not self._surface_is_new:
            return
        dirty = None if self._surface_is_new or changed is None else (
            changed.left, changed.top, changed.right, changed.bottom)
        self._surface.present(self._rect.left, self._rect.top, dirty)
        self._surface_is_new = False

    # --- egér -------------------------------------------------------------
    def _bind_mouse(self) -> None:
        events = {
            "<ButtonPress-1>": cv2.EVENT_LBUTTONDOWN,
            "<B1-Motion>": cv2.EVENT_MOUSEMOVE,
            "<ButtonRelease-1>": cv2.EVENT_LBUTTONUP,
            "<ButtonPress-3>": cv2.EVENT_RBUTTONDOWN,
        }
        for sequence, cv_event in events.items():
            self._canvas.bind(sequence, lambda e, ev=cv_event: self._forward(ev, e.x, e.y, 0))
        self._canvas.bind("<MouseWheel>", lambda e: self._forward(
            cv2.EVENT_MOUSEWHEEL, e.x, e.y, e.delta << WHEEL_SHIFT))

    def _forward(self, cv_event: int, x: int, y: int, flags: int) -> None:
        if self._layer.handle_mouse(cv_event, x, y, flags):
            self.needs_redraw = True

    def destroy(self) -> None:
        if self._surface is not None:
            self._surface.close()
        self._window.destroy()
