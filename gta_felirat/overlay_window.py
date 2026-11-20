"""Átlátszó, mindig felül lévő feliratréteg egy másik ablak (pl. PS Remote Play) fölött.

- A kulcsszínnel kitöltött részek teljesen átlátszók, és a kattintás is átmegy
  rajtuk, így a Remote Play kapja meg az egeret.
- Maga a feliratpanel megfogható: húzható, görgethető, jobb klikkel visszaáll.
- WS_EX_NOACTIVATE: a panelre kattintás nem veszi el a fókuszt a Remote Playtől.
- WDA_EXCLUDEFROMCAPTURE: a réteg kimarad a képrögzítésből, így az OCR az eredeti
  angol feliratot látja, nem a saját fordításunkat.
"""

import ctypes
import logging
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .ocr import Box
from .subtitle_layer import Shown, SubtitleLayer
from .window_capture import ScreenRect

KEY_COLOR_RGB = (1, 2, 3)  # majdnem fekete: a betűk élsimítása sem ad színes szegélyt
KEY_COLOR_HEX = "#010203"
REGION_COLOR = (0, 200, 255)
OUTLINE_PX = 2
GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
WDA_EXCLUDEFROMCAPTURE = 0x00000011
WHEEL_SHIFT = 16

logger = logging.getLogger(__name__)


class OverlayWindow:
    def __init__(self, root: tk.Tk, layer: SubtitleLayer) -> None:
        self._layer = layer
        self._rect: ScreenRect | None = None
        self._base: np.ndarray | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._visible = False
        self.needs_redraw = True
        self._window = tk.Toplevel(root)
        self._window.overrideredirect(True)
        self._window.configure(bg=KEY_COLOR_HEX)
        self._window.attributes("-topmost", True)
        self._window.attributes("-transparentcolor", KEY_COLOR_HEX)
        # Vászon kulcsszínnel: csak a megváltozott sáv képét frissítjük, nem az egész ablakot
        self._canvas = tk.Canvas(self._window, bd=0, highlightthickness=0, bg=KEY_COLOR_HEX)
        self._canvas.pack(fill="both", expand=True)
        self._image_item = self._canvas.create_image(0, 0, anchor="nw", state="hidden")
        self._bind_mouse()
        self._window.withdraw()
        self._styled = False

    # --- elhelyezés -------------------------------------------------------
    def place(self, rect: ScreenRect) -> None:
        if rect != self._rect:
            self._rect = rect
            self._window.geometry(f"{rect.width}x{rect.height}+{rect.left}+{rect.top}")
            self._base = np.empty((rect.height, rect.width, 3), np.uint8)
            self._base[:] = KEY_COLOR_RGB[::-1]  # BGR
            self.needs_redraw = True
        if not self._visible:
            self._window.deiconify()
            self._visible = True
        if not self._styled:
            self._window.update_idletasks()
            self._apply_window_styles()
            self._styled = True

    def hide(self) -> None:
        if self._visible:
            self._window.withdraw()
            self._visible = False

    @property
    def hwnd(self) -> int:
        """A réteg felső szintű ablakának azonosítója (a takarás-ellenőrzés ezt nem veszi takarásnak)."""
        return int(self._window.wm_frame(), 16)

    def keep_on_top(self) -> None:
        if self._visible:
            self._window.attributes("-topmost", True)
            self._window.lift()

    def _apply_window_styles(self) -> None:
        user32 = ctypes.windll.user32
        hwnd = self.hwnd
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW)
        if not user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE):
            # Régebbi Windowson nincs ilyen; ekkor a pipeline szűri ki a saját szövegünket
            logger.warning("A réteget nem sikerült kizárni a képrögzítésből (Windows 10 2004+ kell)")

    # --- rajzolás ---------------------------------------------------------
    def render(self, shown: Shown | None, hide_original: bool, preview_region: Box | None) -> None:
        """Csak a panelt (és a takarást) tartalmazó sávot rajzolja újra: néhány ms a teljes ablak helyett."""
        if self._base is None:
            return
        self.needs_redraw = False
        frame = self._layer.draw(self._base, shown, hide_original, preview_region)
        if preview_region is not None:
            frame = frame.copy() if frame is self._base else frame
            cv2.rectangle(frame, (preview_region.left, preview_region.top),
                          (preview_region.right, preview_region.bottom), REGION_COLOR, 2)
        dirty = self._dirty_box(shown, hide_original, preview_region, frame.shape)
        if dirty is None:
            self._canvas.itemconfigure(self._image_item, state="hidden")
            return
        crop = frame[dirty.top:dirty.bottom, dirty.left:dirty.right, ::-1]
        self._photo = ImageTk.PhotoImage(Image.fromarray(np.ascontiguousarray(crop)))
        self._canvas.coords(self._image_item, dirty.left, dirty.top)
        self._canvas.itemconfigure(self._image_item, image=self._photo, state="normal")

    def _dirty_box(self, shown: Shown | None, hide_original: bool, preview_region: Box | None,
                   shape: tuple[int, ...]) -> Box | None:
        boxes = []
        if self._layer.last_rect is not None:
            x, y, width, height = self._layer.last_rect
            boxes.append(Box(x, y, x + width, y + height))
        if shown is not None and hide_original:
            boxes.append(shown.anchor)
        if preview_region is not None:
            boxes.append(Box(preview_region.left - OUTLINE_PX, preview_region.top - OUTLINE_PX,
                             preview_region.right + OUTLINE_PX, preview_region.bottom + OUTLINE_PX))
        if not boxes:
            return None
        frame_height, frame_width = shape[:2]
        box = Box(max(0, min(b.left for b in boxes)), max(0, min(b.top for b in boxes)),
                  min(frame_width, max(b.right for b in boxes)), min(frame_height, max(b.bottom for b in boxes)))
        return box if box.width > 0 and box.height > 0 else None

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
        self._window.destroy()
