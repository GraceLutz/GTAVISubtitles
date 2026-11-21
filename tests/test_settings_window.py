import tkinter as tk
from dataclasses import replace

import numpy as np
import pytest

from gta_felirat import i18n
from gta_felirat.config import AppConfig, Region
from gta_felirat.overlay_window import OverlayWindow
from gta_felirat.region_editor import RegionEditor
from gta_felirat.settings_window import SettingsCallbacks, SettingsWindow
from gta_felirat.subtitle_layer import SubtitleLayer
from gta_felirat.window_capture import ScreenRect


@pytest.fixture(scope="module")
def shared_window():
    # Egyetlen Tk-gyökér a modulnak: Windowson a többszöri létrehozás néha elbukik
    i18n.set_language("en")
    callbacks = SettingsCallbacks(on_change=lambda updates, save: None, on_session_key=lambda name, key: None,
                                  has_key=lambda name: False, status=lambda: "ok", on_preview=lambda shown: None)
    try:
        settings = SettingsWindow(AppConfig(ui_language="en"), callbacks)
    except tk.TclError:
        pytest.skip("no display for Tk")
    yield settings
    settings.close()
    i18n.set_language(i18n.AUTO)


def switch_language(window, code):
    i18n.set_language(code)
    window.sync(replace(AppConfig(), ui_language=code))
    window.pump()


@pytest.fixture
def window(shared_window):
    switch_language(shared_window, "en")
    return shared_window


def test_language_switch_rebuilds_the_window_in_the_new_language(window):
    english_title = window.root.title()

    switch_language(window, "hu")

    assert window.root.title() == i18n.catalog("hu")["settings.title"] != english_title


def test_language_switch_keeps_other_windows_under_the_root_alive(window):
    # Valós hiba: a nyelvváltás a feliratréteg ablakát is megsemmisítette, és a felirat eltűnt
    overlay = tk.Toplevel(window.root)

    switch_language(window, "hu")

    assert overlay.winfo_exists()
    overlay.destroy()


def test_keeping_the_overlay_on_top_does_not_raise_it_through_tk(window, monkeypatch):
    # Valós hiba: a Tk lift() előtérbe hozta a réteget, és másodpercenként elvette a billentyűzetet
    # a beállítóablaktól, így nem lehetett a mezőibe írni
    overlay = OverlayWindow(window.root, SubtitleLayer(AppConfig(), on_change=lambda updates: None),
                            RegionEditor(Region(), on_change=lambda updates: None))
    overlay.place(ScreenRect(50, 50, 320, 200))
    raised = []
    monkeypatch.setattr(overlay._window, "lift", lambda *args: raised.append("lift"))
    monkeypatch.setattr(overlay._window, "focus_force", lambda *args: raised.append("focus_force"))

    overlay.keep_on_top()
    window.pump()

    assert raised == []
    overlay.destroy()


def test_overlay_draws_in_place_and_clears_when_nothing_is_shown(window):
    from gta_felirat.frosted_cover import Backdrop
    from gta_felirat.layer_plan import Shown, ShownLine
    from gta_felirat.ocr import Box

    config = AppConfig(hide_original=True, panel_opacity=0, font_size_ratio=0.05)
    overlay = OverlayWindow(window.root, SubtitleLayer(config, on_change=lambda updates: None),
                            RegionEditor(Region(), on_change=lambda updates: None))
    overlay.place(ScreenRect(40, 40, 640, 360))
    line = Box(200, 250, 440, 280)
    band = Backdrop(np.full((120, 640, 3), 30, np.uint8), Box(0, 220, 640, 340))
    shown = Shown("Szia!", line, (ShownLine("Szia!", line),))

    overlay.render(shown, True, None, band)  # első kirakás: a teljes felület
    overlay.render(shown, True, None, Backdrop(band.image, band.area))  # második: csak a megváltozott rész
    drawn = overlay._surface.pixels.copy()
    overlay.render(None, True, None, None)

    assert drawn[:, :, 3].max() == 255  # a fordítás betűi teljesen fednek
    assert not overlay._surface.pixels.any()  # ha nincs mit mutatni, a felület újra üres
    overlay.destroy()
