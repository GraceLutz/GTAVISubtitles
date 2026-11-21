"""A feliratpanel beállítóablaka (Tkinter): felületi nyelv, célnyelv, fordító, API-kulcs, méret, átlátszóság.

Ugyanabban a szálban fut, mint a játékablak: az alkalmazás minden képkockánál
meghívja a pump()-ot, így a két ablak nem akad össze.
"""

import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from typing import Callable

from .config import BOLD_FONT_PATH, MAX_FONT_SIZE_RATIO, MIN_FONT_SIZE_RATIO, THIN_FONT_PATH, AppConfig
from .i18n import AUTO, UI_LANGUAGES, current_language, t
from .languages import LANGUAGES, find_language
from .translator import KEY_ENV_VARS, TRANSLATOR_NAMES, translator_label

STATUS_REFRESH_SECONDS = 0.5
PERCENT = 100.0
AI_MODEL_FIELDS = {"claude": "claude_model", "openai": "openai_model"}


@dataclass(frozen=True)
class SettingsCallbacks:
    on_change: Callable[[dict, bool], None]   # (módosítások, mentse-e)
    on_session_key: Callable[[str, str], None]  # (fordító, kulcs) — használat és mentés
    has_key: Callable[[str], bool]
    status: Callable[[], str]
    on_preview: Callable[[bool], None] | None = None  # Remote Play mód: minta + figyelt sáv
    on_close: Callable[[], None] | None = None  # None: bezáráskor csak elrejtjük
    on_forget_key: Callable[[str], None] | None = None  # mentett kulcs törlése
    is_key_saved: Callable[[str], bool] | None = None


class SettingsWindow:
    def __init__(self, config: AppConfig, callbacks: SettingsCallbacks) -> None:
        self._config = config
        self._callbacks = callbacks
        self._syncing = False
        self._next_status_at = 0.0
        self._visible = True
        self._root = tk.Tk()
        self._root.resizable(False, False)
        self._root.protocol("WM_DELETE_WINDOW", callbacks.on_close or self.toggle)
        self._build()
        self.sync(config)

    # --- felépítés -------------------------------------------------------
    def _build(self) -> None:
        self._built_language = current_language()
        self._root.title(t("settings.title"))
        outer = ttk.Frame(self._root, padding=12)
        outer.grid(sticky="nsew")
        self._content = outer
        self._build_ui_language(outer)
        frame = ttk.Frame(outer)
        frame.grid(row=1, column=0, sticky="nsew")
        self._build_translation(frame)
        self._build_appearance(frame)
        if self._callbacks.on_preview is not None:
            self._preview = tk.BooleanVar(value=False)
            ttk.Checkbutton(frame, text=t("settings.preview"),
                            variable=self._preview,
                            command=lambda: self._callbacks.on_preview(self._preview.get())
                            ).grid(row=12, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self._status = ttk.Label(frame, text="", wraplength=380, foreground="#555")
        self._status.grid(row=20, column=0, columnspan=3, sticky="w", pady=(10, 0))

    def _build_ui_language(self, outer: ttk.Frame) -> None:
        # A nyelvek a saját nevükön szerepelnek: az is megtalálja a sajátját, aki a mostanit nem érti
        self._ui_choices = {t("settings.ui_language_auto"): AUTO,
                            **{name: code for code, name in UI_LANGUAGES.items()}}
        self._ui_language = tk.StringVar()
        row = ttk.Frame(outer)
        row.grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Label(row, text=f"🌐 {t('settings.ui_language')}").grid(row=0, column=0, sticky="w", padx=(0, 8))
        box = ttk.Combobox(row, textvariable=self._ui_language, state="readonly", width=30,
                           values=list(self._ui_choices))
        box.grid(row=0, column=1, sticky="w")
        box.bind("<<ComboboxSelected>>",
                 lambda _e: self._change({"ui_language": self._ui_choices[self._ui_language.get()]}))

    def _build_translation(self, frame: ttk.Frame) -> None:
        self._language = tk.StringVar()
        self._translator = tk.StringVar()
        self._custom_code = tk.StringVar()
        self._key = tk.StringVar()
        self._model = tk.StringVar()
        ttk.Label(frame, text=t("settings.target_language")).grid(row=0, column=0, sticky="w")
        language_box = ttk.Combobox(frame, textvariable=self._language, state="readonly", width=26,
                                    values=[language.label for language in LANGUAGES])
        language_box.grid(row=0, column=1, sticky="w")
        language_box.bind("<<ComboboxSelected>>", lambda _e: self._on_language_selected())
        custom = ttk.Entry(frame, textvariable=self._custom_code, width=8)
        custom.grid(row=0, column=2, padx=(6, 0))
        custom.bind("<Return>", lambda _e: self._on_custom_code())
        ttk.Label(frame, text=t("settings.custom_code_hint"), foreground="#777").grid(
            row=1, column=1, columnspan=2, sticky="w")
        ttk.Label(frame, text=t("settings.translator")).grid(row=2, column=0, sticky="w", pady=(8, 0))
        translator_box = ttk.Combobox(frame, textvariable=self._translator, state="readonly", width=26,
                                      values=[translator_label(name) for name in TRANSLATOR_NAMES])
        translator_box.grid(row=2, column=1, sticky="w", pady=(8, 0))
        translator_box.bind("<<ComboboxSelected>>", lambda _e: self._on_translator_selected())
        ttk.Label(frame, text=t("settings.api_key")).grid(row=3, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self._key, show="•", width=28).grid(row=3, column=1, sticky="w")
        ttk.Button(frame, text=t("settings.save"), command=self._on_key).grid(row=3, column=2, padx=(6, 0))
        if self._callbacks.on_forget_key is not None:
            ttk.Button(frame, text=t("settings.delete"), command=self._on_forget_key).grid(row=3, column=3, padx=(6, 0))
        self._key_status = ttk.Label(frame, text="", foreground="#777", wraplength=300)
        self._key_status.grid(row=4, column=1, columnspan=2, sticky="w")
        ttk.Label(frame, text=t("settings.ai_model")).grid(row=5, column=0, sticky="w")
        self._model_entry = ttk.Entry(frame, textvariable=self._model, width=28)
        self._model_entry.grid(row=5, column=1, sticky="w")
        self._model_entry.bind("<Return>", lambda _e: self._on_model())

    def _build_appearance(self, frame: ttk.Frame) -> None:
        self._font = tk.DoubleVar()
        self._opacity = tk.DoubleVar()
        self._hide_original = tk.BooleanVar()
        self._only_spoken = tk.BooleanVar()
        self._translate_objectives = tk.BooleanVar()
        self._thin_font = tk.BooleanVar()
        self._visible_in_recordings = tk.BooleanVar()
        self._streamer_mode = tk.BooleanVar()
        self._clean_output = tk.BooleanVar()
        ttk.Separator(frame).grid(row=6, column=0, columnspan=3, sticky="ew", pady=10)
        ttk.Label(frame, text=t("settings.font_size")).grid(row=7, column=0, sticky="w")
        font_scale = tk.Scale(frame, variable=self._font, orient="horizontal", length=220,
                              from_=MIN_FONT_SIZE_RATIO * PERCENT, to=MAX_FONT_SIZE_RATIO * PERCENT,
                              resolution=0.1, command=lambda _v: self._on_appearance(save=False))
        font_scale.grid(row=7, column=1, columnspan=2, sticky="w")
        ttk.Label(frame, text=t("settings.background_opacity")).grid(row=8, column=0, sticky="w")
        opacity_scale = tk.Scale(frame, variable=self._opacity, orient="horizontal", length=220,
                                 from_=0, to=100, resolution=1,
                                 command=lambda _v: self._on_appearance(save=False))
        opacity_scale.grid(row=8, column=1, columnspan=2, sticky="w")
        for scale in (font_scale, opacity_scale):
            scale.bind("<ButtonRelease-1>", lambda _e: self._on_appearance(save=True))
        ttk.Checkbutton(frame, text=t("settings.hide_original"), variable=self._hide_original,
                        command=lambda: self._change({"hide_original": self._hide_original.get()})
                        ).grid(row=9, column=1, columnspan=2, sticky="w")
        ttk.Checkbutton(frame, text=t("settings.only_spoken"), variable=self._only_spoken,
                        command=lambda: self._change({"only_spoken_lines": self._only_spoken.get()})
                        ).grid(row=13, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Checkbutton(frame, text=t("settings.translate_objectives"), variable=self._translate_objectives,
                        command=lambda: self._change({"translate_objectives": self._translate_objectives.get()})
                        ).grid(row=14, column=0, columnspan=3, sticky="w")
        ttk.Checkbutton(frame, text=t("settings.thin_font"), variable=self._thin_font,
                        command=lambda: self._change(
                            {"font_path": THIN_FONT_PATH if self._thin_font.get() else BOLD_FONT_PATH})
                        ).grid(row=15, column=0, columnspan=3, sticky="w")
        ttk.Checkbutton(frame, text=t("settings.visible_in_recordings"),
                        variable=self._visible_in_recordings,
                        command=lambda: self._change({"visible_in_recordings": self._visible_in_recordings.get()})
                        ).grid(row=16, column=0, columnspan=3, sticky="w")
        ttk.Checkbutton(frame, text=t("settings.streamer_mode"), variable=self._streamer_mode,
                        command=lambda: self._change({"streamer_mode": self._streamer_mode.get()})
                        ).grid(row=17, column=0, columnspan=3, sticky="w")
        ttk.Checkbutton(frame, text=t("settings.clean_output"), variable=self._clean_output,
                        command=lambda: self._change({"clean_output": self._clean_output.get()})
                        ).grid(row=18, column=0, columnspan=3, sticky="w")
        ttk.Button(frame, text=t("settings.reset_position"),
                   command=lambda: self._change({"subtitle_x": None, "subtitle_y": None})
                   ).grid(row=10, column=1, columnspan=2, sticky="w", pady=(4, 0))
        preview_hint = t("settings.preview_hint_checkbox" if self._callbacks.on_preview
                         else "settings.preview_hint_key")
        ttk.Label(frame, foreground="#777", wraplength=380,
                  text=t("settings.panel_hint", preview_hint=preview_hint) + " " + t("layer.region_hint")).grid(
            row=11, column=0, columnspan=3, sticky="w", pady=(8, 0))

    # --- események -------------------------------------------------------
    def _change(self, updates: dict, save: bool = True) -> None:
        if not self._syncing:
            self._callbacks.on_change(updates, save)

    def _on_language_selected(self) -> None:
        chosen = next(lang for lang in LANGUAGES if lang.label == self._language.get())
        self._change({"target_lang": chosen.code})

    def _on_custom_code(self) -> None:
        code = self._custom_code.get().strip()
        if code:
            self._change({"target_lang": code})

    def _on_translator_selected(self) -> None:
        name = next(k for k in TRANSLATOR_NAMES if translator_label(k) == self._translator.get())
        self._change({"translator": name})

    def _on_key(self) -> None:
        name = self._config.translator
        if name in KEY_ENV_VARS and self._key.get().strip():
            self._callbacks.on_session_key(name, self._key.get().strip())
            self._key.set("")

    def _on_forget_key(self) -> None:
        name = self._config.translator
        if name in KEY_ENV_VARS and self._callbacks.on_forget_key is not None:
            self._callbacks.on_forget_key(name)

    def _on_model(self) -> None:
        field_name = AI_MODEL_FIELDS.get(self._config.translator)
        if field_name and self._model.get().strip():
            self._change({field_name: self._model.get().strip()})

    def _on_appearance(self, save: bool) -> None:
        self._change({"font_size_ratio": round(self._font.get() / PERCENT, 4),
                      "panel_opacity": int(round(self._opacity.get() / PERCENT * 255))}, save)

    # --- kívülről hívott -------------------------------------------------
    def sync(self, config: AppConfig) -> None:
        """Az ablak mezőit a jelenlegi beállításokhoz igazítja (pl. görgős méretezés után)."""
        self._config = config
        if current_language() != self._built_language:
            self._rebuild()
        self._syncing = True
        try:
            self._ui_language.set(next(label for label, code in self._ui_choices.items()
                                       if code == config.ui_language))
            self._language.set(find_language(config.target_lang).label)
            self._translator.set(translator_label(config.translator))
            self._font.set(round(config.font_size_ratio * PERCENT, 1))
            self._opacity.set(round(config.panel_opacity / 255 * PERCENT))
            self._hide_original.set(config.hide_original)
            self._only_spoken.set(config.only_spoken_lines)
            self._translate_objectives.set(config.translate_objectives)
            self._thin_font.set(config.font_path == THIN_FONT_PATH)
            self._visible_in_recordings.set(config.visible_in_recordings)
            self._streamer_mode.set(config.streamer_mode)
            self._clean_output.set(config.clean_output)
            field_name = AI_MODEL_FIELDS.get(config.translator)
            self._model.set(getattr(config, field_name) if field_name else "")
            self._model_entry.configure(state="normal" if field_name else "disabled")
            self._refresh_key_status()
        finally:
            self._syncing = False

    def _refresh_key_status(self) -> None:
        name = self._config.translator
        if name not in KEY_ENV_VARS:
            text = t("settings.key_not_needed" if name == "google" else "settings.key_auto")
        elif self._callbacks.is_key_saved is not None and self._callbacks.is_key_saved(name):
            text = t("settings.key_saved")
        elif self._callbacks.has_key(name):
            text = t("settings.key_session")
        else:
            text = t("settings.key_missing", variable=KEY_ENV_VARS[name])
        self._key_status.configure(text=text)

    def _rebuild(self) -> None:
        """A felületi nyelv váltása után az ablak az új nyelven épül újra."""
        # Csak a saját tartalmunkat bontjuk le: a gyökér alatt él a feliratréteg ablaka is
        self._content.destroy()
        self._build()

    def toggle(self) -> None:
        self._visible = not self._visible
        if self._visible:
            self._root.deiconify()
        else:
            self._root.withdraw()

    @property
    def root(self) -> tk.Tk:
        return self._root

    def refresh_status(self) -> None:
        now = time.monotonic()
        if now >= self._next_status_at:
            self._next_status_at = now + STATUS_REFRESH_SECONDS
            self._status.configure(text=self._callbacks.status())

    def pump(self) -> None:
        self.refresh_status()
        self._root.update()

    def close(self) -> None:
        try:
            self._root.destroy()
        except tk.TclError:
            pass  # már bezárták
