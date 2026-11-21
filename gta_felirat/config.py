"""Beállítások betöltése és ellenőrzése (config.json)."""

import json
import re
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from .i18n import AUTO, UI_LANGUAGES, t

SUPPORTED_TRANSLATORS = ("auto", "deepl", "claude", "openai", "google")
DEFAULT_CLAUDE_MODEL = "claude-haiku-5-5"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
MIN_FONT_SIZE_RATIO = 0.015
MAX_FONT_SIZE_RATIO = 0.08
# Vékony (a GTA-feliratokhoz hasonló) és félkövér betű; a panelen váltható
THIN_FONT_PATH = "C:/Windows/Fonts/segoeui.ttf"
BOLD_FONT_PATH = "C:/Windows/Fonts/segoeuib.ttf"
DEFAULT_FONT_PATH = THIN_FONT_PATH
# A beszélő nevének színe, mint a GTA-ban („Locals:” világos türkiz)
DEFAULT_SPEAKER_COLOR = "#7ECFD6"
HEX_COLOR_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")
MIN_WEBCAM_WIDTH_RATIO = 0.05
MAX_WEBCAM_WIDTH_RATIO = 0.5


class ConfigError(ValueError):
    """Hibás vagy hiányos beállítás."""


@dataclass(frozen=True)
class Region:
    """Képrészlet a teljes kép arányában (0.0–1.0)."""

    left: float = 0.12
    top: float = 0.78
    right: float = 0.88
    bottom: float = 0.98

    def validate(self) -> None:
        values = (self.left, self.top, self.right, self.bottom)
        if any(v < 0.0 or v > 1.0 for v in values):
            raise ConfigError(t("config.region_range"))
        if self.left >= self.right or self.top >= self.bottom:
            raise ConfigError(t("config.region_order"))


@dataclass(frozen=True)
class AppConfig:
    device_index: int = 0
    capture_width: int = 1920
    capture_height: int = 1080
    capture_fps: int = 60
    subtitle_region: Region = field(default_factory=Region)
    ocr_interval_seconds: float = 0.1
    min_text_score: float = 0.6
    stable_reads: int = 2
    similarity_threshold: float = 0.85
    hold_seconds: float = 0.8
    translator: str = "auto"
    source_lang: str = "en"
    target_lang: str = "hu"
    claude_model: str = DEFAULT_CLAUDE_MODEL
    openai_model: str = DEFAULT_OPENAI_MODEL
    hide_original: bool = True
    # A réteg látszik a monitorrögzítésben (Medal, OBS → Kick, Twitch, YouTube)
    visible_in_recordings: bool = True
    only_spoken_lines: bool = True  # csak a „Név: szöveg” párbeszédsorokat fordítja
    # Remote Play módban a feliratsáv folyamatos rögzítése külön szálon (gyorsabb takarás és olvasás);
    # false: a régi, olvasásonkénti rögzítés, ha a folyamatos valamiért gondot okozna
    fast_capture: bool = True
    # Remote Play módban soronkénti olvasás: csak az új feliratsort olvassuk be (gyorsabb fordítás, kisebb
    # terhelés); a folyamatos rögzítésre épül. false: mindig a teljes sávot olvassuk
    row_reading: bool = True
    panel_opacity: int = 245
    subtitle_x: float | None = None  # None: az eredeti felirat helyén
    subtitle_y: float | None = None
    font_path: str = DEFAULT_FONT_PATH
    font_size_ratio: float = 0.034
    speaker_color: str = DEFAULT_SPEAKER_COLOR  # "#RRGGBB"
    fullscreen: bool = True
    webcam_device_index: int | None = None
    webcam_x: float = 0.02
    webcam_y: float = 0.02
    webcam_width_ratio: float = 0.2
    webcam_mirror: bool = True
    ui_language: str = AUTO  # "auto": a Windows nyelve, vagy pl. "hu"

    def validate(self) -> None:
        self._validate_webcam()
        self._validate_subtitle_panel()
        self.subtitle_region.validate()
        if self.translator not in SUPPORTED_TRANSLATORS:
            raise ConfigError(t("config.translator_choice", choices=SUPPORTED_TRANSLATORS))
        if self.device_index < 0:
            raise ConfigError(t("config.device_negative"))
        if self.stable_reads < 1:
            raise ConfigError(t("config.stable_reads_min"))
        if not 0.0 < self.similarity_threshold <= 1.0:
            raise ConfigError(t("config.similarity_range"))
        if self.ocr_interval_seconds <= 0 or self.hold_seconds < 0:
            raise ConfigError(t("config.timing_range"))
        if not 0.0 <= self.min_text_score <= 1.0:
            raise ConfigError(t("config.min_score_range"))
        if self.ui_language != AUTO and self.ui_language not in UI_LANGUAGES:
            raise ConfigError(t("config.ui_language_choice", choices=sorted(UI_LANGUAGES)))

    def _validate_subtitle_panel(self) -> None:
        if not self.target_lang.strip():
            raise ConfigError(t("config.target_lang_empty"))
        if not MIN_FONT_SIZE_RATIO <= self.font_size_ratio <= MAX_FONT_SIZE_RATIO:
            raise ConfigError(t("config.font_size_range", minimum=MIN_FONT_SIZE_RATIO, maximum=MAX_FONT_SIZE_RATIO))
        if not 0 <= self.panel_opacity <= 255:
            raise ConfigError(t("config.opacity_range"))
        if not isinstance(self.speaker_color, str) or not HEX_COLOR_PATTERN.match(self.speaker_color):
            raise ConfigError(t("config.speaker_color_format"))
        if (self.subtitle_x is None) != (self.subtitle_y is None):
            raise ConfigError(t("config.subtitle_xy_pair"))
        if self.subtitle_x is not None and not (0.0 <= self.subtitle_x <= 1.0 and 0.0 <= self.subtitle_y <= 1.0):
            raise ConfigError(t("config.subtitle_xy_range"))

    def _validate_webcam(self) -> None:
        if not (0.0 <= self.webcam_x <= 1.0 and 0.0 <= self.webcam_y <= 1.0):
            raise ConfigError(t("config.webcam_xy_range"))
        if not MIN_WEBCAM_WIDTH_RATIO <= self.webcam_width_ratio <= MAX_WEBCAM_WIDTH_RATIO:
            raise ConfigError(t("config.webcam_width_range", minimum=MIN_WEBCAM_WIDTH_RATIO,
                                maximum=MAX_WEBCAM_WIDTH_RATIO))
        if self.webcam_device_index is None:
            return
        if self.webcam_device_index < 0:
            raise ConfigError(t("config.webcam_device_negative"))
        if self.webcam_device_index == self.device_index:
            raise ConfigError(t("config.webcam_same_device"))


def config_from_dict(raw: dict) -> AppConfig:
    known = {f.name for f in fields(AppConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(t("config.unknown_settings", names=sorted(unknown)))
    values = dict(raw)
    if "subtitle_region" in values:
        region_raw = values["subtitle_region"]
        if not isinstance(region_raw, dict):
            raise ConfigError(t("config.region_not_object"))
        try:
            values["subtitle_region"] = Region(**region_raw)
        except TypeError as err:
            raise ConfigError(t("config.region_invalid", err=err)) from err
    config = AppConfig(**values)
    config.validate()
    return config


def load_config(path: Path | None) -> AppConfig:
    if path is None or not path.exists():
        config = AppConfig()
        config.validate()
        return config
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise ConfigError(t("config.invalid_json", path=path, err=err)) from err
    if not isinstance(raw, dict):
        raise ConfigError(t("config.root_not_object", path=path))
    return config_from_dict(raw)


def with_overrides(config: AppConfig, **overrides) -> AppConfig:
    present = {k: v for k, v in overrides.items() if v is not None}
    updated = replace(config, **present)
    updated.validate()
    return updated


def save_settings(path: Path, updates: dict) -> None:
    """A megadott kulcsokat beírja a config.json-ba; a többi beállítást nem bántja."""
    unknown = set(updates) - {f.name for f in fields(AppConfig)}
    if unknown:
        raise ConfigError(t("config.unknown_setting_save", names=sorted(unknown)))
    raw = {}
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as err:
            raise ConfigError(t("config.invalid_json_not_saved", path=path, err=err)) from err
    rounded = {k: round(v, 4) if isinstance(v, float) else v for k, v in updates.items()}
    temp_path = path.with_suffix(".tmp")
    content = json.dumps({**raw, **rounded}, indent=2, ensure_ascii=False) + "\n"
    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(path)
