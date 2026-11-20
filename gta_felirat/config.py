"""Beállítások betöltése és ellenőrzése (config.json)."""

import json
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

SUPPORTED_TRANSLATORS = ("auto", "deepl", "claude", "openai", "google")
DEFAULT_CLAUDE_MODEL = "claude-haiku-5-5"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
MIN_FONT_SIZE_RATIO = 0.015
MAX_FONT_SIZE_RATIO = 0.08
DEFAULT_FONT_PATH = "C:/Windows/Fonts/segoeuib.ttf"
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
            raise ConfigError("subtitle_region értékei 0.0 és 1.0 között legyenek")
        if self.left >= self.right or self.top >= self.bottom:
            raise ConfigError("subtitle_region: left < right és top < bottom kell")


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
    only_spoken_lines: bool = True  # csak a „Név: szöveg” párbeszédsorokat fordítja
    panel_opacity: int = 245
    subtitle_x: float | None = None  # None: az eredeti felirat helyén
    subtitle_y: float | None = None
    font_path: str = DEFAULT_FONT_PATH
    font_size_ratio: float = 0.034
    fullscreen: bool = True
    webcam_device_index: int | None = None
    webcam_x: float = 0.02
    webcam_y: float = 0.02
    webcam_width_ratio: float = 0.2
    webcam_mirror: bool = True

    def validate(self) -> None:
        self._validate_webcam()
        self._validate_subtitle_panel()
        self.subtitle_region.validate()
        if self.translator not in SUPPORTED_TRANSLATORS:
            raise ConfigError(f"translator csak ezek egyike lehet: {SUPPORTED_TRANSLATORS}")
        if self.device_index < 0:
            raise ConfigError("device_index nem lehet negatív")
        if self.stable_reads < 1:
            raise ConfigError("stable_reads legalább 1 legyen")
        if not 0.0 < self.similarity_threshold <= 1.0:
            raise ConfigError("similarity_threshold 0 és 1 közé essen")
        if self.ocr_interval_seconds <= 0 or self.hold_seconds < 0:
            raise ConfigError("ocr_interval_seconds > 0 és hold_seconds >= 0 kell")
        if not 0.0 <= self.min_text_score <= 1.0:
            raise ConfigError("min_text_score 0 és 1 közé essen")

    def _validate_subtitle_panel(self) -> None:
        if not self.target_lang.strip():
            raise ConfigError("target_lang nem lehet üres")
        if not MIN_FONT_SIZE_RATIO <= self.font_size_ratio <= MAX_FONT_SIZE_RATIO:
            raise ConfigError(f"font_size_ratio {MIN_FONT_SIZE_RATIO} és {MAX_FONT_SIZE_RATIO} közé essen")
        if not 0 <= self.panel_opacity <= 255:
            raise ConfigError("panel_opacity 0 és 255 közé essen")
        if (self.subtitle_x is None) != (self.subtitle_y is None):
            raise ConfigError("subtitle_x és subtitle_y vagy mindkettő null, vagy mindkettő szám")
        if self.subtitle_x is not None and not (0.0 <= self.subtitle_x <= 1.0 and 0.0 <= self.subtitle_y <= 1.0):
            raise ConfigError("subtitle_x és subtitle_y 0.0 és 1.0 közé essen")

    def _validate_webcam(self) -> None:
        if not (0.0 <= self.webcam_x <= 1.0 and 0.0 <= self.webcam_y <= 1.0):
            raise ConfigError("webcam_x és webcam_y 0.0 és 1.0 közé essen")
        if not MIN_WEBCAM_WIDTH_RATIO <= self.webcam_width_ratio <= MAX_WEBCAM_WIDTH_RATIO:
            raise ConfigError(f"webcam_width_ratio {MIN_WEBCAM_WIDTH_RATIO} és "
                              f"{MAX_WEBCAM_WIDTH_RATIO} közé essen")
        if self.webcam_device_index is None:
            return
        if self.webcam_device_index < 0:
            raise ConfigError("webcam_device_index nem lehet negatív")
        if self.webcam_device_index == self.device_index:
            raise ConfigError("A webkamera és a capture card sorszáma nem lehet ugyanaz")


def config_from_dict(raw: dict) -> AppConfig:
    known = {f.name for f in fields(AppConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"Ismeretlen beállítás(ok): {sorted(unknown)}")
    values = dict(raw)
    if "subtitle_region" in values:
        region_raw = values["subtitle_region"]
        if not isinstance(region_raw, dict):
            raise ConfigError("subtitle_region objektum legyen (left, top, right, bottom)")
        try:
            values["subtitle_region"] = Region(**region_raw)
        except TypeError as err:
            raise ConfigError(f"subtitle_region hibás: {err}") from err
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
        raise ConfigError(f"{path} nem érvényes JSON: {err}") from err
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} gyökere objektum legyen")
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
        raise ConfigError(f"Ismeretlen beállítás mentése: {sorted(unknown)}")
    raw = {}
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as err:
            raise ConfigError(f"{path} nem érvényes JSON, nem mentem felül: {err}") from err
    rounded = {k: round(v, 4) if isinstance(v, float) else v for k, v in updates.items()}
    temp_path = path.with_suffix(".tmp")
    content = json.dumps({**raw, **rounded}, indent=2, ensure_ascii=False) + "\n"
    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(path)
