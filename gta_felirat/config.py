"""Beállítások betöltése és ellenőrzése (config.json)."""

import json
import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path

from .i18n import AUTO, UI_LANGUAGES, t

SUPPORTED_TRANSLATORS = ("auto", "deepl", "claude", "openai", "google")
# A capture card megnyitásának módja: auto (DirectShow, ha az nem megy, Media Foundation), dshow, msmf
SUPPORTED_CAPTURE_APIS = ("auto", "dshow", "msmf")
DEFAULT_CLAUDE_MODEL = "claude-haiku-5-5"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
MIN_FONT_SIZE_RATIO = 0.015
MAX_FONT_SIZE_RATIO = 0.08
MIN_PANEL_WIDTH_RATIO = 0.3  # ennél keskenyebb panelben a magyar mondat túl sok sorba törne
MAX_PANEL_WIDTH_RATIO = 1.0
MIN_PANEL_HEIGHT_RATIO = 0.04  # kb. egy sor
MAX_PANEL_HEIGHT_RATIO = 0.5
# Vékony (a GTA-feliratokhoz hasonló) és félkövér betű; a panelen váltható
THIN_FONT_PATH = "C:/Windows/Fonts/segoeui.ttf"
BOLD_FONT_PATH = "C:/Windows/Fonts/segoeuib.ttf"
DEFAULT_FONT_PATH = THIN_FONT_PATH
# A beszélő nevének színe, mint a GTA-ban („Locals:” világos türkiz)
DEFAULT_SPEAKER_COLOR = "#7ECFD6"
HEX_COLOR_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")
MIN_WEBCAM_WIDTH_RATIO = 0.05
MAX_WEBCAM_WIDTH_RATIO = 0.5
# A tiszta kép késleltetése: rövidebbnél a fordítás nem ér be, hosszabbnál a nézőnek már zavaró
MIN_CLEAN_DELAY_SECONDS = 0.5
MAX_CLEAN_DELAY_SECONDS = 5.0


class ConfigError(ValueError):
    """Hibás vagy hiányos beállítás."""


@dataclass(frozen=True)
class Region:
    """Képrészlet a teljes kép arányában (0.0–1.0)."""

    left: float = 0.06  # széles sáv: a hosszú feliratsor széle se lógjon ki belőle
    # 4-5 beszélő egymás fölött, soronként akár két képernyősorral is beférjen (a játék a régebbi sorokat
    # felfelé tolja; 1080p-ben egy sor kb. 40 px, a legalsó kb. a 985. képpontnál ér véget)
    top: float = 0.55
    right: float = 0.94
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
    capture_api: str = "auto"
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
    translate_objectives: bool = True  # a küldetésutasítás sorát is lefordítja, a saját sorára
    streamer_mode: bool = False  # a sértő szavakat (n-szó stb.) kicsillagozva mutatja (streamer_filter)
    # Külön ablak a nézőknek (OBS, Medal): a játékkép késleltetve, rajta már csak a kész magyar felirat
    clean_output: bool = False
    clean_output_delay_seconds: float = 1.0
    # A kiírt magyar felirat az exe melletti obs_felirat.txt-be is (OBS: Szöveg forrás, „Olvasás fájlból”);
    # a tiszta kép bekapcsolásakor annak késésével
    obs_text_file: bool = True
    # Remote Play módban a feliratsáv folyamatos rögzítése külön szálon (gyorsabb takarás és olvasás);
    # false: a régi, olvasásonkénti rögzítés, ha a folyamatos valamiért gondot okozna
    fast_capture: bool = True
    # Remote Play módban soronkénti olvasás: csak az új feliratsort olvassuk be (gyorsabb fordítás, kisebb
    # terhelés); a folyamatos rögzítésre épül. false: mindig a teljes sávot olvassuk
    row_reading: bool = True
    # A feliratsáv egészében keresünk szöveget, nem csak a kép közepén: ha a játék máshova (oldalra, más sorba)
    # teszi a szöveget, azt is kiolvassuk. false: csak a középre igazított felirat, mint korábban
    whole_band_text: bool = True
    panel_opacity: int = 245
    # A fordítás legfeljebb ilyen széles (a kép szélességéhez mérve); a Minta módban a panel szélénél húzva állítható
    panel_width_ratio: float = 0.9
    # A panel legalább ilyen magas (a kép magasságához mérve); None: épp akkora, amekkora a szöveg.
    # A Minta módban a panel tetejét vagy alját húzva állítható
    panel_height_ratio: float | None = None
    subtitle_x: float | None = None  # None: az eredeti felirat helyén
    subtitle_y: float | None = None
    font_path: str = DEFAULT_FONT_PATH
    font_size_ratio: float = 0.034
    speaker_color: str = DEFAULT_SPEAKER_COLOR  # "#RRGGBB"
    # A beszélőnevek tanulása, a program maga írja: {"learned": [új nevek], "fixes": {"Raut": {"name": "Raul",
    # "count": 12}}}; a hibás forma nem akadályozza az indulást, a program figyelmen kívül hagyja
    speaker_names: dict = field(default_factory=dict)
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
        if self.capture_api not in SUPPORTED_CAPTURE_APIS:
            raise ConfigError(t("config.capture_api_choice", choices=SUPPORTED_CAPTURE_APIS))
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
        if not MIN_CLEAN_DELAY_SECONDS <= self.clean_output_delay_seconds <= MAX_CLEAN_DELAY_SECONDS:
            raise ConfigError(t("config.clean_delay_range", minimum=MIN_CLEAN_DELAY_SECONDS,
                                maximum=MAX_CLEAN_DELAY_SECONDS))

    def _validate_subtitle_panel(self) -> None:
        if not self.target_lang.strip():
            raise ConfigError(t("config.target_lang_empty"))
        if not MIN_FONT_SIZE_RATIO <= self.font_size_ratio <= MAX_FONT_SIZE_RATIO:
            raise ConfigError(t("config.font_size_range", minimum=MIN_FONT_SIZE_RATIO, maximum=MAX_FONT_SIZE_RATIO))
        if not 0 <= self.panel_opacity <= 255:
            raise ConfigError(t("config.opacity_range"))
        if not MIN_PANEL_WIDTH_RATIO <= self.panel_width_ratio <= MAX_PANEL_WIDTH_RATIO:
            raise ConfigError(t("config.panel_width_range", minimum=MIN_PANEL_WIDTH_RATIO,
                                maximum=MAX_PANEL_WIDTH_RATIO))
        if self.panel_height_ratio is not None and not (
                MIN_PANEL_HEIGHT_RATIO <= self.panel_height_ratio <= MAX_PANEL_HEIGHT_RATIO):
            raise ConfigError(t("config.panel_height_range", minimum=MIN_PANEL_HEIGHT_RATIO,
                                maximum=MAX_PANEL_HEIGHT_RATIO))
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
        # A webkamera DirectShow-sorszámmal nyílik; Media Foundation-ön a kártya ugyanazzal a számmal más eszköz lehet
        if self.webcam_device_index == self.device_index and self.capture_api != "msmf":
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
    rounded = {k: _saved_value(v) for k, v in updates.items()}
    temp_path = path.with_suffix(".tmp")
    content = json.dumps({**raw, **rounded}, indent=2, ensure_ascii=False) + "\n"
    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(path)


def _saved_value(value):
    """JSON-ba írható alak: a tört négy tizedesre kerekítve, a sáv (Region) objektumként."""
    if is_dataclass(value):
        return {key: _saved_value(item) for key, item in asdict(value).items()}
    return round(value, 4) if isinstance(value, float) else value
