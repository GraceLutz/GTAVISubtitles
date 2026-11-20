"""A két mód (capture card ablak, Remote Play réteg) közös része: beállítások élő módosítása."""

import logging
from dataclasses import replace
from pathlib import Path

from .config import AppConfig, ConfigError, save_settings
from .ocr import SubtitleReader
from .subtitle_layer import SubtitleLayer
from .subtitle_pipeline import SubtitlePipeline
from .translation_service import TRANSLATOR_FIELDS, TranslationService, choice_from
from .translation_store import TranslationStore
from .translator import CachedTranslator

STORE_FILE_NAME = "mentett_forditasok.json"

logger = logging.getLogger(__name__)


class SubtitleAppBase:
    def __init__(self, config: AppConfig, reader: SubtitleReader,
                 translator: CachedTranslator | None, config_path: Path | None) -> None:
        self._config = config
        self._config_path = config_path
        store = TranslationStore(config_path.parent / STORE_FILE_NAME) if config_path else None
        self._translation = TranslationService(choice_from(config), translator, store)
        self._pipeline = SubtitlePipeline(config, reader, self._translation)
        self._subtitles = SubtitleLayer(config, on_change=self.apply_updates)
        self._settings = None

    def apply_updates(self, updates: dict, save: bool = True) -> None:
        """A panelből vagy egérrel jövő módosítás: azonnal életbe lép, és mentődik."""
        try:
            new_config = replace(self._config, **updates)
            new_config.validate()
        except (ConfigError, TypeError) as err:
            logger.warning("Érvénytelen beállítás: %s", err)
            return
        self._config = new_config
        self._subtitles.update_config(new_config)
        self._pipeline.update_config(new_config)
        if TRANSLATOR_FIELDS & set(updates):
            self._translation.rebuild(choice_from(new_config))
            self._pipeline.retranslate_current()
        if self._settings is not None:
            self._settings.sync(new_config)
        self._on_config_changed()
        if save:
            self._save(updates)

    def _on_config_changed(self) -> None:
        """Alosztály felülírhatja (pl. újrarajzolás)."""

    def _set_session_key(self, name: str, key: str) -> None:
        self._translation.set_session_key(name, key)
        self._pipeline.retranslate_current()
        if self._settings is not None:
            self._settings.sync(self._config)

    def _save(self, updates: dict) -> None:
        if self._config_path is None:
            return
        try:
            save_settings(self._config_path, updates)
        except (ConfigError, OSError) as err:
            logger.warning("A beállítást nem sikerült menteni: %s", err)

    def _create_settings_window(self, **extra):
        from .settings_window import SettingsCallbacks, SettingsWindow

        return SettingsWindow(self._config, SettingsCallbacks(
            on_change=self.apply_updates, on_session_key=self._set_session_key,
            has_key=self._translation.has_key, status=self._translation.status, **extra))
