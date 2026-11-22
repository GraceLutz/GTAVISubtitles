"""A két mód (capture card ablak, Remote Play réteg) közös része: beállítások élő módosítása."""

import logging
import sys
from dataclasses import replace
from pathlib import Path

from . import speaker_names
from .config import AppConfig, ConfigError, save_settings
from .i18n import set_language, t
from .key_vault import KeyVault, KeyVaultError, WindowsCredentialVault
from .ocr import SubtitleReader
from .region_editor import RegionEditor
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
        if config_path:
            # A tanult nevek és a névjavítások a config.json-ba kerülnek, a program magától írja
            speaker_names.use_config(config.speaker_names,
                                     save=lambda data: self._save({"speaker_names": data}))
        self._translation = TranslationService(choice_from(config), translator, store,
                                               vault=_open_key_vault() if config_path else None)
        self._pipeline = self._create_pipeline(config, reader)
        self._subtitles = SubtitleLayer(config, on_change=self.apply_updates)
        # A figyelt sáv egérrel (csak beállításkor, amikor a sáv látszik)
        self._region_editor = RegionEditor(config.subtitle_region, on_change=self.apply_updates)
        self._settings = None

    def _create_pipeline(self, config: AppConfig, reader: SubtitleReader) -> SubtitlePipeline:
        """Alosztály felülírhatja (a Remote Play mód soralapú pipeline-t használ)."""
        return SubtitlePipeline(config, reader, self._translation)

    def apply_updates(self, updates: dict, save: bool = True) -> None:
        """A panelből vagy egérrel jövő módosítás: azonnal életbe lép, és mentődik."""
        try:
            new_config = replace(self._config, **updates)
            new_config.validate()
        except (ConfigError, TypeError) as err:
            logger.warning(t("app.invalid_setting", err=err))
            return
        self._config = new_config
        if "ui_language" in updates:
            set_language(new_config.ui_language)  # a beállítóablak a sync()-ben újraépül
        self._subtitles.update_config(new_config)
        self._region_editor.update_region(new_config.subtitle_region)
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

    def _forget_key(self, name: str) -> None:
        self._translation.forget_key(name)
        if self._settings is not None:
            self._settings.sync(self._config)

    def _save(self, updates: dict) -> None:
        if self._config_path is None:
            return
        try:
            save_settings(self._config_path, updates)
        except (ConfigError, OSError) as err:
            logger.warning(t("app.save_failed", err=err))

    def _create_settings_window(self, **extra):
        from .settings_window import SettingsCallbacks, SettingsWindow

        return SettingsWindow(self._config, SettingsCallbacks(
            on_change=self.apply_updates, on_session_key=self._set_session_key,
            has_key=self._translation.has_key, status=self._translation.status,
            on_forget_key=self._forget_key, is_key_saved=self._translation.is_key_saved, **extra))


def _open_key_vault() -> KeyVault | None:
    """A Windows Hitelesítőadat-kezelő; más rendszeren (vagy ha nem elérhető) nincs kulcsmentés."""
    if sys.platform != "win32":
        return None
    try:
        return WindowsCredentialVault()
    except (OSError, KeyVaultError) as err:
        logger.warning(t("app.key_vault_unavailable", err=err))
        return None
