"""A program felületének nyelve: a t(key) a választott nyelven adja vissza a szöveget.

A szövegek a locales/ modulokban vannak (Python-szótárak, így a PyInstaller-csomagba is
bekerülnek). Ha egy fordítás hiányzik, az angol szöveg jelenik meg. Alapból a Windows
felületének nyelvét követjük ("auto"); a beállítóablakban átállítható.
"""

import ctypes
import locale
import logging
import sys

from .locales import de, en, es, fr, hu, pl, pt

AUTO = "auto"
DEFAULT_LANGUAGE = "en"
# A nyelvek a saját nevükön, hogy mindenki megtalálja a sajátját
UI_LANGUAGES = {
    "en": "English",
    "hu": "Magyar",
    "de": "Deutsch",
    "es": "Español",
    "fr": "Français",
    "pl": "Polski",
    "pt": "Português",
}
_CATALOGS = {"en": en.MESSAGES, "hu": hu.MESSAGES, "de": de.MESSAGES, "es": es.MESSAGES,
             "fr": fr.MESSAGES, "pl": pl.MESSAGES, "pt": pt.MESSAGES}
LOCALE_NAME_MAX_LENGTH = 85

logger = logging.getLogger(__name__)
_current = DEFAULT_LANGUAGE


def _system_locale_name() -> str:
    """A Windows felületének nyelve, pl. "hu-HU" (más rendszeren a locale beállítása)."""
    if sys.platform == "win32":
        buffer = ctypes.create_unicode_buffer(LOCALE_NAME_MAX_LENGTH)
        if ctypes.windll.kernel32.GetUserDefaultLocaleName(buffer, LOCALE_NAME_MAX_LENGTH):
            return buffer.value
    return (locale.getlocale()[0] or "").replace("_", "-")


def resolve_language(setting: str) -> str:
    """A beállításból ("auto" vagy nyelvkód) a ténylegesen használt felületi nyelv."""
    code = setting
    if setting == AUTO:
        code = _system_locale_name().split("-")[0].lower()
    return code if code in UI_LANGUAGES else DEFAULT_LANGUAGE


def set_language(setting: str) -> str:
    global _current
    _current = resolve_language(setting)
    return _current


def current_language() -> str:
    return _current


def catalog(code: str) -> dict[str, str]:
    return _CATALOGS[code]


def t(key: str, **values) -> str:
    """A szöveg a jelenlegi nyelven; ha nincs lefordítva, angolul."""
    text = _CATALOGS[_current].get(key) or en.MESSAGES.get(key)
    if text is None:
        logger.warning("Missing UI text: %s", key)
        return key
    return text.format(**values) if values else text
