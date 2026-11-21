import re
import string
from pathlib import Path

import pytest

from gta_felirat import i18n
from gta_felirat.config import AppConfig, ConfigError, config_from_dict
from gta_felirat.locales import en

PACKAGE_DIR = Path(i18n.__file__).parent
KEY_USE = re.compile(r"""\bt\(\s*["']([a-z0-9_.]+)["']""")


def placeholders(text: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


@pytest.fixture(autouse=True)
def restore_language():
    yield
    i18n.set_language("en")


def test_every_key_used_in_the_code_exists_in_english():
    used = set()
    for path in PACKAGE_DIR.glob("*.py"):
        used |= set(KEY_USE.findall(path.read_text(encoding="utf-8")))
    assert used, "a kód még egy t() hívást sem tartalmaz"
    assert sorted(used - set(en.MESSAGES)) == []


@pytest.mark.parametrize("code", sorted(i18n.UI_LANGUAGES))
def test_translations_only_use_known_keys_and_the_same_placeholders(code):
    catalog = i18n.catalog(code)
    assert sorted(set(catalog) - set(en.MESSAGES)) == []
    for key, text in catalog.items():
        assert placeholders(text) == placeholders(en.MESSAGES[key]), key


def test_hungarian_is_complete():
    assert sorted(set(en.MESSAGES) - set(i18n.catalog("hu"))) == []


def test_text_is_returned_in_the_chosen_language_with_values():
    i18n.set_language("hu")
    assert i18n.t("settings.title") == i18n.catalog("hu")["settings.title"]
    i18n.set_language("en")
    assert i18n.t("settings.title") == en.MESSAGES["settings.title"]


def test_missing_translation_falls_back_to_english(monkeypatch):
    monkeypatch.setitem(i18n._CATALOGS, "de", {})
    i18n.set_language("de")
    assert i18n.t("settings.title") == en.MESSAGES["settings.title"]


def test_placeholders_are_filled():
    i18n.set_language("en")
    text = i18n.t("settings.key_missing", variable="DEEPL_API_KEY")
    assert "DEEPL_API_KEY" in text


@pytest.mark.parametrize("system, expected", [
    ("hu-HU", "hu"), ("de-AT", "de"), ("pt-BR", "pt"), ("en-GB", "en"), ("ja-JP", "en"), ("", "en"),
])
def test_auto_follows_the_windows_language(monkeypatch, system, expected):
    monkeypatch.setattr(i18n, "_system_locale_name", lambda: system)
    assert i18n.resolve_language(i18n.AUTO) == expected


def test_explicit_language_wins_over_windows(monkeypatch):
    monkeypatch.setattr(i18n, "_system_locale_name", lambda: "hu-HU")
    assert i18n.resolve_language("fr") == "fr"


def test_config_accepts_auto_and_supported_languages_only():
    assert AppConfig().ui_language == i18n.AUTO
    assert config_from_dict({"ui_language": "pl"}).ui_language == "pl"
    with pytest.raises(ConfigError):
        config_from_dict({"ui_language": "xx"})
