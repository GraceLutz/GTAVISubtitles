"""Célnyelvek: kód, saját nyelvű név (a panelhez), angol név (az AI-nak), DeepL-kód."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str        # Google-kód is egyben
    label: str       # a beállítópanelen ez látszik: a nyelv saját nevén
    english_name: str
    deepl_code: str | None  # None: a DeepL nem támogatja


LANGUAGES: tuple[Language, ...] = (
    Language("hu", "Magyar", "Hungarian", "HU"),
    Language("en", "English", "English", "EN-US"),
    Language("de", "Deutsch", "German", "DE"),
    Language("fr", "Français", "French", "FR"),
    Language("es", "Español", "Spanish", "ES"),
    Language("it", "Italiano", "Italian", "IT"),
    Language("pt", "Português (Brasil)", "Brazilian Portuguese", "PT-BR"),
    Language("nl", "Nederlands", "Dutch", "NL"),
    Language("pl", "Polski", "Polish", "PL"),
    Language("cs", "Čeština", "Czech", "CS"),
    Language("sk", "Slovenčina", "Slovak", "SK"),
    Language("ro", "Română", "Romanian", "RO"),
    Language("hr", "Hrvatski", "Croatian", None),
    Language("sr", "Српски", "Serbian", None),
    Language("sl", "Slovenščina", "Slovenian", "SL"),
    Language("bg", "Български", "Bulgarian", "BG"),
    Language("uk", "Українська", "Ukrainian", "UK"),
    Language("ru", "Русский", "Russian", "RU"),
    Language("el", "Ελληνικά", "Greek", "EL"),
    Language("tr", "Türkçe", "Turkish", "TR"),
    Language("sv", "Svenska", "Swedish", "SV"),
    Language("da", "Dansk", "Danish", "DA"),
    Language("fi", "Suomi", "Finnish", "FI"),
    Language("no", "Norsk", "Norwegian", "NB"),
    Language("ar", "العربية", "Arabic", "AR"),
    Language("hi", "हिन्दी", "Hindi", None),
    Language("zh-CN", "中文（简体）", "Simplified Chinese", "ZH-HANS"),
    Language("ja", "日本語", "Japanese", "JA"),
    Language("ko", "한국어", "Korean", "KO"),
    Language("vi", "Tiếng Việt", "Vietnamese", None),
    Language("id", "Bahasa Indonesia", "Indonesian", "ID"),
)

_BY_CODE = {language.code.lower(): language for language in LANGUAGES}


def find_language(code: str) -> Language:
    """Ismeretlen kódnál is ad vissza nyelvet: az AI-nak a kódot adjuk át névként."""
    known = _BY_CODE.get(code.lower())
    if known is not None:
        return known
    return Language(code, code, code, code.upper())
