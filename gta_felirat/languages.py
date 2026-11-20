"""Célnyelvek: kód, magyar név (a panelhez), angol név (az AI-nak), DeepL-kód."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str        # Google-kód is egyben
    label: str       # a beállítópanelen ez látszik
    english_name: str
    deepl_code: str | None  # None: a DeepL nem támogatja


LANGUAGES: tuple[Language, ...] = (
    Language("hu", "Magyar", "Hungarian", "HU"),
    Language("en", "Angol", "English", "EN-US"),
    Language("de", "Német", "German", "DE"),
    Language("fr", "Francia", "French", "FR"),
    Language("es", "Spanyol", "Spanish", "ES"),
    Language("it", "Olasz", "Italian", "IT"),
    Language("pt", "Portugál (brazil)", "Brazilian Portuguese", "PT-BR"),
    Language("nl", "Holland", "Dutch", "NL"),
    Language("pl", "Lengyel", "Polish", "PL"),
    Language("cs", "Cseh", "Czech", "CS"),
    Language("sk", "Szlovák", "Slovak", "SK"),
    Language("ro", "Román", "Romanian", "RO"),
    Language("hr", "Horvát", "Croatian", None),
    Language("sr", "Szerb", "Serbian", None),
    Language("sl", "Szlovén", "Slovenian", "SL"),
    Language("bg", "Bolgár", "Bulgarian", "BG"),
    Language("uk", "Ukrán", "Ukrainian", "UK"),
    Language("ru", "Orosz", "Russian", "RU"),
    Language("el", "Görög", "Greek", "EL"),
    Language("tr", "Török", "Turkish", "TR"),
    Language("sv", "Svéd", "Swedish", "SV"),
    Language("da", "Dán", "Danish", "DA"),
    Language("fi", "Finn", "Finnish", "FI"),
    Language("no", "Norvég", "Norwegian", "NB"),
    Language("ar", "Arab", "Arabic", "AR"),
    Language("hi", "Hindi", "Hindi", None),
    Language("zh-CN", "Kínai (egyszerűsített)", "Simplified Chinese", "ZH-HANS"),
    Language("ja", "Japán", "Japanese", "JA"),
    Language("ko", "Koreai", "Korean", "KO"),
    Language("vi", "Vietnámi", "Vietnamese", None),
    Language("id", "Indonéz", "Indonesian", "ID"),
)

_BY_CODE = {language.code.lower(): language for language in LANGUAGES}


def find_language(code: str) -> Language:
    """Ismeretlen kódnál is ad vissza nyelvet: az AI-nak a kódot adjuk át névként."""
    known = _BY_CODE.get(code.lower())
    if known is not None:
        return known
    return Language(code, code, code, code.upper())
