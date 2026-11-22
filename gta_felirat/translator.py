"""Fordító kiválasztása és gyorsítótárazott fordítás (a beszélő nevét megtartva).

Egy többsoros felirat még le nem fordított sorai egyetlen kérésben mennek el, ha a
szolgáltatás tudja (translate_many); a gyorsítótár soronkénti, mert a játék görgeti a sorokat.
"""

import logging
import os
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Mapping, Protocol, TypeVar, runtime_checkable

from .backends import ClaudeBackend, DeepLBackend, GoogleBackend, OpenAIBackend
from .i18n import t
from .languages import find_language
from .translation_store import TranslationStore

CACHE_SIZE = 2000
MAX_SPEAKER_NAME_LENGTH = 24
# „Jason: Gyere már!” — a beszélő nevét nem fordítjuk le
_SPEAKER_PREFIX = re.compile(r"^([A-Z][\w.'&\- ]{0,%d}):\s+(.+)$" % (MAX_SPEAKER_NAME_LENGTH - 1))
# Új beszélő kezdete egy lezárt mondat után: „… breath now. Lucia: Stop talking.”
# „In Spanish”: nem beszélő neve, hanem megjegyzés a nyelvről, ezért ezt is lefordítjuk
_LANGUAGE_NOTE = re.compile(r"^In [A-Z][a-z]+$")
_DOUBLE_QUOTES = "\"„”“«»"
_SPEAKER_BOUNDARY = re.compile(r"(?<=[.!?…])\s+(?=[A-Z](?:[^\W\d_]|[ .'&-]){0,30}:\s)")

KEY_ENV_VARS = {"deepl": "DEEPL_API_KEY", "claude": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
# „auto” esetén ebben a sorrendben választunk abból, amihez van kulcs
AUTO_PRIORITY = ("deepl", "claude", "openai")
TRANSLATOR_NAMES = ("auto", "deepl", "claude", "openai", "google")
_BRAND_LABELS = {"deepl": "DeepL", "claude": "Claude (Anthropic)", "openai": "ChatGPT (OpenAI)"}

_SpeakerLine = tuple[str, str]  # (beszélő vagy nyelvi megjegyzés – lehet üres, a kimondott szöveg)
_Asked = TypeVar("_Asked")
_Answered = TypeVar("_Answered")


def translator_label(name: str) -> str:
    """A fordító neve a felületen (az „Automatikus” és az „ingyenes” a felület nyelvén)."""
    if name in _BRAND_LABELS:
        return _BRAND_LABELS[name]
    if name in ("auto", "google"):
        return t(f"translator.{name}")
    return name

logger = logging.getLogger(__name__)


class TranslationError(RuntimeError):
    """A fordítószolgáltatás hibát adott vagy nem érhető el."""


class TranslationBackend(Protocol):
    def translate(self, text: str) -> str: ...


@runtime_checkable
class BatchTranslationBackend(TranslationBackend, Protocol):
    """Választható bővítés (pl. DeepL): több sor egy kérésben, soronként egy válasz a sorok sorrendjében."""

    def translate_many(self, texts: list[str]) -> list[str]: ...


@dataclass(frozen=True)
class TranslatorChoice:
    name: str          # auto, deepl, claude, openai, google
    target_lang: str   # nyelvkód, pl. hu
    claude_model: str
    openai_model: str


def resolve_key(name: str, session_keys: Mapping[str, str]) -> str:
    """A panelen beírt kulcs elsőbbséget élvez a környezeti változóval szemben."""
    session = session_keys.get(name, "").strip()
    if session:
        return session
    return os.environ.get(KEY_ENV_VARS.get(name, ""), "").strip()


def resolve_backend_name(name: str, session_keys: Mapping[str, str]) -> str:
    if name != "auto":
        return name
    for candidate in AUTO_PRIORITY:
        if resolve_key(candidate, session_keys):
            return candidate
    return "google"


def create_backend(choice: TranslatorChoice, session_keys: Mapping[str, str] | None = None
                   ) -> TranslationBackend:
    keys = session_keys or {}
    name = resolve_backend_name(choice.name, keys)
    target = find_language(choice.target_lang)
    if name == "google":
        logger.info(t("translator.active", translator="Google", language=target.label))
        return GoogleBackend(target)
    api_key = resolve_key(name, keys)
    if not api_key:
        raise TranslationError(t("translator.missing_key", translator=translator_label(name),
                                 variable=KEY_ENV_VARS[name]))
    logger.info(t("translator.active", translator=translator_label(name), language=target.label))
    if name == "deepl":
        return DeepLBackend(api_key, target)
    if name == "claude":
        return ClaudeBackend(api_key, target, choice.claude_model)
    if name == "openai":
        return OpenAIBackend(api_key, target, choice.openai_model)
    raise TranslationError(t("translator.unknown", name=name))


def split_speaker(text: str) -> tuple[str, str]:
    """('Jason', 'We go now.')-t ad vissza, ha van beszélőnév, különben ('', text)."""
    match = _SPEAKER_PREFIX.match(text)
    if not match:
        return "", text
    speaker, line = match.group(1).strip(), match.group(2)
    # Három szónál hosszabb „név” valószínűleg mondat, nem beszélő
    if len(speaker.split()) > 3:
        return "", text
    return speaker, line


def split_speakers(text: str) -> list[str]:
    """Megszólalásonként (sortörés) és beszélőnként bontja a szöveget."""
    return [part for line in text.split("\n") for part in _SPEAKER_BOUNDARY.split(line.strip()) if part]


def _cache_key(text: str) -> str:
    return text.strip().lower()


def _texts_to_translate(parts: list[_SpeakerLine]) -> list[str]:
    """Amit fordítani kell: a kimondott szöveg, és utána a nyelvre utaló megjegyzés („In Spanish”)."""
    texts: list[str] = []
    for speaker, line in parts:
        texts.append(line)
        if _LANGUAGE_NOTE.match(speaker):
            texts.append(speaker)
    return texts


def _without_added_quotes(source: str, translated: str) -> str:
    """A fordító által az egész sor köré tett idézőjel nélkül: a DeepL az aposztróffal kezdődő sort („'Ey,
    Pooch.”) idézetnek veszi. Ha az eredeti is idézőjellel kezdődik, az idézőjel marad."""
    if source.lstrip()[:1] in _DOUBLE_QUOTES:
        return translated
    if len(translated) >= 2 and translated[0] in _DOUBLE_QUOTES and translated[-1] in _DOUBLE_QUOTES:
        return translated[1:-1].strip()
    quotes = [ch for ch in translated if ch in _DOUBLE_QUOTES]
    if quotes == [translated[:1]]:  # pár nélküli nyitó idézőjel („„Hé, Lu, mondd meg…”)
        return translated[1:].strip()
    return translated


def _join_speaker(part: _SpeakerLine, translations: Mapping[str, str]) -> str:
    """A lefordított sor: a beszélő neve marad, a nyelvi megjegyzés lefordítva kerül elé."""
    speaker, line = part
    translated = _without_added_quotes(line, translations[_cache_key(line)])
    if not speaker:
        return translated
    label = translations[_cache_key(speaker)] if _LANGUAGE_NOTE.match(speaker) else speaker
    return f"{label}: {translated}"


def _checked(answer: object) -> str:
    """A szolgáltatás válasza a szélső szóközök nélkül; az üres válasz hiba, nem fordítás."""
    if not isinstance(answer, str) or not answer.strip():
        raise TranslationError(t("translator.empty_response"))
    return answer.strip()


class CachedTranslator:
    """Ugyanazt a mondatot csak egyszer fordítja le; a beszélő nevét megtartja."""

    def __init__(self, backend: TranslationBackend, cache_size: int = CACHE_SIZE, *,
                 store: TranslationStore | None = None, namespace: str = "") -> None:
        self._backend = backend
        self._store = store  # lemezre mentett fordítások (indítások között is megmaradnak)
        self._namespace = namespace
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._cache_size = cache_size
        # Az előre indított és a megerősített fordítás két szálból éri el a gyorsítótárat
        self._lock = threading.Lock()
        self.last_network_at = 0.0  # utolsó valódi hívás (time.monotonic), a melegen tartáshoz

    def warm_up(self) -> None:
        """Megnyitja / nyitva tartja a kapcsolatot a szolgáltatással, fordítás nélkül."""
        warm = getattr(self._backend, "warm_up", None)
        self.last_network_at = time.monotonic()
        if warm is not None:
            warm()

    def translate(self, text: str) -> str:
        """Beszélőnként külön sorba teszi a fordítást; a még le nem fordított sorok egy kérésben mennek."""
        parts = [split_speaker(part) for part in split_speakers(text)]
        translations = self._translate_all(_texts_to_translate(parts))
        return "\n".join(_join_speaker(part, translations) for part in parts)

    def _translate_all(self, texts: list[str]) -> dict[str, str]:
        """Kulcs → fordítás minden sorra: ami megvan, a gyorsítótárból, a többi a szolgáltatástól."""
        known: dict[str, str] = {}
        missing: dict[str, str] = {}  # kulcs → a sor, ahogy olvastuk; az ismétlődő sor csak egyszer megy el
        for text in texts:
            key = _cache_key(text)
            if key in known or key in missing:
                continue
            cached = self._lookup(key)
            if cached is None:
                missing[key] = text
            else:
                known[key] = cached
        return {**known, **self._fetch(missing)} if missing else known

    def _lookup(self, key: str) -> str | None:
        """A kész fordítás hálózat nélkül: a memóriából, vagy a lemezre mentettek közül."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        stored = self._store.get(self._namespace, key) if self._store is not None else None
        if stored is not None:
            self._remember(key, stored)
        return stored

    def _fetch(self, missing: dict[str, str]) -> dict[str, str]:
        """A hiányzó sorok fordítása a szolgáltatással; ami elkészült, az hiba esetén is megmarad.

        Minden kész sor azonnal a memóriába kerül (a másik szál is lássa), a lemezre pedig
        a végén egyetlen fájlírással, akárhány sor jött.
        """
        fresh: dict[str, str] = {}
        try:
            for key, answer in zip(missing, self._answers(list(missing.values()))):
                fresh[key] = _checked(answer)
                self._remember(key, fresh[key])
        finally:
            if self._store is not None:
                self._store.put_many(self._namespace, fresh)
        return fresh

    def _answers(self, texts: list[str]) -> Iterator[str]:
        """A sorok fordítása sorrendben: egyetlen kérésben, ha a szolgáltatás tudja, különben soronként."""
        if not isinstance(self._backend, BatchTranslationBackend):
            for text in texts:
                yield self._ask(self._backend.translate, text)
            return
        answers = self._ask(self._backend.translate_many, texts)
        if len(answers) != len(texts):
            raise TranslationError(t("translator.empty_response"))
        yield from answers

    def _ask(self, request: Callable[[_Asked], _Answered], asked: _Asked) -> _Answered:
        """Egy valódi hívás a szolgáltatás felé; annak sokféle kivételét egységes hibává alakítja."""
        self.last_network_at = time.monotonic()
        try:
            return request(asked)
        except Exception as err:  # a külső szolgáltatások sokféle kivételt dobnak
            raise TranslationError(t("translator.failed", err=err)) from err

    def _remember(self, key: str, translated: str) -> None:
        with self._lock:
            self._cache[key] = translated
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
