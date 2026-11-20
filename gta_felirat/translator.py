"""Fordító kiválasztása és gyorsítótárazott fordítás (a beszélő nevét megtartva)."""

import logging
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping, Protocol

from .backends import ClaudeBackend, DeepLBackend, GoogleBackend, OpenAIBackend
from .languages import find_language
from .translation_store import TranslationStore

CACHE_SIZE = 2000
MAX_SPEAKER_NAME_LENGTH = 24
# „Jason: Gyere már!” — a beszélő nevét nem fordítjuk le
_SPEAKER_PREFIX = re.compile(r"^([A-Z][\w.'\- ]{0,%d}):\s+(.+)$" % (MAX_SPEAKER_NAME_LENGTH - 1))

KEY_ENV_VARS = {"deepl": "DEEPL_API_KEY", "claude": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
# „auto” esetén ebben a sorrendben választunk abból, amihez van kulcs
AUTO_PRIORITY = ("deepl", "claude", "openai")
TRANSLATOR_LABELS = {
    "auto": "Automatikus",
    "deepl": "DeepL",
    "claude": "Claude (Anthropic)",
    "openai": "ChatGPT (OpenAI)",
    "google": "Google (ingyenes)",
}

logger = logging.getLogger(__name__)


class TranslationError(RuntimeError):
    """A fordítószolgáltatás hibát adott vagy nem érhető el."""


class TranslationBackend(Protocol):
    def translate(self, text: str) -> str: ...


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
        logger.info("Fordító: Google → %s", target.label)
        return GoogleBackend(target)
    api_key = resolve_key(name, keys)
    if not api_key:
        raise TranslationError(
            f"{TRANSLATOR_LABELS[name]}: hiányzik az API-kulcs ({KEY_ENV_VARS[name]}), "
            "add meg a beállítópanelen vagy környezeti változóban")
    logger.info("Fordító: %s → %s", TRANSLATOR_LABELS[name], target.label)
    if name == "deepl":
        return DeepLBackend(api_key, target)
    if name == "claude":
        return ClaudeBackend(api_key, target, choice.claude_model)
    if name == "openai":
        return OpenAIBackend(api_key, target, choice.openai_model)
    raise TranslationError(f"Ismeretlen fordító: {name}")


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
        speaker, line = split_speaker(text)
        translated = self._translate_line(line)
        return f"{speaker}: {translated}" if speaker else translated

    def _translate_line(self, line: str) -> str:
        key = line.strip().lower()
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        stored = self._store.get(self._namespace, key) if self._store else None
        if stored is not None:
            self._remember(key, stored)
            return stored
        self.last_network_at = time.monotonic()
        try:
            result = self._backend.translate(line)
        except Exception as err:  # a külső szolgáltatások sokféle kivételt dobnak
            raise TranslationError(f"Fordítás sikertelen: {err}") from err
        if not result or not result.strip():
            raise TranslationError("A fordító üres választ adott")
        translated = result.strip()
        self._remember(key, translated)
        if self._store:
            self._store.put(self._namespace, key, translated)
        return translated

    def _remember(self, key: str, translated: str) -> None:
        with self._lock:
            self._cache[key] = translated
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
