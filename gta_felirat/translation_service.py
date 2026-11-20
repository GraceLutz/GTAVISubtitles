"""A fordító futás közbeni cseréje (nyelv, szolgáltatás, kulcs) és állapota a panelnek."""

import logging
import threading
import time
from concurrent.futures import Executor, Future

from .config import AppConfig
from .languages import find_language
from .translation_store import TranslationStore
from .translator import (TRANSLATOR_LABELS, CachedTranslator, TranslationError, TranslatorChoice,
                         create_backend, resolve_backend_name, resolve_key)

# Ennyi csend után egy ingyenes kéréssel nyitva tartjuk a kapcsolatot (a szerverek ~60 mp után zárnak)
KEEP_WARM_SECONDS = 20.0
TRANSLATOR_FIELDS = frozenset({"translator", "target_lang", "claude_model", "openai_model"})

logger = logging.getLogger(__name__)


def choice_from(config: AppConfig) -> TranslatorChoice:
    return TranslatorChoice(config.translator, config.target_lang, config.claude_model, config.openai_model)


class TranslationService:
    def __init__(self, choice: TranslatorChoice, translator: CachedTranslator | None = None,
                 store: TranslationStore | None = None) -> None:
        self._lock = threading.Lock()
        self._store = store
        self._choice = choice
        self._session_keys: dict[str, str] = {}
        self._translator = translator
        self._error = ""
        self._last_latency_ms: int | None = None
        # Az első olvasásnál előre elindított fordítások (szöveg → folyamatban lévő kérés)
        self._inflight: dict[str, Future[str]] = {}
        self._warming = False
        if translator is None:
            self.rebuild(choice)

    def rebuild(self, choice: TranslatorChoice) -> None:
        """Új fordító a beállítások szerint; hiba esetén a panel kiírja, mi hiányzik."""
        try:
            translator = CachedTranslator(create_backend(choice, self._session_keys),
                                          store=self._store, namespace=self._namespace(choice))
            error = ""
        except (TranslationError, ValueError) as err:
            translator, error = None, str(err)
            logger.warning("%s", err)
        with self._lock:
            self._choice, self._translator, self._error = choice, translator, error
            self._inflight = {}

    def _namespace(self, choice: TranslatorChoice) -> str:
        """Fordítónként, AI-modellenként és nyelvenként külön mentett fordítások."""
        backend = resolve_backend_name(choice.name, self._session_keys)
        model = {"claude": choice.claude_model, "openai": choice.openai_model}.get(backend, "")
        return ":".join(part for part in (backend, model, choice.target_lang) if part)

    def set_session_key(self, name: str, key: str) -> None:
        """A kulcs csak a memóriában marad, fájlba nem írjuk."""
        self._session_keys[name] = key
        self.rebuild(self._choice)

    def has_key(self, name: str) -> bool:
        return bool(resolve_key(name, self._session_keys))

    def prefetch(self, text: str, pool: Executor) -> None:
        """Előre elindítja a fordítást, hogy mire a felirat megerősítést kap, kész legyen."""
        with self._lock:
            if text in self._inflight or self._translator is None:
                return
            future = pool.submit(self._translate_now, text)
            self._inflight[text] = future
        future.add_done_callback(lambda _done: self._forget(text, future))

    def _forget(self, text: str, future: Future[str]) -> None:
        with self._lock:
            if self._inflight.get(text) is future:
                del self._inflight[text]

    def translate(self, text: str) -> str:
        with self._lock:
            pending = self._inflight.get(text)
        if pending is not None:
            try:
                return pending.result()
            except TranslationError:
                logger.info("Az előre indított fordítás nem sikerült, újrapróbálom")
        return self._translate_now(text)

    def _translate_now(self, text: str) -> str:
        with self._lock:
            translator, error = self._translator, self._error
        if translator is None:
            raise TranslationError(error or "Nincs beállított fordító")
        started = time.monotonic()
        result = translator.translate(text)
        self._last_latency_ms = int((time.monotonic() - started) * 1000)
        return result

    def keep_warm(self) -> None:
        """Minden képkockánál hívható: ha rég volt hívás, háttérszálon nyitva tartja a kapcsolatot."""
        with self._lock:
            translator = self._translator
            if (translator is None or self._warming
                    or time.monotonic() - translator.last_network_at < KEEP_WARM_SECONDS):
                return
            self._warming = True
        threading.Thread(target=self._warm, args=(translator,), daemon=True, name="keep-warm").start()

    def _warm(self, translator: CachedTranslator) -> None:
        try:
            translator.warm_up()
        except Exception as err:  # a melegen tartás hibája nem akadályozhatja a fordítást
            logger.debug("Kapcsolat melegen tartása nem sikerült: %s", err)
        finally:
            with self._lock:
                self._warming = False

    def status(self) -> str:
        with self._lock:
            choice, error = self._choice, self._error
        backend = resolve_backend_name(choice.name, self._session_keys)
        text = f"Fordító: {TRANSLATOR_LABELS.get(backend, backend)} → {find_language(choice.target_lang).label}"
        if self._last_latency_ms is not None:
            text += f" · utolsó fordítás {self._last_latency_ms} ms"
        return f"{text}\n⚠ {error}" if error else text
