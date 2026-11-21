"""Fordítószolgáltatások: DeepL, Claude, OpenAI GPT és Google.

A DeepL, a Claude és az OpenAI hívása sima HTTPS-kérés (requests), így nem
kell külön SDK. A kulcsokat környezeti változóból vagy a panelből kapjuk,
fájlba sosem írjuk. Minden valódi kérés mért ideje info szinten a naplóba kerül.
"""

import logging
import math
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

import requests

from .i18n import t
from .languages import Language

REQUEST_TIMEOUT_SECONDS = 10
MIN_REQUEST_TIMEOUT_SECONDS = 0.1  # a requests nem fogad el nulla vagy negatív időkorlátot
# Egy fordítás összesen (újrapróbálással együtt) legfeljebb ennyi ideig tarthat
TRANSLATION_BUDGET_SECONDS = 10.0
MIN_RETRY_SECONDS = 2.0  # ennél kevesebb maradék időre már nem próbáljuk újra
MAX_ATTEMPTS = 2  # az első kérés és legfeljebb egy újrapróbálás
# Túlterhelés (429/5xx) után ennyit várunk: élő feliratnál egy egész másodperc már túl sok.
# Megszakadt kapcsolatnál nem várunk, mert nincs mire: azonnal új kapcsolaton próbáljuk újra.
RETRY_DELAY_SECONDS = 0.3
MAX_RETRY_DELAY_SECONDS = 1.0  # a szerver Retry-After kérését legfeljebb eddig tartjuk be
RETRYABLE_STATUS = {429, 500, 502, 503, 504, 529}
BAD_KEY_STATUS = {401, 403}  # hibás, lejárt vagy ehhez a címhez nem jogosult kulcs
FIRST_ERROR_STATUS = 400  # ettől felfelé a HTTP-állapot hibát jelent
MS_PER_SECOND = 1000  # a naplóba ezredmásodpercben írjuk a mért időt
CONTEXT_LINES = 3
MAX_OUTPUT_TOKENS = 300
DEEPL_MAX_TEXTS = 50  # a DeepL egy kérésben legfeljebb ennyi szöveget fogad el

DEEPL_FREE_URL = "https://api-free.deepl.com/v2/translate"
DEEPL_PRO_URL = "https://api.deepl.com/v2/translate"
CLAUDE_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_API_VERSION = "2023-06-01"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
# Ingyenes, karaktert/tokent nem fogyasztó címek: csak a kapcsolatot tartják nyitva
DEEPL_FREE_USAGE_URL = "https://api-free.deepl.com/v2/usage"
DEEPL_PRO_USAGE_URL = "https://api.deepl.com/v2/usage"
CLAUDE_MODELS_URL = "https://api.anthropic.com/v1/models?limit=1"
OPENAI_MODELS_URL = "https://api.openai.com/v1/models"
# A szolgáltatások neve a naplóban és a hibaüzenetekben
DEEPL_SERVICE = "DeepL"
CLAUDE_SERVICE = "Claude"
OPENAI_SERVICE = "GPT"
GOOGLE_SERVICE = "Google"

logger = logging.getLogger(__name__)

# Egy közös, nyitva tartott kapcsolat: nem kell minden mondatnál új HTTPS-kézfogás
_session = requests.Session()


class BackendError(RuntimeError):
    """A szolgáltatás hibát adott; az üzenet a felhasználónak szól, kulcsot nem tartalmaz."""


@dataclass(frozen=True)
class _Endpoint:
    """Egy szolgáltatás címe a hitelesítő fejléccel (a fejléc a kulcs miatt a repr-ben sem látszik)."""

    service: str
    url: str
    headers: dict[str, str] = field(repr=False)


@dataclass(frozen=True)
class _RoundTrip:
    """Egy valódi kérés mérése: a naplóba az idő és a méret kerül, szöveg és kulcs soha."""

    service: str
    text_count: int
    character_count: int
    started: float

    @classmethod
    def start(cls, service: str, texts: Sequence[str]) -> "_RoundTrip":
        return cls(service, len(texts), sum(len(text) for text in texts), time.monotonic())

    def log(self, outcome: str) -> None:
        """Info szinten, hogy élesben is látsszon, mennyi ideig tart egy kérés a szolgáltatásnál."""
        elapsed_ms = round((time.monotonic() - self.started) * MS_PER_SECOND)
        logger.info("%s round trip: %d ms, %s, %d texts, %d chars",
                    self.service, elapsed_ms, outcome, self.text_count, self.character_count)


def _post_json(endpoint: _Endpoint, payload: dict, texts: Sequence[str]) -> dict:
    """POST legfeljebb egy újrapróbálással, összesen max. 10 mp-ig.

    Megszakadt kapcsolatnál azonnal újrapróbálunk, túlterhelésnél (429/5xx) rövid szünet után.
    A texts a kérésben küldött szöveg: a naplóba csak a darabszáma és a hossza kerül.
    """
    deadline = time.monotonic() + TRANSLATION_BUDGET_SECONDS
    for attempt in range(1, MAX_ATTEMPTS + 1):
        is_last_attempt = attempt == MAX_ATTEMPTS
        trip = _RoundTrip.start(endpoint.service, texts)
        timeout = min(REQUEST_TIMEOUT_SECONDS, max(deadline - trip.started, MIN_REQUEST_TIMEOUT_SECONDS))
        try:
            response = _session.post(endpoint.url, headers=endpoint.headers, json=payload, timeout=timeout)
        except requests.RequestException as err:
            trip.log(type(err).__name__)
            if is_last_attempt or deadline - time.monotonic() < MIN_RETRY_SECONDS:
                raise BackendError(t("backend.network_error", error_type=type(err).__name__)) from err
            continue  # megszakadt kapcsolat: nincs mire várni, azonnal új kapcsolaton próbáljuk
        trip.log(f"HTTP {response.status_code}")
        if response.status_code not in RETRYABLE_STATUS:
            return _checked_json(response)
        pause = _retry_pause(response)
        if is_last_attempt or deadline - time.monotonic() - pause < MIN_RETRY_SECONDS:
            break
        time.sleep(pause)
    raise BackendError(t("backend.overloaded"))


def _checked_json(response: requests.Response) -> dict:
    """A sikeres válasz tartalma; hibás kulcsnál és más HTTP-hibánál a felhasználónak szóló hiba."""
    if response.status_code in BAD_KEY_STATUS:
        raise BackendError(t("backend.bad_key"))
    if response.status_code >= FIRST_ERROR_STATUS:
        raise BackendError(t("backend.http_error", status=response.status_code))
    return response.json()


def _retry_pause(response: requests.Response) -> float:
    """Szünet az újrapróbálás előtt: amennyit a szerver kér (Retry-After), de a felirat miatt röviden."""
    try:
        asked = float(response.headers.get("Retry-After", ""))
    except ValueError:  # nincs ilyen fejléc, vagy dátumot tartalmaz
        return RETRY_DELAY_SECONDS
    if not math.isfinite(asked):
        return RETRY_DELAY_SECONDS
    return min(max(asked, 0.0), MAX_RETRY_DELAY_SECONDS)


def _warm_get(url: str, headers: dict) -> None:
    """Egy olcsó GET, hogy a HTTPS-kapcsolat nyitva maradjon (új kézfogás ~150–250 ms)."""
    try:
        response = _session.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.RequestException as err:
        raise BackendError(t("backend.network_error", error_type=type(err).__name__)) from err
    if response.status_code in BAD_KEY_STATUS:
        raise BackendError(t("backend.bad_key"))


class DeepLBackend:
    def __init__(self, api_key: str, target: Language) -> None:
        if target.deepl_code is None:
            raise BackendError(t("backend.deepl_unsupported", language=target.label))
        self._target = target.deepl_code
        # Az ingyenes DeepL-kulcs :fx-re végződik, és másik címen fut
        is_free = api_key.endswith(":fx")
        self._endpoint = _Endpoint(DEEPL_SERVICE, DEEPL_FREE_URL if is_free else DEEPL_PRO_URL,
                                   {"Authorization": f"DeepL-Auth-Key {api_key}"})
        self._usage_url = DEEPL_FREE_USAGE_URL if is_free else DEEPL_PRO_USAGE_URL

    def warm_up(self) -> None:
        _warm_get(self._usage_url, self._endpoint.headers)

    def translate(self, text: str) -> str:
        return self.translate_many([text])[0]

    def translate_many(self, texts: list[str]) -> list[str]:
        """Több sor egyetlen kérésben: egy hálózati kör, a válaszok a sorok sorrendjében.

        A DeepL a sorokat egymástól függetlenül fordítja, ezért az eredmény ugyanaz, mint soronként kérve.
        """
        translated: list[str] = []
        for start in range(0, len(texts), DEEPL_MAX_TEXTS):
            translated.extend(self._translate_chunk(texts[start:start + DEEPL_MAX_TEXTS]))
        return translated

    def _translate_chunk(self, texts: list[str]) -> list[str]:
        payload = {"text": texts, "source_lang": "EN", "target_lang": self._target}
        data = _post_json(self._endpoint, payload, texts)
        try:
            answers = [item["text"] for item in data["translations"]]
        except (KeyError, TypeError) as err:
            raise BackendError(t("backend.unexpected_response", service=DEEPL_SERVICE)) from err
        if len(answers) != len(texts):
            raise BackendError(t("backend.unexpected_response", service=DEEPL_SERVICE))
        return answers


def build_prompt(target: Language, context: list[str], text: str) -> tuple[str, str]:
    system = (
        f"You translate English video game subtitles into {target.english_name}. "
        "Write natural, spoken-style dialogue that fits a crime action game: keep slang, "
        "tone and profanity level, keep names untranslated. "
        "Reply with the translation only, no quotes, no notes."
    )
    if not context:
        return system, text
    earlier = "\n".join(context)
    user = f"Earlier lines (context only, do not translate):\n{earlier}\n\nTranslate this line:\n{text}"
    return system, user


class _LlmBackend:
    """Közös rész: az előző pár sort kontextusként küldjük (nem, megszólítás, hangnem)."""

    def __init__(self, api_key: str, target: Language, model: str) -> None:
        self._api_key = api_key
        self._target = target
        self._model = model
        self._context: deque[str] = deque(maxlen=CONTEXT_LINES)

    def translate(self, text: str) -> str:
        system, user = build_prompt(self._target, list(self._context), text)
        result = self._complete(system, user).strip()
        self._context.append(text)
        return result

    def _complete(self, system: str, user: str) -> str:
        raise NotImplementedError


class ClaudeBackend(_LlmBackend):
    def warm_up(self) -> None:
        _warm_get(CLAUDE_MODELS_URL, {"x-api-key": self._api_key, "anthropic-version": CLAUDE_API_VERSION})

    def _complete(self, system: str, user: str) -> str:
        data = _post_json(
            _Endpoint(CLAUDE_SERVICE, CLAUDE_URL,
                      {"x-api-key": self._api_key, "anthropic-version": CLAUDE_API_VERSION}),
            {"model": self._model, "max_tokens": MAX_OUTPUT_TOKENS, "system": system,
             "messages": [{"role": "user", "content": user}]},
            [user],
        )
        try:
            return "".join(block["text"] for block in data["content"] if block.get("type") == "text")
        except (KeyError, TypeError) as err:
            raise BackendError(t("backend.unexpected_response", service=CLAUDE_SERVICE)) from err


class OpenAIBackend(_LlmBackend):
    def warm_up(self) -> None:
        _warm_get(OPENAI_MODELS_URL, {"Authorization": f"Bearer {self._api_key}"})

    def _complete(self, system: str, user: str) -> str:
        data = _post_json(
            _Endpoint(OPENAI_SERVICE, OPENAI_URL, {"Authorization": f"Bearer {self._api_key}"}),
            {"model": self._model, "messages": [{"role": "system", "content": system},
                                                {"role": "user", "content": user}]},
            [user],
        )
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as err:
            raise BackendError(t("backend.unexpected_response", service=OPENAI_SERVICE)) from err


class _RequestsWithTimeout:
    """A deep-translator időkorlát nélkül hívja a requests.get-et: ezen át mindig kap egyet.

    Időkorlát nélkül egy beragadt kérés örökre lefoglalna egy fordítószálat.
    """

    @staticmethod
    def get(url: str, **kwargs) -> requests.Response:
        return requests.get(url, **{"timeout": REQUEST_TIMEOUT_SECONDS, **kwargs})


class GoogleBackend:
    def __init__(self, target: Language) -> None:
        from deep_translator import GoogleTranslator
        from deep_translator import google as google_module
        from deep_translator.exceptions import BaseError

        # A könyvtár a saját modulszintű requests-hivatkozásán át kér: azt cseréljük az időkorlátosra
        google_module.requests = _RequestsWithTimeout
        self._client = GoogleTranslator(source="en", target=target.code)
        self._library_error = BaseError
        # A kliens a kérés szövegét a saját állapotában tartja: egyszerre több szálról hívva összekeverné
        # a szövegeket (és a rossz fordítás a mentett fordítások közé kerülne)
        self._lock = threading.Lock()

    def translate(self, text: str) -> str:
        trip = _RoundTrip.start(GOOGLE_SERVICE, [text])
        try:
            with self._lock:
                result = self._client.translate(text)
        except requests.RequestException as err:
            trip.log(type(err).__name__)
            # A requests hibaszövegében ott a teljes cím a felirat szövegével: csak a hiba fajtáját adjuk tovább
            raise BackendError(t("backend.network_error", error_type=type(err).__name__)) from err
        except self._library_error as err:
            trip.log(type(err).__name__)
            # A könyvtár hibaszövege a felirat szövegét is tartalmazhatja: azt nem adjuk tovább
            raise BackendError(t("backend.unexpected_response", service=GOOGLE_SERVICE)) from err
        trip.log("OK")
        return result
