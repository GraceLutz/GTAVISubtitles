"""Fordítószolgáltatások: DeepL, Claude, OpenAI GPT és Google.

A DeepL, a Claude és az OpenAI hívása sima HTTPS-kérés (requests), így nem
kell külön SDK. A kulcsokat környezeti változóból vagy a panelből kapjuk,
fájlba sosem írjuk.
"""

import time
from collections import deque

import requests

from .languages import Language

REQUEST_TIMEOUT_SECONDS = 10
# Egy fordítás összesen (újrapróbálással együtt) legfeljebb ennyi ideig tarthat
TRANSLATION_BUDGET_SECONDS = 10.0
MIN_RETRY_SECONDS = 2.0  # ennél kevesebb maradék időre már nem próbáljuk újra
RETRY_DELAY_SECONDS = 1.0
RETRYABLE_STATUS = {429, 500, 502, 503, 529}
CONTEXT_LINES = 3
MAX_OUTPUT_TOKENS = 300

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

# Egy közös, nyitva tartott kapcsolat: nem kell minden mondatnál új HTTPS-kézfogás
_session = requests.Session()


class BackendError(RuntimeError):
    """A szolgáltatás hibát adott; az üzenet a felhasználónak szól, kulcsot nem tartalmaz."""


def _post_json(url: str, headers: dict, payload: dict) -> dict:
    """POST egy újrapróbálással túlterhelés vagy hálózati hiba esetén, összesen max. 10 mp-ig."""
    deadline = time.monotonic() + TRANSLATION_BUDGET_SECONDS
    for attempt in range(2):
        remaining = deadline - time.monotonic()
        can_retry = attempt == 0 and remaining - RETRY_DELAY_SECONDS >= MIN_RETRY_SECONDS
        try:
            response = _session.post(url, headers=headers, json=payload,
                                     timeout=min(REQUEST_TIMEOUT_SECONDS, max(remaining, 0.1)))
        except requests.RequestException as err:
            if can_retry and deadline - time.monotonic() - RETRY_DELAY_SECONDS >= MIN_RETRY_SECONDS:
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            raise BackendError(f"Hálózati hiba: {type(err).__name__}") from err
        if response.status_code in RETRYABLE_STATUS and can_retry:
            time.sleep(RETRY_DELAY_SECONDS)
            continue
        if response.status_code in (401, 403):
            raise BackendError("Hibás vagy lejárt API-kulcs")
        if response.status_code >= 400:
            raise BackendError(f"A szolgáltatás hibát adott (HTTP {response.status_code})")
        return response.json()
    raise BackendError("A szolgáltatás túlterhelt, próbáld később")


def _warm_get(url: str, headers: dict) -> None:
    """Egy olcsó GET, hogy a HTTPS-kapcsolat nyitva maradjon (új kézfogás ~150–250 ms)."""
    try:
        response = _session.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.RequestException as err:
        raise BackendError(f"Hálózati hiba: {type(err).__name__}") from err
    if response.status_code in (401, 403):
        raise BackendError("Hibás vagy lejárt API-kulcs")


class DeepLBackend:
    def __init__(self, api_key: str, target: Language) -> None:
        if target.deepl_code is None:
            raise BackendError(f"A DeepL nem fordít erre a nyelvre: {target.label}")
        self._api_key = api_key
        self._target = target.deepl_code
        # Az ingyenes DeepL-kulcs :fx-re végződik, és másik címen fut
        is_free = api_key.endswith(":fx")
        self._url = DEEPL_FREE_URL if is_free else DEEPL_PRO_URL
        self._usage_url = DEEPL_FREE_USAGE_URL if is_free else DEEPL_PRO_USAGE_URL

    def warm_up(self) -> None:
        _warm_get(self._usage_url, {"Authorization": f"DeepL-Auth-Key {self._api_key}"})

    def translate(self, text: str) -> str:
        data = _post_json(self._url, {"Authorization": f"DeepL-Auth-Key {self._api_key}"},
                          {"text": [text], "source_lang": "EN", "target_lang": self._target})
        try:
            return data["translations"][0]["text"]
        except (KeyError, IndexError, TypeError) as err:
            raise BackendError("A DeepL váratlan választ adott") from err


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
            CLAUDE_URL,
            {"x-api-key": self._api_key, "anthropic-version": CLAUDE_API_VERSION},
            {"model": self._model, "max_tokens": MAX_OUTPUT_TOKENS, "system": system,
             "messages": [{"role": "user", "content": user}]},
        )
        try:
            return "".join(block["text"] for block in data["content"] if block.get("type") == "text")
        except (KeyError, TypeError) as err:
            raise BackendError("A Claude váratlan választ adott") from err


class OpenAIBackend(_LlmBackend):
    def warm_up(self) -> None:
        _warm_get(OPENAI_MODELS_URL, {"Authorization": f"Bearer {self._api_key}"})

    def _complete(self, system: str, user: str) -> str:
        data = _post_json(
            OPENAI_URL,
            {"Authorization": f"Bearer {self._api_key}"},
            {"model": self._model, "messages": [{"role": "system", "content": system},
                                                {"role": "user", "content": user}]},
        )
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as err:
            raise BackendError("A GPT váratlan választ adott") from err


class GoogleBackend:
    def __init__(self, target: Language) -> None:
        from deep_translator import GoogleTranslator

        self._client = GoogleTranslator(source="en", target=target.code)

    def translate(self, text: str) -> str:
        return self._client.translate(text)
