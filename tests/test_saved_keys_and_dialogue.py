import sys
import time
import uuid

import pytest

from gta_felirat.config import AppConfig
from gta_felirat.key_vault import KeyVaultError, WindowsCredentialVault
from gta_felirat.ocr import Box, OcrResult, clean_ocr_text, is_spoken_line
from gta_felirat.subtitle_pipeline import SubtitlePipeline
from gta_felirat.translation_service import TranslationService
from gta_felirat.translator import CachedTranslator, TranslatorChoice

DEEPL = TranslatorChoice("deepl", "hu", "claude-haiku-5-5", "gpt-4.1-mini")


class MemoryVault:
    def __init__(self, keys=None, fail_on_save=False):
        self.keys = dict(keys or {})
        self.fail_on_save = fail_on_save

    def load(self, name):
        return self.keys.get(name)

    def save(self, name, key):
        if self.fail_on_save:
            raise KeyVaultError("A kulcsot nem sikerült elmenteni (Windows-hiba 5)")
        self.keys[name] = key

    def delete(self, name):
        self.keys.pop(name, None)


@pytest.fixture(autouse=True)
def no_env_keys(monkeypatch):
    for var in ("DEEPL_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)


# --- mentett API-kulcsok ------------------------------------------------------

def test_saved_key_is_used_at_startup_without_typing_it_again():
    service = TranslationService(DEEPL, vault=MemoryVault({"deepl": "abc:fx"}))

    assert service.has_key("deepl")
    assert service.is_key_saved("deepl")
    assert "hiányzik" not in service.status()


def test_key_typed_in_the_panel_is_saved_for_next_start():
    vault = MemoryVault()
    TranslationService(DEEPL, vault=vault).set_session_key("deepl", "abc:fx")

    assert TranslationService(DEEPL, vault=vault).has_key("deepl")


def test_forgotten_key_is_gone_now_and_after_restart():
    vault = MemoryVault({"deepl": "abc:fx"})
    service = TranslationService(DEEPL, vault=vault)

    service.forget_key("deepl")

    assert not service.has_key("deepl")
    assert not TranslationService(DEEPL, vault=vault).has_key("deepl")


def test_key_still_works_for_this_run_when_saving_fails():
    service = TranslationService(DEEPL, vault=MemoryVault(fail_on_save=True))

    service.set_session_key("deepl", "abc:fx")

    assert service.has_key("deepl")
    assert not service.is_key_saved("deepl")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Hitelesítőadat-kezelő")
def test_windows_credential_vault_round_trip():
    vault = WindowsCredentialVault(prefix=f"gta_felirat_teszt_{uuid.uuid4().hex}/")
    try:
        assert vault.load("deepl") is None
        vault.save("deepl", "abc:fx-ékezet")
        assert vault.load("deepl") == "abc:fx-ékezet"
    finally:
        vault.delete("deepl")
    assert vault.load("deepl") is None


# --- csak a kimondott szöveg --------------------------------------------------

@pytest.mark.parametrize("text", [
    "Lucia: Jeez, what are you on, bro?",
    "Vagrant: Take a breath now. You ain't made for this. Lucia: Stop talking.",
    "Locals: Enough! I'm not your doormat!",
])
def test_dialogue_lines_count_as_spoken(text):
    assert is_spoken_line(text)


@pytest.mark.parametrize("text", [
    "Go to the Vice City Port.",
    "English About Us Privacy policy Publisher Terms & Conditions",
    "Press the button to open the map.",
    "Jason 00:04",  # a telefonhívás időzítője, nem beszélő
    "Jason Call Ended D",
])
def test_other_screen_text_is_not_spoken(text):
    assert not is_spoken_line(text)


@pytest.mark.parametrize("raw, cleaned", [
    ("Lucia: Hold tight, alright? Jason 00:05 D", "Lucia: Hold tight, alright?"),
    ("Jason: Hit me back if you need something. Jason 00:01 00",
     "Jason: Hit me back if you need something."),
    ("08 Vagrant: Frog-looking bitch.", "Vagrant: Frog-looking bitch."),
    ("Lucia: You got this, bro. 8", "Lucia: You got this, bro."),
    ("Lucia: People don't talk to each other anymore. C8", "Lucia: People don't talk to each other anymore."),
    ("Locals: Hell yeah!! X", "Locals: Hell yeah!!"),
    ("SEL Locals: I know you? I get around a lot.. T", "Locals: I know you? I get around a lot.."),
    ("Jason: Me and Lucia. We go.", "Jason: Me and Lucia. We go."),
])
def test_phone_timer_and_stray_numbers_are_removed(raw, cleaned):
    assert clean_ocr_text(raw) == cleaned


class FixedReader:
    def __init__(self, text):
        self.text = text

    def read_band(self, band, area):
        return OcrResult(text=self.text, box=Box(area.left, area.top, area.left + 100, area.top + 30))


class EchoBackend:
    def __init__(self):
        self.calls = []

    def translate(self, text):
        self.calls.append(text)
        return f"HU({text})"


def run_two_reads(config, text):
    backend = EchoBackend()
    service = TranslationService(DEEPL, CachedTranslator(backend))
    pipeline = SubtitlePipeline(config, FixedReader(text), service)
    area = pipeline.band_area(1920, 1080)
    try:
        for step in range(2):
            now = step * 1.0
            pipeline.submit_band(lambda: None, area, (1920, 1080), now)
            deadline = time.monotonic() + 2
            while pipeline._ocr_future is not None and time.monotonic() < deadline:
                pipeline.poll(now)
                time.sleep(0.01)
        time.sleep(0.1)
        pipeline.poll(2.0)
    finally:
        pipeline.shutdown()
    return backend.calls, pipeline.shown


def test_only_spoken_lines_skips_mission_text():
    calls, shown = run_two_reads(AppConfig(only_spoken_lines=True), "Go to the Vice City Port.")

    assert calls == []
    assert shown is None


def test_spoken_line_is_translated_when_only_spoken_lines_is_on():
    calls, shown = run_two_reads(AppConfig(only_spoken_lines=True), "Lucia: Stop talking.")

    assert calls == ["Stop talking."]  # a beszélő nevét nem küldjük fordításra
    assert shown is not None


def test_all_text_is_translated_when_the_option_is_off():
    calls, _shown = run_two_reads(AppConfig(only_spoken_lines=False), "Go to the Vice City Port.")

    assert calls == ["Go to the Vice City Port."]
