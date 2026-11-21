import codecs
import threading
from concurrent.futures import wait
from dataclasses import replace

import numpy as np
import pytest

from gta_felirat import subtitle_pipeline
from gta_felirat.config import AppConfig, Region
from gta_felirat.i18n import t
from gta_felirat.layer_plan import ShownLine
from gta_felirat.line_placement import PENDING_HIDE_SECONDS
from gta_felirat.ocr import EMPTY_RESULT, Box, OcrLine, OcrResult, region_to_box
from gta_felirat.row_pipeline import (EAGER_HIDE_SECONDS, RECHECK_SECONDS, STILL_SECONDS, UNREAD_HIDE_SECONDS,
                                      RowPipeline)
from gta_felirat.row_tracker import BandSnapshot, RowObs, RowText
from gta_felirat.translation_service import TranslationService
from gta_felirat.translator import CachedTranslator, TranslatorChoice

FRAME = (1920, 1080)
SMALL_FRAME = (1280, 720)
AREA = region_to_box(Region(), *FRAME)
BAND = np.zeros((AREA.height, AREA.width, 3), np.uint8)
UPPER = Box(700, 913, 1220, 950)
LOWER = Box(780, 957, 1140, 994)
OBJECTIVE = Box(880, 1009, 1040, 1057)  # „Find Dunk”: a párbeszédsorok alatt áll
TROUBLE = "We ain't lookin' for trouble, okay?"
HOW = "How about you, how you doin'?"
MISSION = "Go to the Vice City Port."
HOLD = AppConfig().hold_seconds
TICK = 0.015  # az app rendes időzítőjének üteme
WAIT_SECONDS = 5  # felső határ a fordító szálra várva; a tesztek ezredmásodpercek alatt végeznek


def hu(text):
    """Hamis fordítás, amely nem hasonlít az angolra."""
    return "HU<" + codecs.encode(text, "rot13") + ">"


class GatedBackend:
    """Hamis fordító: addig nem válaszol, amíg a teszt el nem engedi, így a „közben” állapot is vizsgálható."""

    def __init__(self):
        self.calls = []
        self.called = threading.Event()
        self.release = threading.Event()

    def translate(self, text):
        self.calls.append(text)
        self.called.set()
        self.release.wait(timeout=WAIT_SECONDS)
        return hu(text)


class FailingBackend:
    def __init__(self):
        self.release = threading.Event()

    def translate(self, text):
        raise ConnectionError("offline")


class LineReader:
    """A régi (teljes olvasású) út olvasója: mindig ugyanazt a sort adja, a megadott helyen."""

    def __init__(self, box, with_lines=True):
        self.box = box
        self.with_lines = with_lines

    def read_band(self, band, area):
        return OcrResult(TROUBLE, self.box, (OcrLine(TROUBLE, self.box),) if self.with_lines else ())


class EmptyReader:
    """A régi út olvasója, amikor nincs felirat a sávban."""

    def read_band(self, band, area):
        return EMPTY_RESULT


def row(row_id, box, first_seen=0.0, stable=2, **changes):
    return replace(RowObs(row_id, box, 0.1, first_seen, stable, sure=True), **changes)


class Scene:
    """A pipeline a tesztben: pillanatképeket és sorszövegeket kap, szimulált idővel."""

    def __init__(self, backend, reader, config):
        self.backend = backend
        self.woken = threading.Event()
        service = TranslationService(TranslatorChoice("deepl", "hu", "", ""), CachedTranslator(backend))
        self.pipeline = RowPipeline(AppConfig(**{"only_spoken_lines": False, **config}), reader, service,
                                    wake=self.woken.set)

    @property
    def shown(self):
        return self.pipeline.shown

    def show(self, now, *rows, frame_size=FRAME):
        return self.pipeline.receive(BandSnapshot(BAND, AREA, frame_size, now, tuple(rows)), (), now)

    def read(self, now, *texts, score=0.99):
        return self.pipeline.receive(None, [RowText(row_id, text, score) for row_id, text in texts], now)

    def finish_translation(self):
        """Elengedi a hamis fordítót, és megvárja (valós időben) a fordító szálat."""
        future = self.pipeline._translate_future
        self.backend.release.set()
        if future is not None:  # None: a hibával azonnal végző fordítást már az előző lépés begyűjtötte
            wait([future], timeout=WAIT_SECONDS)
        self.backend.release.clear()

    def translated(self, now):
        """Megjön a fordítás: a szimulált időben begyűjtjük."""
        self.finish_translation()
        return self.pipeline.pump(now)

    def tick_until_change(self, start, end):
        """A rendes időzítő ütemében hívja a pump-ot; az első változás idejét adja vissza (None: nem volt)."""
        now = start
        while now < end:
            if self.pipeline.pump(now):
                return now
            now += TICK
        return None

    def speak(self, text=TROUBLE, box=LOWER, row_id=1):
        """Megjelenik egy sor, elolvassuk, és megjön a fordítása (1,0 és 1,3 mp között)."""
        self.show(1.0, row(row_id, box, first_seen=1.0))
        self.read(1.04, (row_id, text))
        self.translated(1.3)

    def read_fully(self, now):
        """Egy olvasás a régi úton (submit_band / poll)."""
        self.pipeline.submit_band(lambda: BAND, AREA, FRAME, now)
        wait([self.pipeline._ocr_future], timeout=WAIT_SECONDS)
        return self.pipeline.poll(now)

    def close(self):
        self.backend.release.set()
        self.pipeline.shutdown()


@pytest.fixture
def make_scene():
    scenes = []

    def make(backend=None, reader=None, **config):
        scenes.append(Scene(backend or GatedBackend(), reader, config))
        return scenes[-1]

    yield make
    for scene in scenes:
        scene.close()


# --- takarás még az olvasás előtt ----------------------------------------------------

def test_new_row_is_hidden_at_first_sight(make_scene):
    scene = make_scene()

    changed = scene.show(1.0, row(1, LOWER, first_seen=1.0, stable=1))

    assert changed
    assert scene.shown.hidden == (LOWER,) and scene.shown.text == ""
    assert scene.backend.calls == []


def test_hiding_before_the_read_is_bounded_in_time(make_scene):
    scene = make_scene()
    scene.show(1.0, row(1, LOWER, first_seen=1.0, stable=1))

    shown_again_at = scene.tick_until_change(1.01, 3.0)

    assert scene.shown is None  # a felismerő nem válaszolt: az angol sor újra látszik
    assert UNREAD_HIDE_SECONDS <= shown_again_at - 1.0 <= UNREAD_HIDE_SECONDS + TICK


def test_row_stays_hidden_until_a_slow_read_arrives_and_its_translation_comes(make_scene):
    scene = make_scene()
    scene.show(1.0, row(1, LOWER, first_seen=1.0, stable=1))

    assert scene.tick_until_change(1.01, 1.8) is None  # a felismerő lassú: az angol sor addig sem villan ki
    scene.read(1.8, (1, TROUBLE))
    while_translating = scene.shown
    scene.translated(2.0)

    assert while_translating.hidden == (LOWER,)
    assert scene.shown.lines == (ShownLine(hu(TROUBLE), LOWER),)


@pytest.mark.parametrize("doubt", [
    pytest.param({"sure": False}, id="the detector is not sure about it"),
    pytest.param({"whiteness": 0.01}, id="it has no subtitle coloured letters"),
    pytest.param({"missing": 1}, id="it dropped out of the picture"),
])
def test_row_that_may_not_be_a_dialogue_line_is_not_hidden_before_it_is_read(make_scene, doubt):
    scene = make_scene()

    scene.show(1.0, row(1, LOWER, first_seen=1.0, stable=1, **doubt))

    assert scene.shown is None


def test_mission_objective_is_left_alone_when_objectives_are_not_translated(make_scene):
    scene = make_scene(translate_objectives=False)

    scene.show(1.0, row(1, OBJECTIVE, first_seen=1.0, stable=1))
    before_read = scene.shown
    scene.read(1.04, (1, "Find Dunk"))
    scene.show(1.066, row(1, OBJECTIVE, first_seen=1.0, stable=3))

    assert before_read is None and scene.shown is None
    assert scene.backend.calls == []


def finish_objective(scene, now):
    """Elengedi a hamis fordítót, megvárja a küldetéssor fordítását, és begyűjti a szimulált időben."""
    future = scene.pipeline._objective._future
    scene.backend.release.set()
    if future is not None:
        wait([future], timeout=WAIT_SECONDS)
    scene.backend.release.clear()
    return scene.pipeline.pump(now)


def test_mission_objective_is_not_hidden_before_it_is_read(make_scene):
    scene = make_scene()

    scene.show(1.0, row(1, OBJECTIVE, first_seen=1.0, stable=1))

    assert scene.shown is None
    assert scene.backend.calls == []


def test_mission_objective_is_translated_onto_its_own_line_under_the_dialogue(make_scene):
    scene = make_scene()
    scene.show(1.0, row(1, LOWER, first_seen=1.0), row(5, OBJECTIVE, first_seen=1.0))

    scene.read(1.04, (1, TROUBLE), (5, "Find Dunk"))
    while_translating = scene.shown
    scene.finish_translation()
    finish_objective(scene, 1.3)

    assert OBJECTIVE in while_translating.hidden  # az angol küldetéssor nem villan ki a fordításig
    assert scene.shown.lines == (ShownLine(hu(TROUBLE), LOWER), ShownLine(hu("Find Dunk"), OBJECTIVE))
    assert sorted(scene.backend.calls) == sorted([TROUBLE, "Find Dunk"])  # két külön kérés, nem egy felirat


def test_mission_objective_alone_is_translated(make_scene):
    scene = make_scene(only_spoken_lines=True)
    scene.show(1.0, row(5, OBJECTIVE, first_seen=1.0))

    scene.read(1.04, (5, "Go to your car"))
    finish_objective(scene, 1.3)

    assert scene.shown.lines == (ShownLine(hu("Go to your car"), OBJECTIVE),)


def test_mission_objective_is_not_translated_again_when_the_dialogue_changes(make_scene):
    scene = make_scene()
    objective = row(5, OBJECTIVE, first_seen=1.0)
    scene.show(1.0, row(1, LOWER, first_seen=1.0), objective)
    scene.read(1.04, (1, TROUBLE), (5, "Find Dunk"))
    scene.finish_translation()
    finish_objective(scene, 1.3)

    scene.show(3.0, row(2, LOWER, first_seen=3.0), objective)
    scene.read(3.04, (2, HOW))
    scene.translated(3.3)

    assert scene.backend.calls.count("Find Dunk") == 1
    assert scene.shown.lines == (ShownLine(hu(HOW), LOWER), ShownLine(hu("Find Dunk"), OBJECTIVE))


def test_mission_objective_translation_goes_when_the_objective_goes(make_scene):
    scene = make_scene()
    scene.show(1.0, row(5, OBJECTIVE, first_seen=1.0))
    scene.read(1.04, (5, "Find Dunk"))
    finish_objective(scene, 1.3)

    scene.show(2.0)

    assert scene.shown is None


@pytest.mark.parametrize("text", ["PICK UP LOOT", "Wanted"])
def test_text_under_the_dialogue_that_is_no_sentence_is_not_translated(make_scene, text):
    scene = make_scene()
    scene.show(1.0, row(5, OBJECTIVE, first_seen=1.0))

    scene.read(1.04, (5, text))

    assert scene.shown is None and scene.backend.calls == []


def test_row_that_is_not_a_subtitle_is_shown_again_and_never_hidden_twice(make_scene):
    scene = make_scene()
    scene.show(1.0, row(7, LOWER, first_seen=1.0, stable=1))

    scene.read(1.04, (7, "PICK UP LOOT"))
    after_read = scene.shown
    scene.show(1.066, row(7, LOWER, first_seen=1.0, stable=3))

    assert after_read is None and scene.shown is None
    assert scene.backend.calls == []


def test_row_known_not_to_be_a_subtitle_is_not_hidden_again_when_it_is_detected_anew(make_scene):
    scene = make_scene()
    scene.show(1.0, row(7, LOWER, first_seen=1.0, stable=1))
    scene.read(1.04, (7, "PICK UP LOOT"))

    scene.show(1.066, row(8, LOWER, first_seen=1.066, stable=1))  # ugyanott, új azonosítóval

    assert scene.shown is None


def test_low_confidence_read_is_not_a_subtitle(make_scene):
    scene = make_scene()
    scene.show(1.0, row(1, LOWER, first_seen=1.0))

    scene.read(1.04, (1, TROUBLE), score=AppConfig().min_text_score - 0.1)

    assert scene.shown is None
    assert scene.backend.calls == []


def test_row_that_is_read_again_successfully_is_translated_after_all(make_scene):
    scene = make_scene()
    scene.show(1.0, row(1, LOWER, first_seen=1.0))
    scene.read(1.04, (1, "We ain't lookin"), score=AppConfig().min_text_score - 0.1)
    after_poor_read = scene.shown

    scene.read(1.3, (1, TROUBLE))
    scene.translated(1.6)

    assert after_poor_read is None
    assert scene.shown.lines == (ShownLine(hu(TROUBLE), LOWER),)


# --- fordítás egyetlen olvasásból ---------------------------------------------------

def test_stable_row_is_translated_after_a_single_read(make_scene):
    scene = make_scene()
    scene.show(1.0, row(1, LOWER, first_seen=1.0, stable=1))
    scene.show(1.033, row(1, LOWER, first_seen=1.0, stable=2))

    scene.read(1.04, (1, TROUBLE))
    hidden_meanwhile = scene.shown
    scene.translated(1.3)

    assert hidden_meanwhile.hidden == (LOWER,) and hidden_meanwhile.text == ""
    assert scene.shown.lines == (ShownLine(hu(TROUBLE), LOWER),)
    assert scene.backend.calls == [TROUBLE]


def test_row_and_its_text_arriving_together_are_taken_in_the_right_order(make_scene):
    scene = make_scene()
    snapshot = BandSnapshot(BAND, AREA, FRAME, 1.0, (row(1, LOWER, first_seen=1.0),))

    scene.pipeline.receive(snapshot, [RowText(1, TROUBLE, 0.99)], 1.0)
    scene.translated(1.3)

    assert scene.shown.lines == (ShownLine(hu(TROUBLE), LOWER),)


def test_text_from_a_single_capture_waits_for_stable_pixels_but_is_translated_ahead(make_scene):
    scene = make_scene()
    scene.show(1.0, row(1, LOWER, first_seen=1.0, stable=1))

    scene.read(1.01, (1, TROUBLE))
    sent_ahead = scene.backend.called.wait(timeout=WAIT_SECONDS)
    scene.finish_translation()  # a fordító már válaszolhat is: amíg a felirat nem végleges, nem írjuk ki
    scene.pipeline.pump(1.02)
    before_second_capture = scene.shown
    scene.show(1.033, row(1, LOWER, first_seen=1.0, stable=2))
    scene.translated(1.3)

    assert sent_ahead and before_second_capture.lines == ()
    assert scene.shown.lines == (ShownLine(hu(TROUBLE), LOWER),)
    assert scene.backend.calls == [TROUBLE]  # a megerősítés az előre elindított fordítást kapja


def test_row_seen_on_a_single_capture_is_translated_when_no_newer_picture_arrives(make_scene):
    scene = make_scene()
    scene.show(1.0, row(1, LOWER, first_seen=1.0, stable=1))
    scene.read(1.04, (1, TROUBLE))

    scene.pipeline.pump(1.0 + STILL_SECONDS + 0.01)  # a kép áll: nincs mire várni a megerősítéssel
    scene.translated(1.5)

    assert scene.shown.lines == (ShownLine(hu(TROUBLE), LOWER),)


def test_tk_thread_is_woken_when_the_translation_is_ready(make_scene):
    scene = make_scene()
    scene.show(1.0, row(1, LOWER, first_seen=1.0))

    scene.read(1.04, (1, TROUBLE))
    woken_too_early = scene.woken.is_set()
    scene.backend.release.set()

    assert not woken_too_early
    assert scene.woken.wait(timeout=WAIT_SECONDS)


def test_text_for_a_row_that_is_gone_is_dropped(make_scene):
    scene = make_scene()
    scene.show(1.0, row(1, LOWER, first_seen=1.0, stable=1))
    scene.show(1.2, row(2, LOWER, first_seen=1.2, stable=1))

    changed = scene.read(1.25, (1, TROUBLE))  # az 1-es sor már nincs a képen

    assert not changed and scene.pipeline.stats.stale_texts == 1
    assert scene.backend.calls == []


def test_sentence_wrapped_over_two_rows_is_translated_as_one_line_over_both_rows(make_scene):
    scene = make_scene()
    first, second = "Got so bad once, I had bounties on me", "in three different states."
    scene.show(1.0, row(1, UPPER, first_seen=1.0), row(2, LOWER, first_seen=1.0))

    scene.read(1.04, (1, first), (2, second))
    scene.translated(1.3)

    whole = f"{first} {second}"
    assert scene.backend.calls == [whole]
    assert scene.shown.lines == (ShownLine(hu(whole), Box(UPPER.left, UPPER.top, UPPER.right, LOWER.bottom)),)


# --- görgő felirat ---------------------------------------------------------------------

def test_translation_follows_its_row_when_the_game_scrolls_it_without_a_new_read(make_scene):
    scene = make_scene()
    scene.speak(TROUBLE, LOWER)

    scene.show(3.0, row(1, UPPER, first_seen=1.0, stable=60), row(2, LOWER, first_seen=3.0, stable=1))

    assert scene.shown.lines == (ShownLine(hu(TROUBLE), UPPER),)  # azonnal, olvasás nélkül követi
    assert scene.shown.hidden == (LOWER,)  # az új sor már takarva
    assert scene.backend.calls == [TROUBLE]


def test_new_line_under_a_translated_one_stays_hidden_until_its_own_translation_arrives(make_scene):
    scene = make_scene()
    scene.speak(TROUBLE, LOWER)
    scene.show(3.0, row(1, UPPER, first_seen=1.0, stable=60), row(2, LOWER, first_seen=3.0, stable=1))
    scene.show(3.033, row(1, UPPER, first_seen=1.0, stable=61), row(2, LOWER, first_seen=3.0, stable=2))

    scene.read(3.04, (2, HOW))
    while_translating = scene.shown
    scene.translated(3.3)

    assert while_translating.lines == (ShownLine(hu(TROUBLE), UPPER),) and while_translating.hidden == (LOWER,)
    assert scene.shown.lines == (ShownLine(hu(TROUBLE), UPPER), ShownLine(hu(HOW), LOWER))
    assert sorted(scene.backend.calls) == sorted([TROUBLE, HOW])  # a régi sort nem fordítjuk újra


def test_unchanged_captures_cause_no_work_and_no_redraw(make_scene):
    scene = make_scene()
    scene.speak(TROUBLE, LOWER)
    updates_before = scene.pipeline.stats.state_updates

    changes = [scene.show(1.4 + step * 0.033, row(1, LOWER, first_seen=1.0, stable=10 + step))
               for step in range(30)]

    assert not any(changes)
    assert scene.pipeline.stats.state_updates == updates_before
    assert scene.backend.calls == [TROUBLE]


def test_old_translation_goes_as_soon_as_a_different_row_takes_its_place(make_scene):
    scene = make_scene()
    scene.speak(TROUBLE, LOWER)
    scene.show(2.967, row(1, LOWER, first_seen=1.0, stable=60))
    longer = Box(600, 957, 1320, 994)  # más szöveg ugyanazon a sorhelyen: más a szélessége

    scene.show(3.0, row(2, longer, first_seen=3.0, stable=1))

    assert scene.shown.lines == () and scene.shown.hidden == (longer,)


def test_translation_that_arrives_while_its_row_is_read_again_is_never_put_over_another_row(make_scene):
    scene = make_scene()
    scene.show(1.0, row(1, LOWER, first_seen=1.0))
    scene.read(1.04, (1, TROUBLE))
    wider = Box(LOWER.left - 60, LOWER.top, LOWER.right + 60, LOWER.bottom)
    scene.show(1.1, row(2, wider, first_seen=1.1, stable=1))  # a sor új azonosítót kapott, még olvasatlan

    scene.translated(1.15)
    while_unread = scene.shown
    scene.read(1.2, (2, TROUBLE))

    assert while_unread.text == "" and while_unread.hidden == (wider,)
    assert scene.shown.lines == (ShownLine(hu(TROUBLE), wider),)


# --- ugyanaz a sor új azonosítóval (megváltozott mögötte a háttér) --------------------

def test_row_detected_anew_at_the_same_place_keeps_its_translation_until_it_is_read(make_scene):
    scene = make_scene()
    scene.speak(TROUBLE, LOWER)
    scene.show(2.967, row(1, LOWER, first_seen=1.0, stable=60))
    wobbled = Box(LOWER.left + 2, LOWER.top + 1, LOWER.right + 3, LOWER.bottom + 1)  # a kereső doboza ingadozik

    scene.show(3.0, row(2, wobbled, first_seen=3.0, stable=1))

    assert [line.text for line in scene.shown.lines] == [hu(TROUBLE)]  # semmi sem villan: a fordítás marad
    assert scene.shown.hidden == ()


@pytest.mark.parametrize("only_spoken", [True, False])
def test_two_row_line_keeps_its_translation_while_one_row_is_detected_anew_somewhere_narrower(make_scene,
                                                                                               only_spoken):
    # Világos, mozgó háttér előtt a felső sor kisebb dobozt kap (nem ugyanaz a hely, nem örökli a szövegét):
    # amíg el nem olvassuk, az alsó sor maradéka („most evenings.”) nem lehet önálló felirat
    scene = make_scene(only_spoken_lines=only_spoken)
    said, second = "I will be located in Pelican Valley near Thorpe Adventures", "most evenings."
    first = f"Kirk: {said}"
    scene.show(1.0, row(1, UPPER, first_seen=1.0), row(2, LOWER, first_seen=1.0))
    scene.read(1.04, (1, first), (2, second))
    scene.translated(1.3)
    whole = f"{said} {second}"  # a fordító a beszélő neve nélkül kapja
    translated = scene.shown
    narrower = Box(UPPER.left + 40, UPPER.top + 2, UPPER.left + 300, UPPER.bottom - 1)

    scene.show(3.0, row(3, narrower, first_seen=3.0, stable=1), row(2, LOWER, first_seen=1.0, stable=60))
    while_unread = scene.shown
    scene.show(3.033, row(3, narrower, first_seen=3.0, stable=2), row(2, LOWER, first_seen=1.0, stable=61))
    scene.read(3.1, (3, first))

    assert while_unread.lines == translated.lines  # a fordítás kint marad, az alsó angol sor takarva
    assert scene.shown.lines == translated.lines
    assert scene.backend.calls == [whole]  # a maradékot nem fordítjuk külön


def test_row_read_without_its_speaker_name_stays_hidden_until_it_is_read_with_the_name(make_scene):
    # Világos háttér előtt a színes beszélőnév kimaradt a sor dobozából: az olvasat „: He's too…”
    scene = make_scene(only_spoken_lines=True)
    said, other = "He's too fucked up. No way he's getting up there.", "I came back here to get dressed!"
    scene.show(1.0, row(1, UPPER, first_seen=1.0), row(2, LOWER, first_seen=1.0))
    scene.read(1.04, (1, f": {said}"), (2, f"Mario: {other}"))
    scene.translated(1.3)
    nameless = scene.shown

    scene.show(1.4, row(3, UPPER, first_seen=1.4), row(2, LOWER, first_seen=1.0, stable=20))
    scene.read(1.45, (3, f"Final Chapter: {said}"))
    scene.translated(1.7)

    assert nameless is not None and UPPER in nameless.hidden  # az angol sor nem látszik
    assert not any(":" in call for call in scene.backend.calls)  # a név nélküli töredéket nem fordítjuk
    assert [line.text for line in scene.shown.lines] == [f"Final Chapter: {hu(said)}", f"Mario: {hu(other)}"]


def test_two_row_line_keeps_its_translation_while_one_row_drops_out_for_a_capture(make_scene):
    scene = make_scene(only_spoken_lines=True)
    said, second = "No, no... She's not a rat.", "She's just desperate because of Jordy."
    scene.show(1.0, row(1, UPPER, first_seen=1.0), row(2, LOWER, first_seen=1.0))
    scene.read(1.04, (1, f"Lucia: {said}"), (2, second))
    scene.translated(1.3)
    translated = scene.shown

    scene.show(2.0, row(1, UPPER, first_seen=1.0, stable=60, missing=1), row(2, LOWER, first_seen=1.0, stable=60))
    while_missing = scene.shown
    scene.show(2.02, row(1, UPPER, first_seen=1.0, stable=61), row(2, LOWER, first_seen=1.0, stable=61))

    assert while_missing.lines == translated.lines  # nem villan ki az angol
    assert scene.shown.lines == translated.lines


def test_reading_that_arrives_after_its_row_was_detected_anew_goes_to_the_row_in_its_place(make_scene):
    # Mozgó háttér előtt a sor szinte minden képen új azonosítót kap: az olvasás mire elkészül, a sor már eltűnt
    scene = make_scene()
    scene.show(1.0, row(1, LOWER, first_seen=1.0, stable=1))
    scene.show(1.02, row(2, Box(LOWER.left + 4, LOWER.top - 4, LOWER.right - 3, LOWER.bottom + 3), first_seen=1.02,
                         stable=1))
    scene.show(1.04, row(3, LOWER, first_seen=1.04, stable=1))

    scene.read(1.06, (1, TROUBLE))
    for step, now in enumerate((1.08, 1.10, 1.12)):  # a doboza is képről képre kicsit más
        scene.show(now, row(4 + step, Box(LOWER.left + step, LOWER.top, LOWER.right, LOWER.bottom), first_seen=now,
                            stable=1))
    scene.translated(1.3)

    assert [line.text for line in scene.shown.lines] == [hu(TROUBLE)]
    assert scene.backend.calls == [TROUBLE]


def test_subtitle_row_stays_subtitle_coloured_while_its_letters_fade_against_a_bright_background(make_scene):
    scene = make_scene(only_spoken_lines=True)
    said, second = "No, no... She's not a rat.", "She's just desperate because of Jordy."
    scene.show(1.0, row(1, UPPER, first_seen=1.0), row(2, LOWER, first_seen=1.0))
    scene.read(1.04, (1, f"Lucia: {said}"), (2, second))
    scene.translated(1.3)
    translated = scene.shown

    faded = Box(UPPER.left - 3, UPPER.top + 3, UPPER.right + 2, UPPER.bottom - 2)  # új azonosító, kevés fehér
    scene.show(2.0, row(3, faded, first_seen=2.0, stable=1, whiteness=0.01, sure=False),
               row(2, LOWER, first_seen=1.0, stable=60))

    assert scene.shown.lines == translated.lines


def test_inherited_text_gives_way_to_what_the_row_really_says(make_scene):
    scene = make_scene()
    scene.speak(TROUBLE, LOWER)
    scene.show(2.967, row(1, LOWER, first_seen=1.0, stable=60))
    scene.show(3.0, row(2, LOWER, first_seen=3.0, stable=1))
    scene.show(3.033, row(2, LOWER, first_seen=3.0, stable=2))

    scene.read(3.04, (2, HOW))
    while_translating = scene.shown
    scene.translated(3.3)

    assert while_translating.lines == () and while_translating.hidden == (LOWER,)
    assert scene.shown.lines == (ShownLine(hu(HOW), LOWER),)


def test_inherited_text_expires_when_the_row_is_never_read(make_scene):
    scene = make_scene()
    scene.speak(TROUBLE, LOWER)
    scene.show(2.967, row(1, LOWER, first_seen=1.0, stable=60))
    scene.show(3.0, row(2, LOWER, first_seen=3.0, stable=1))

    expired_at = scene.tick_until_change(3.01, 5.0)
    after_expiry = scene.shown
    uncovered_at = expired_at
    while scene.shown is not None and uncovered_at < 5.0:
        uncovered_at += TICK
        scene.pipeline.pump(uncovered_at)

    # Nem tudjuk, mit ír a sor: a régi fordítás lekerül, a sort olvasatlanként takarjuk, aztán az sem marad rajta
    assert EAGER_HIDE_SECONDS <= expired_at - 3.0 <= EAGER_HIDE_SECONDS + TICK
    assert after_expiry.lines == () and after_expiry.hidden == (LOWER,)
    assert scene.shown is None
    assert UNREAD_HIDE_SECONDS <= uncovered_at - 3.0 <= UNREAD_HIDE_SECONDS + TICK


# --- eltűnő felirat és késő fordítás ------------------------------------------------

def test_translation_lingers_for_the_hold_time_after_its_row_disappears(make_scene):
    scene = make_scene()
    scene.speak(TROUBLE, LOWER)
    scene.show(2.0, row(1, LOWER, first_seen=1.0, stable=30))

    scene.show(2.033)
    lingering = scene.shown
    gone_at = scene.tick_until_change(2.04, 4.0)

    assert lingering.lines == (ShownLine(hu(TROUBLE), LOWER),)
    assert scene.shown is None
    assert HOLD <= gone_at - 2.0 <= HOLD + RECHECK_SECONDS + TICK


def test_row_that_is_only_remembered_does_not_keep_the_subtitle_alive(make_scene):
    scene = make_scene()
    scene.speak(TROUBLE, LOWER)
    scene.show(2.0, row(1, LOWER, first_seen=1.0, stable=30))

    for step in range(1, 11):  # a sor már nem látszik, de a sorkövető még nem engedte el
        scene.show(2.0 + step * 0.033, row(1, LOWER, first_seen=1.0, stable=30, missing=step))
    scene.show(2.363)
    gone_at = scene.tick_until_change(2.37, 4.0)

    assert HOLD <= gone_at - 2.033 <= HOLD + RECHECK_SECONDS + TICK  # az óra az első hiányzó képtől jár


def test_subtitle_that_vanishes_after_a_long_still_picture_still_lingers_for_the_hold_time(make_scene):
    scene = make_scene()
    scene.speak(TROUBLE, LOWER)
    scene.tick_until_change(1.3, 6.0)  # öt másodpercig nem jön újabb kép: a rögzítés csak változáskor ad sávot

    scene.show(6.0)
    lingering = scene.shown
    gone_at = scene.tick_until_change(6.01, 8.0)

    assert lingering.lines == (ShownLine(hu(TROUBLE), LOWER),)
    assert HOLD <= gone_at - 6.0 <= HOLD + RECHECK_SECONDS + TICK


def test_translation_lingers_while_an_unrelated_row_stays_on_screen(make_scene):
    scene = make_scene(translate_objectives=False)
    objective = row(5, OBJECTIVE, first_seen=1.0)
    scene.show(1.0, row(1, LOWER, first_seen=1.0), objective)
    scene.read(1.04, (1, TROUBLE), (5, "Find Dunk"))
    scene.translated(1.3)
    scene.show(2.0, row(1, LOWER, first_seen=1.0, stable=30), objective)

    scene.show(2.033, objective)  # a párbeszédsor eltűnt, a küldetésutasítás maradt
    lingering = scene.shown
    gone_at = scene.tick_until_change(2.04, 4.0)

    assert lingering.lines == (ShownLine(hu(TROUBLE), LOWER),)
    assert scene.shown is None
    assert HOLD <= gone_at - 2.0 <= HOLD + RECHECK_SECONDS + TICK


def test_subtitle_that_cleared_starts_from_scratch_when_it_returns(make_scene):
    scene = make_scene()
    scene.speak(TROUBLE, LOWER)
    scene.show(2.0, row(1, LOWER, first_seen=1.0, stable=30))
    scene.show(2.033)
    scene.tick_until_change(2.04, 4.0)
    cleared = scene.shown
    scene.woken.clear()

    scene.show(5.0, row(2, LOWER, first_seen=5.0))
    scene.read(5.04, (2, TROUBLE))
    translated_again = scene.woken.wait(timeout=WAIT_SECONDS)  # új feliratként indult újra a fordítása
    scene.pipeline.pump(5.3)

    assert cleared is None and translated_again
    assert scene.shown.lines == (ShownLine(hu(TROUBLE), LOWER),)
    assert scene.backend.calls == [TROUBLE]  # másodszorra a gyorsítótárból jön


def test_untranslated_line_shows_again_when_its_translation_is_late(make_scene):
    scene = make_scene()
    scene.show(1.0, row(1, LOWER, first_seen=1.0))

    scene.read(1.04, (1, TROUBLE))
    hidden = scene.shown
    shown_again_at = scene.tick_until_change(1.05, 4.0)

    assert hidden.hidden == (LOWER,)
    assert scene.shown is None  # a fordítás késik: az angol sor újra látszik
    assert PENDING_HIDE_SECONDS <= shown_again_at - 1.04 <= PENDING_HIDE_SECONDS + RECHECK_SECONDS + TICK


def test_failed_translation_shows_the_block_message(make_scene):
    scene = make_scene(FailingBackend())
    scene.show(1.0, row(1, LOWER, first_seen=1.0))

    scene.read(1.04, (1, TROUBLE))
    scene.translated(1.3)

    assert scene.shown.text == t("pipeline.failed_prefix") + " " + TROUBLE
    assert scene.shown.lines == () and scene.shown.anchor == LOWER


def test_new_line_on_the_last_picture_is_confirmed_by_time_even_while_a_message_is_shown(make_scene):
    scene = make_scene(FailingBackend())
    scene.show(1.0, row(1, LOWER, first_seen=1.0))
    scene.read(1.04, (1, TROUBLE))
    scene.translated(1.3)  # hibaüzenet áll a képen: semmit sem takarunk, nincs miért sűrűn ránézni
    scene.show(2.0, row(1, UPPER, first_seen=1.0, stable=30), row(2, LOWER, first_seen=2.0, stable=1))
    scene.read(2.04, (2, HOW))

    scene.pipeline.pump(2.0 + STILL_SECONDS + 0.01)  # nem jött újabb kép
    scene.translated(2.5)

    assert scene.shown.text == t("pipeline.failed_prefix") + f" {TROUBLE}\n{HOW}"
    assert scene.shown.anchor == Box(UPPER.left, UPPER.top, UPPER.right, LOWER.bottom)


def test_translation_that_never_arrives_shows_the_timeout_message(make_scene, monkeypatch):
    monkeypatch.setattr(subtitle_pipeline, "TRANSLATION_DEADLINE_SECONDS", 0.0)
    scene = make_scene()
    scene.show(1.0, row(1, LOWER, first_seen=1.0))

    scene.read(1.04, (1, TROUBLE))

    assert scene.shown.text == t("pipeline.timeout_prefix", seconds=0.0) + " " + TROUBLE


# --- csak a kimondott szöveg ------------------------------------------------------

def test_only_spoken_lines_leaves_mission_text_alone(make_scene):
    scene = make_scene(only_spoken_lines=True)
    scene.show(1.0, row(1, LOWER, first_seen=1.0))

    scene.read(1.04, (1, MISSION))

    assert scene.shown is None
    assert scene.backend.calls == []


def test_spoken_line_is_translated_when_only_spoken_lines_is_on(make_scene):
    scene = make_scene(only_spoken_lines=True)

    scene.speak("Lucia: Stop talking.", LOWER)

    assert scene.backend.calls == ["Stop talking."]  # a beszélő nevét nem küldjük fordításra
    assert scene.shown.lines == (ShownLine("Lucia: " + hu("Stop talking."), LOWER),)


def test_spoken_line_under_a_music_caption_is_translated_when_only_spoken_lines_is_on(make_scene):
    scene = make_scene(only_spoken_lines=True)
    scene.show(1.0, row(1, UPPER, first_seen=1.0), row(2, LOWER, first_seen=1.0))

    scene.read(1.04, (1, "[♪ Money For Nothing – Dire Straits ♪]"), (2, "Lucia: Let's go!"))
    scene.translated(1.3)

    assert scene.backend.calls == ["Let's go!"]
    assert scene.shown.lines == (ShownLine("Lucia: " + hu("Let's go!"), LOWER),)


def test_switching_only_spoken_lines_on_removes_a_translated_mission_text(make_scene):
    scene = make_scene()
    scene.speak(MISSION, LOWER)

    scene.pipeline.update_config(AppConfig(only_spoken_lines=True))
    changed = scene.pipeline.pump(1.4)

    assert changed and scene.shown is None


# --- ablakméret ---------------------------------------------------------------------

def test_snapshot_made_for_another_window_size_is_ignored(make_scene):
    scene = make_scene()
    scene.pipeline.set_frame_size(FRAME)

    changed = scene.show(1.0, row(1, LOWER, first_seen=1.0, stable=1), frame_size=SMALL_FRAME)

    assert not changed and scene.shown is None
    assert scene.pipeline.snapshot is None and scene.pipeline.stats.stale_snapshots == 1


def test_resizing_the_window_drops_rows_and_late_results_but_keeps_the_translation(make_scene):
    scene = make_scene()
    scene.speak(TROUBLE, LOWER)
    scaled = Box(520, 638, 760, 663)  # ugyanaz a sor a kisebb ablakban

    scene.pipeline.set_frame_size(SMALL_FRAME)
    after_resize = scene.shown
    late_result_used = scene.read(1.5, (1, HOW))  # a régi méretű kép sorához tartozik
    scene.show(1.6, row(9, scaled, first_seen=1.6), frame_size=SMALL_FRAME)
    scene.read(1.64, (9, TROUBLE))

    assert after_resize is None  # a régi dobozok sehová sem mutatnak
    assert not late_result_used and scene.pipeline.stats.stale_texts == 1
    assert scene.shown.lines == (ShownLine(hu(TROUBLE), scaled),)
    assert scene.backend.calls == [TROUBLE]  # nem fordítottunk újra


def test_snapshot_of_a_new_size_resets_the_places_even_when_nobody_announced_it(make_scene):
    scene = make_scene()
    scene.speak(TROUBLE, LOWER)

    changed = scene.show(1.5, frame_size=SMALL_FRAME)

    assert changed and scene.shown is None


def test_accepted_snapshot_is_available_as_the_backdrop_of_what_is_shown(make_scene):
    scene = make_scene()
    snapshot = BandSnapshot(BAND, AREA, FRAME, 1.0, (row(1, LOWER, first_seen=1.0, stable=1),))

    scene.pipeline.receive(snapshot, (), 1.0)

    assert scene.pipeline.snapshot is snapshot


# --- a régi, teljes olvasású út ugyanazon a példányon ---------------------------------

def test_full_read_path_still_works_on_a_row_pipeline(make_scene):
    scene = make_scene(reader=LineReader(LOWER))

    scene.read_fully(0.0)
    scene.read_fully(1.0)  # két egyező olvasás kell a megerősítéshez
    scene.finish_translation()
    scene.pipeline.poll(1.1)

    assert scene.shown.lines == (ShownLine(hu(TROUBLE), LOWER),)


def test_full_read_path_without_line_boxes_still_shows_one_block(make_scene):
    scene = make_scene(reader=LineReader(LOWER, with_lines=False))

    scene.read_fully(0.0)
    scene.read_fully(1.0)
    scene.finish_translation()
    scene.pipeline.poll(1.1)

    assert scene.shown.text == hu(TROUBLE) and scene.shown.lines == ()
    assert scene.shown.anchor == LOWER


def test_full_read_path_takes_over_when_the_rows_are_suspended(make_scene):
    scene = make_scene(reader=LineReader(UPPER))
    scene.speak(TROUBLE, LOWER)

    scene.pipeline.suspend_rows()
    kept = scene.shown
    scene.read_fully(1.5)  # a régi út ugyanazt a sort feljebb görgetve olvassa

    assert kept.lines == (ShownLine(hu(TROUBLE), LOWER),)
    assert scene.shown.lines == (ShownLine(hu(TROUBLE), UPPER),)
    assert scene.backend.calls == [TROUBLE]


def test_suspending_the_rows_uncovers_what_only_the_row_path_would_uncover_again(make_scene):
    scene = make_scene()
    scene.speak(TROUBLE, LOWER)
    scene.show(2.0, row(1, UPPER, first_seen=1.0, stable=30), row(2, LOWER, first_seen=2.0, stable=1))
    covered_before = scene.shown.hidden

    scene.pipeline.suspend_rows()

    assert covered_before == (LOWER,)  # az új sort olvasás előtt takartuk: ezt a régi út sosem venné le
    assert scene.shown.hidden == () and scene.shown.lines == (ShownLine(hu(TROUBLE), UPPER),)


def test_row_read_before_a_suspension_keeps_its_text_when_the_row_path_resumes(make_scene):
    scene = make_scene()
    scene.speak(TROUBLE, LOWER)
    scene.pipeline.suspend_rows()

    scene.show(1.5, row(1, LOWER, first_seen=1.0, stable=15))  # a felismerő ezt a sort már nem olvassa újra

    assert scene.shown.lines == (ShownLine(hu(TROUBLE), LOWER),)
    assert scene.shown.hidden == ()


def test_timer_leaves_the_subtitle_alone_while_the_rows_are_suspended(make_scene):
    scene = make_scene()
    scene.speak(TROUBLE, LOWER)
    scene.pipeline.suspend_rows()

    changed = scene.pipeline.pump(1.3 + HOLD + 5)

    assert not changed
    assert scene.shown.lines == (ShownLine(hu(TROUBLE), LOWER),)


def test_timer_does_not_keep_the_subtitle_alive_for_the_full_read_path(make_scene):
    scene = make_scene(reader=EmptyReader())
    scene.speak(TROUBLE, LOWER)
    scene.pipeline.suspend_rows()

    scene.pipeline.pump(5.0)
    scene.read_fully(5.05)  # a régi út már üres sávot olvas: a felirat régen eltűnt

    assert scene.shown is None


# --- a felirat eltűnése álló jeleneten ---------------------------------------------------

VIDEO_FRAME_GAP = 0.033  # a capture card képét mutató ablak (pl. Veyra) álló jeleneten is ~30 képet küld másodpercenként


def test_translation_disappears_while_a_video_window_keeps_sending_the_empty_band(make_scene):
    scene = make_scene()
    scene.speak(TROUBLE, LOWER)
    scene.show(2.0, row(1, LOWER, first_seen=1.0, stable=30))

    now, next_frame, gone_at = 2.033, 2.033, None
    while now < 6.0 and gone_at is None:
        if now >= next_frame:
            scene.show(now)  # az angol sor eltűnt: minden képkockán ugyanaz az üres sáv
            next_frame += VIDEO_FRAME_GAP
        else:
            scene.pipeline.pump(now)
        gone_at = now if scene.shown is None else None
        now += TICK

    assert gone_at is not None
    assert gone_at - 2.033 <= HOLD + RECHECK_SECONDS + VIDEO_FRAME_GAP + TICK


def test_failed_reads_on_the_full_reading_path_do_not_keep_the_translation_up(make_scene):
    class FailingReader:
        def read_band(self, band, area):
            raise RuntimeError("the capture failed")

    scene = make_scene(reader=FailingReader())
    scene.speak(TROUBLE, LOWER)
    scene.pipeline.suspend_rows()  # a régi, teljes olvasású út folytatja

    scene.read_fully(1.5)
    lingering = scene.shown
    scene.read_fully(1.5 + HOLD)

    assert lingering is not None  # egy hibás olvasás még nem tünteti el (nem villan)
    assert scene.shown is None


@pytest.mark.parametrize("missing_captures", [1, 2])
def test_translation_disappears_when_the_picture_stops_while_its_row_is_only_remembered(make_scene,
                                                                                       missing_captures):
    scene = make_scene()
    scene.speak(TROUBLE, LOWER)
    scene.show(2.0, row(1, LOWER, first_seen=1.0, stable=30))
    for step in range(1, missing_captures + 1):  # elhalványul: a sorkövető még emlékszik rá
        scene.show(2.0 + step * VIDEO_FRAME_GAP, row(1, LOWER, first_seen=1.0, stable=30, missing=step))

    gone_at = scene.tick_until_change(2.1, 6.0)  # álló jelenet: több kép nem jön

    assert gone_at is not None
    assert gone_at - 2.033 <= HOLD + RECHECK_SECONDS + TICK


# --- a tévéműsor felirata ---------------------------------------------------------------

TV_LINE = "Noah. What is your name?"


@pytest.mark.parametrize("text, is_dim, is_translated", [
    pytest.param(TV_LINE, True, True, id="dim tv subtitle"),
    # Valós eset: a rajzfilm felirata hófehér, vékony betűs, körvonal nélküli, két sorban
    pytest.param("If you come back without my money, I'll turn you into cowboy boots!", False, True,
                 id="white tv subtitle that sounds spoken"),
    pytest.param(TV_LINE, False, True, id="white question"),
    pytest.param(MISSION, False, False, id="white mission text"),
    pytest.param("Press the button to open the map.", False, False, id="white help text"),
    pytest.param("Lose the cops", False, False, id="white objective"),
])
def test_nameless_tv_subtitle_is_spoken_text_too(make_scene, text, is_dim, is_translated):
    # Valós hiba: a tévéműsor feliratában nincs „Név:”, ezért a „Csak a kimondott szöveg” angolul hagyta
    scene = make_scene(only_spoken_lines=True)
    scene.show(1.0, row(1, LOWER, first_seen=1.0, dim=is_dim))

    scene.read(1.04, (1, text))
    scene.translated(1.3)

    assert (scene.backend.calls == [text]) == is_translated
    assert (scene.shown is not None and scene.shown.text == hu(text)) == is_translated
