"""Eldönti, mikor jelent meg új felirat, és mikor tűnt el.

Az OCR minden olvasásnál kicsit mást adhat (egy-egy betű eltérés), ezért
hasonlóság alapján döntünk, és csak stabil szöveget küldünk fordításra.
"""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

_WHITESPACE = re.compile(r"\s+")
CUT_START_MARKS = ",.;:!?)]"  # ezzel kezdődő olvasatnak az eleje lemaradt
APOSTROPHES = "'’"


@dataclass(frozen=True)
class NewSubtitle:
    text: str


@dataclass(frozen=True)
class SubtitleCleared:
    pass


TrackerEvent = NewSubtitle | SubtitleCleared


def normalize(text: str) -> str:
    """Egységes szóközök; a sortörés (megszólalásonként külön sor) megmarad."""
    lines = (_WHITESPACE.sub(" ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def similarity(first: str, second: str) -> float:
    if not first or not second:
        return 0.0
    return SequenceMatcher(None, first.lower(), second.lower()).ratio()


def is_same_subtitle(first: str, second: str, threshold: float) -> bool:
    """Ugyanaz a felirat: ugyanannyi sor, és soronként hasonlók.

    Soronként nézzük, mert egy hosszú sor alá érkező rövid új sor („You're good.”) az egész
    szöveghez mérve alig számít, pedig új megszólalás.
    """
    first_lines, second_lines = first.split("\n"), second.split("\n")
    return len(first_lines) == len(second_lines) and all(
        similarity(a, b) >= threshold for a, b in zip(first_lines, second_lines))


def is_cut_at_start(text: str) -> bool:
    """A felirat eleje lemaradt: a játék sora nagybetűvel, idézőjellel vagy aposztróffal kezdődik, sosem
    kisbetűvel vagy írásjellel („eed y'all…”, „, I'mma…”, „ucia: I see…”). Ezt nem fordítjuk le: a teljes
    olvasat egy pillanat múlva jön."""
    start = text.lstrip()
    first = start[:1]
    if first in APOSTROPHES:
        first = start[1:2]  # „'cause you're…” csonka, az „'Ey, Pooch.” teljes
    return first.islower() or first in CUT_START_MARKS


def is_fuller(new: str, old: str) -> bool:
    """A new ugyanannak a feliratnak a teljesebb olvasata: egymást követő soraiban soronként tartalmazza az old
    sorait, és betűket is ad hozzá. A felirat beúszásakor az OCR a sor elejét vagy végét elvághatja („ul: We gotta…”
    → „Raul: We gotta…”, „mamit” → „mamita.”), vagy egy egész sort kihagyhat (két beszélőnél a felsőt); ilyenkor a
    teljes szöveget kell fordítani, különben a csonka marad kint."""
    new_lines, old_lines = new.split("\n"), old.split("\n")
    if not old or len(new_lines) < len(old_lines):
        return False
    added = sum(ch.isalpha() for ch in new) - sum(ch.isalpha() for ch in old)
    return added > 0 and any(_contains_lines(new_lines[start:start + len(old_lines)], old_lines)
                             for start in range(len(new_lines) - len(old_lines) + 1))


def _contains_lines(new_lines: list[str], old_lines: list[str]) -> bool:
    return all(old_line.lower() in new_line.lower() for new_line, old_line in zip(new_lines, old_lines))


class SubtitleTracker:
    def __init__(self, stable_reads: int, similarity_threshold: float, hold_seconds: float) -> None:
        self._stable_reads = stable_reads
        self._threshold = similarity_threshold
        self._hold_seconds = hold_seconds
        self._committed = ""
        self._last_seen = 0.0
        self._candidate = ""
        self._candidate_count = 0

    @property
    def current(self) -> str:
        return self._committed

    def update(self, raw_text: str, now: float, confirmed: bool = False) -> TrackerEvent | None:
        """Egy olvasás eredménye.

        confirmed: a hívó már tudja, hogy a szöveg stabil (a soralapú úton a sor képpontjai nem
        változnak), ezért nem kell hozzá még egy egyező olvasás.
        """
        text = normalize(raw_text)
        if not text:
            return self._handle_empty(now)
        if is_cut_at_start(text):
            if self._committed:
                self._last_seen = now  # a felirat még látszik, csak az elejét nem láttuk
            return None
        is_same = is_same_subtitle(text, self._committed, self._threshold) and not is_fuller(text, self._committed)
        # A kiírt felirat egy részének olvasata („Yeah, Dre'Quan,” a teljes sor után): ugyanaz a sor látszik még
        if is_same or is_fuller(self._committed, text):
            self._last_seen = now
            self._candidate, self._candidate_count = "", 0
            return None
        if confirmed:
            return self._commit(text, now)
        return self._handle_candidate(text, now)

    def touch(self, now: float) -> None:
        """A kiírt felirat még látszik (a képpontok szerint): az eltűnés utáni várakozás órája innen indul."""
        if self._committed:
            self._last_seen = now

    def _handle_empty(self, now: float) -> TrackerEvent | None:
        self._candidate, self._candidate_count = "", 0
        if self._committed and now - self._last_seen >= self._hold_seconds:
            self._committed = ""
            return SubtitleCleared()
        return None

    def _handle_candidate(self, text: str, now: float) -> TrackerEvent | None:
        if is_same_subtitle(text, self._candidate, self._threshold):
            self._candidate_count += 1
            # A hosszabb változat általában a teljesebb olvasat
            if len(text) > len(self._candidate):
                self._candidate = text
        else:
            self._candidate, self._candidate_count = text, 1
        if self._candidate_count < self._stable_reads:
            return None
        return self._commit(self._candidate, now)

    def _commit(self, text: str, now: float) -> NewSubtitle:
        self._committed = text
        self._last_seen = now
        self._candidate, self._candidate_count = "", 0
        return NewSubtitle(text)
