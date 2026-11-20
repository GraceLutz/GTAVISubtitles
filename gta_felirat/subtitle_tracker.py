"""Eldönti, mikor jelent meg új felirat, és mikor tűnt el.

Az OCR minden olvasásnál kicsit mást adhat (egy-egy betű eltérés), ezért
hasonlóság alapján döntünk, és csak stabil szöveget küldünk fordításra.
"""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class NewSubtitle:
    text: str


@dataclass(frozen=True)
class SubtitleCleared:
    pass


TrackerEvent = NewSubtitle | SubtitleCleared


def normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def similarity(first: str, second: str) -> float:
    if not first or not second:
        return 0.0
    return SequenceMatcher(None, first.lower(), second.lower()).ratio()


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

    def update(self, raw_text: str, now: float) -> TrackerEvent | None:
        text = normalize(raw_text)
        if not text:
            return self._handle_empty(now)
        if similarity(text, self._committed) >= self._threshold:
            self._last_seen = now
            self._candidate, self._candidate_count = "", 0
            return None
        return self._handle_candidate(text, now)

    def _handle_empty(self, now: float) -> TrackerEvent | None:
        self._candidate, self._candidate_count = "", 0
        if self._committed and now - self._last_seen >= self._hold_seconds:
            self._committed = ""
            return SubtitleCleared()
        return None

    def _handle_candidate(self, text: str, now: float) -> TrackerEvent | None:
        if similarity(text, self._candidate) >= self._threshold:
            self._candidate_count += 1
            # A hosszabb változat általában a teljesebb olvasat
            if len(text) > len(self._candidate):
                self._candidate = text
        else:
            self._candidate, self._candidate_count = text, 1
        if self._candidate_count < self._stable_reads:
            return None
        self._committed = self._candidate
        self._last_seen = now
        self._candidate, self._candidate_count = "", 0
        return NewSubtitle(self._committed)
