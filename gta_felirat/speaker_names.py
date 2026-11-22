"""A beszélőnevek megtanulása és a félreolvasott nevek javítása.

Az OCR a nevet néha egy-két betűvel elrontja („Raut”, „L'ucia”, „Toufists”). A feliratban a név a fordítás
része, és a színét is a név alapján kapja, ezért a hibás nevet az ismert névre javítjuk. Ismert név: a
játékban gyakori szereplők listája, és amit a program maga megtanult: egy új nevet, amely nem hasonlít
egyetlen ismertre sem, sokszor egymás után olvasott.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

LEARNED_NAMES_FILE = "tanult_nevek.json"
# A játékban gyakori beszélők (a felvételek naplójából): ezeket az első olvasástól javítjuk
KNOWN_SPEAKERS = ("Bae-Luxe", "Boobie", "Clerk", "Cop", "DWNPLY", "Dre'Quan", "Jacki", "Jason", "Kamal", "Levi",
                  "Local", "Locals", "Lucia", "Partygoer", "Pooch", "Raul", "Roxy", "SWAT", "Staff Member",
                  "Tourist", "Tourists", "Wax")
LEARN_AFTER_READS = 30  # ennyi olvasás után lesz egy új név ismert
MAX_SUBSTITUTIONS = 2  # azonos hosszú névben legfeljebb ennyi betű lehet félreolvasva
SPEAKER_PREFIX = re.compile(r"^([A-Z][\w.'’&\- ]{0,23}?)(:\s)")


def _letters(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalpha())


def is_misread_of(name: str, known: str) -> bool:
    """A name a known OCR-hibás olvasata: ugyanazok a betűk írásjel nélkül („L'ucia”, „Dre Quan”), legfeljebb
    két betű tér el azonos hosszon („Raut”, „Toufists”), vagy egy betűvel több („GTourist”, „Bae-Lauxe”).
    Rövidebb név nem az: a „Jack” más szereplő is lehet, mint a „Jacki”."""
    if name == known:
        return False
    plain, target = _letters(name), _letters(known)
    if plain == target:
        return True
    if len(plain) == len(target):
        return sum(a != b for a, b in zip(plain, target)) <= min(MAX_SUBSTITUTIONS, len(target) // 3)
    if len(plain) == len(target) + 1 and len(target) >= 4:
        return any(plain[:i] + plain[i + 1:] == target for i in range(len(plain)))
    return False


class SpeakerBook:
    """Az ismert nevek; a saját olvasásaiból tanul (path: ide menti a megtanult neveket)."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._counts: dict[str, int] = {}
        self._learned: set[str] = set()
        if path is not None and path.exists():
            try:
                self._learned = set(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError) as error:
                logger.warning("A megtanult nevek nem olvashatók (%s): üres listával indulok", error)

    @property
    def known(self) -> set[str]:
        return set(KNOWN_SPEAKERS) | self._learned

    def fix(self, text: str) -> str:
        """A szöveg soronként, a félreolvasott beszélőnév helyén az ismert névvel."""
        return "\n".join(self._fixed_line(line) for line in text.split("\n"))

    def _fixed_line(self, line: str) -> str:
        match = SPEAKER_PREFIX.match(line)
        if match is None:
            return line
        name = match.group(1)
        known = self.known
        if name in known:
            return line
        candidates = [other for other in known if is_misread_of(name, other)]
        if len(candidates) == 1:
            return candidates[0] + line[match.end(1):]
        if not candidates:
            self._count(name)
        return line  # több ismert névre is hasonlít: nem tudjuk, melyik, marad

    def _count(self, name: str) -> None:
        self._counts[name] = self._counts.get(name, 0) + 1
        if self._counts[name] == LEARN_AFTER_READS:
            self._learned.add(name)
            logger.info("Új beszélőnév megtanulva: %s", name)
            self._save()

    def _save(self) -> None:
        if self._path is None:
            return
        try:
            self._path.write_text(json.dumps(sorted(self._learned), ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError as error:
            logger.warning("A megtanult nevek nem menthetők (%s)", error)


_book = SpeakerBook()


def use_file(path: Path) -> None:
    """Innentől ebből a fájlból tanul és ide ment (a config.json mellé)."""
    global _book
    _book = SpeakerBook(path)


def fix_speaker_names(text: str) -> str:
    return _book.fix(text)
