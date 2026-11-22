"""A beszélőnevek megtanulása és a félreolvasott nevek javítása.

Az OCR a nevet néha egy-két betűvel elrontja („Raut”, „L'ucia”, „Toufists”). A feliratban a név a fordítás
része, és a színét is a név alapján kapja, ezért a hibás nevet az ismert névre javítjuk. Ismert név: a
játékban gyakori szereplők listája, és amit a program maga megtanult: egy új nevet, amely nem hasonlít
egyetlen ismertre sem, sokszor egymás után olvasott.

A program a tanultakat magától a config.json „speaker_names” kulcsába írja:
{"learned": ["Brian"], "fixes": {"Raut": {"name": "Raul", "count": 12}}}
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

logger = logging.getLogger(__name__)

# A játékban gyakori beszélők (a felvételek naplójából): ezeket az első olvasástól javítjuk
KNOWN_SPEAKERS = ("Bae-Luxe", "Boobie", "Clerk", "Cop", "DWNPLY", "Dre'Quan", "Jacki", "Jason", "Kamal", "Levi",
                  "Local", "Locals", "Lucia", "Partygoer", "Pooch", "Raul", "Roxy", "SWAT", "Staff Member",
                  "Tourist", "Tourists", "Wax")
LEARN_AFTER_READS = 30  # ennyi olvasás után lesz egy új név ismert
MAX_SUBSTITUTIONS = 2  # azonos hosszú névben legfeljebb ennyi betű lehet félreolvasva
SAVE_FIX_COUNT_EVERY = 10  # egy ismert javítás számlálóját ennyi újabb javításonként menti (ne írjon minden sornál)
SPEAKER_PREFIX = re.compile(r"^([A-Z][\w.'’&\- ]{0,23}?)(:\s)")

SaveCallback = Callable[[dict], None]


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
    """Az ismert nevek és a javítások; a saját olvasásaiból tanul.

    data: a config.json „speaker_names” értéke; save: ezzel menti az új állapotot (None: nem ment)."""

    def __init__(self, data: dict | None = None, save: SaveCallback | None = None) -> None:
        self._save_callback = save
        self._counts: dict[str, int] = {}
        self._learned, self._fixes = _read_data(data)

    @property
    def known(self) -> set[str]:
        return set(KNOWN_SPEAKERS) | self._learned

    @property
    def data(self) -> dict:
        """A config.json-ba írható alak."""
        return {"learned": sorted(self._learned),
                "fixes": {wrong: dict(fix) for wrong, fix in sorted(self._fixes.items())}}

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
            self._record_fix(name, candidates[0])
            return candidates[0] + line[match.end(1):]
        if not candidates:
            self._count(name)
        return line  # több ismert névre is hasonlít: nem tudjuk, melyik, marad

    def _record_fix(self, wrong: str, right: str) -> None:
        fix = self._fixes.get(wrong)
        if fix is None or fix["name"] != right:
            self._fixes[wrong] = {"name": right, "count": 1}
            logger.info("Névjavítás: %s → %s", wrong, right)
            self._save()
            return
        fix["count"] += 1
        if fix["count"] % SAVE_FIX_COUNT_EVERY == 0:
            self._save()

    def _count(self, name: str) -> None:
        self._counts[name] = self._counts.get(name, 0) + 1
        if self._counts[name] == LEARN_AFTER_READS:
            self._learned.add(name)
            logger.info("Új beszélőnév megtanulva: %s", name)
            self._save()

    def _save(self) -> None:
        if self._save_callback is None:
            return
        try:
            self._save_callback(self.data)
        except (OSError, ValueError) as error:
            logger.warning("A tanult nevek nem menthetők (%s)", error)


def _read_data(data: dict | None) -> tuple[set[str], dict[str, dict]]:
    """A config.json-ból olvasott állapot; a hibás részt kihagyja (figyelmeztetéssel), az indulást nem akadályozza."""
    if not data:
        return set(), {}
    if not isinstance(data, dict):
        logger.warning("A speaker_names nem objektum: üres listával indulok")
        return set(), {}
    learned_raw = data.get("learned", [])
    fixes_raw = data.get("fixes", {})
    learned_items = learned_raw if isinstance(learned_raw, list) else [None]
    fix_items = fixes_raw.items() if isinstance(fixes_raw, dict) else [(None, None)]
    learned = {name for name in learned_items if isinstance(name, str)}
    fixes = {wrong: {"name": fix["name"], "count": fix["count"]}
             for wrong, fix in fix_items if _is_valid_fix(fix)}
    if len(learned) < len(learned_items) or len(fixes) < len(fix_items):
        logger.warning("A speaker_names egy része hibás, azt kihagytam")
    return learned, fixes


def _is_valid_fix(fix) -> bool:
    return (isinstance(fix, dict) and isinstance(fix.get("name"), str)
            and isinstance(fix.get("count"), int) and fix["count"] >= 0)


_book = SpeakerBook()


def use_config(data: dict | None, save: SaveCallback | None) -> None:
    """Innentől a config.json „speaker_names” értékéből tanul, és a save-vel menti oda."""
    global _book
    _book = SpeakerBook(data, save)


def fix_speaker_names(text: str) -> str:
    return _book.fix(text)
