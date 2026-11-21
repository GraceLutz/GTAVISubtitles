"""Streamer mód: a sértő szavak (rasszista és homofób gyalázkodás) kicsillagozva jelennek meg.

Csak a kirajzolt szöveget érinti: a fordítás és a mentett fordítások változatlanok, így a mód bármikor
ki-be kapcsolható. A szó első betűje marad, a többi csillag („n****”), a ragozott alak is („niggám”).
"""

import re

MASK = "*"

# Szótövek: a szó elejére illeszkednek, a toldalék (angol többes szám, magyar rag) is a szóhoz tartozik.
# Csak olyan tő, ami köznapi szó elején nem fordul elő (a „Niger” egy g-vel, a „buzgó” más tő).
_STEMS = (
    r"nigg",  # nigger, nigga, niggas, niggaz, niggám, niggerek
    r"néger",  # a spanyol/portugál „negro” (fekete) nem: oda is fordítunk
    r"fagg",  # faggot
    r"buzi",
    r"buzeráns",
    r"tranny",
    r"trannies",
    r"cigó",
)
# Egész szavak: rövidek, toldalékkal együtt már köznapi szó lehetne („spicy”), ezért csak pontosan, -s-sel
_WORDS = (
    r"fags?",
    r"spics?",
    r"chinks?",
    r"kikes?",
    r"wetbacks?",
    r"beaners?",
    r"retards?",
    r"retarded",
)

_PATTERN = re.compile(
    r"\b(?:(?:" + "|".join(_STEMS) + r")\w*|(?:" + "|".join(_WORDS) + r"))\b",
    re.IGNORECASE,
)


def _masked(match: re.Match[str]) -> str:
    word = match.group(0)
    return word[0] + MASK * (len(word) - 1)


def censor(text: str) -> str:
    """A szöveg a sértő szavak kicsillagozásával (első betű marad)."""
    return _PATTERN.sub(_masked, text)
