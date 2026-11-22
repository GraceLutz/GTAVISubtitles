"""A kiírt magyar felirat szövegfájlba, hogy az OBS saját szövegforrásként tegye a streamre.

OBS-ben: Szöveg (GDI+) forrás, „Olvasás fájlból”, és ez a fájl. A betűtípust, a színt és a helyet ott lehet
beállítani. A fájlban mindig az éppen látható fordítás áll (beszélőnévvel, soronként); ha nincs felirat, üres.
A cserét egy lépésben végezzük (ideiglenes fájl, majd csere), így az OBS soha nem lát félig írt szöveget.
Ha az OBS épp olvassa a fájlt, és a Windows nem engedi a cserét, a következő hívás újrapróbálja.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .layer_plan import Shown
from .streamer_filter import censor

FILE_NAME = "obs_felirat.txt"
ENCODING = "utf-8"

logger = logging.getLogger(__name__)


def shown_text(shown: Shown | None, *, streamer_mode: bool = False) -> str:
    """A fájlba kerülő szöveg: a lefordított sorok (a csak eltakart angol sorok nem); üres, ha nincs felirat."""
    if shown is None:
        return ""
    text = "\n".join(line.text for line in shown.lines) if shown.lines else shown.text
    return censor(text) if streamer_mode else text


class ObsTextFile:
    """Az éppen kiírt felirat egy szövegfájlban; csak akkor ír, ha a szöveg változott."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._written: str | None = None  # a fájlban most álló szöveg (None: még nem írtunk)
        self._warned = False

    @property
    def path(self) -> Path:
        return self._path

    def update(self, text: str) -> None:
        if text == self._written:
            return
        temporary = self._path.with_name(self._path.name + ".new")
        try:
            temporary.write_text(text, encoding=ENCODING)
            os.replace(temporary, self._path)
        except OSError as error:
            if not self._warned:  # minden képkockánál újrapróbáljuk: csak egyszer írjuk a naplóba
                logger.warning("Az OBS-szövegfájl nem írható (%s): %s", self._path, error)
                self._warned = True
            return
        self._written, self._warned = text, False

    def clear(self) -> None:
        """Kilépéskor vagy kikapcsoláskor: ne maradjon kint az utolsó felirat a streamen (fájlt nem hoz létre)."""
        if self._written is not None or self._path.exists():
            self.update("")
