"""A beszélő nevének színe a feliratsorban: a játék minden szereplő nevét a saját színével írja.

    colors = sample_speaker_colors(band, area, "Lucia: Stop talking.", line_box)  # (("Lucia", (126, 207, 214)),)
    palette.learn(colors); palette.for_text(text)                                 # a megtanult színek

Mérés (a sáv képpontjain, a sor magasságában, a sáv teljes szélességében, mert a színes név a sor
dobozától balra is eshet):
1. betűpixel: világos (a legnagyobb csatorna legalább V_MIN), és két SZEMKÖZTI oldalán (bal–jobb vagy
   fent–lent) néhány pixelen belül ott a fekete körvonal; a háttér a körvonal mellett csak egy oldalról
   sötét, ezért kiesik;
2. a betűpixelek összefüggő foltjai a betűk; mindegyiknek csak a világosabb felét (a vonal közepét)
   nézzük, és ennek telítettsége (max − min csatorna) dönti el, színes-e (név) vagy fehér (szöveg);
3. az egymást követő színes betűk futamai a nevek (a sormagasságnál nagyobb rés kettévágja őket),
   a k-adik futam a szövegben a k-adik beszélő; ha a darabszám nem egyezik, nem mérünk semmit;
4. a név színe a magpixelek mediánja, csatornánként a fehér szöveg mért fehérjéhez igazítva (a játék
   fehérje ~235, sötétített jelenetben ~180): így a név színe a jelenet fényességétől független.

A sample_speaker_colors tiszta függvény; a SpeakerPalette csak egy szálról használható.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .translator import _LANGUAGE_NOTE, split_speaker, split_speakers

if TYPE_CHECKING:  # az ocr modul ezt a modult használja: futáskor nem importálhatjuk vissza
    from .ocr import Box

__all__ = ["RGB", "SpeakerColors", "SpeakerPalette", "sample_speaker_colors", "speaker_names"]

type RGB = tuple[int, int, int]  # R, G, B sorrendben (nem BGR)
type SpeakerColors = tuple[tuple[str, RGB], ...]  # (név, szín) párok, a név úgy, ahogy a kettőspont előtt áll

BGR_CHANNELS = 3  # a BGRA kép alfa csatornáját nem nézzük
MAX_LEVEL = 255
V_MIN = 120  # betűpixel: a legnagyobb csatornája legalább ennyi
EDGE_MIN = 80  # a körvonal ennyivel sötétebb a betűnél
OUTLINE_REACH = 4  # ... és ennyi pixelen belül van mindkét szemközti oldalon
ROW_PAD = OUTLINE_REACH  # a sor doboza fölött és alatt ennyi pixelt is nézünk (a körvonal miatt)
# Betű csak a sormagassághoz mért elég nagy folt: a betűk közé és a betűk belsejébe (a, o, e) szorult
# háttérfolt kicsi és alacsony; a kettőspont pöttye is kiesik, de az nem kell
MIN_LETTER_HEIGHT_SHARE = 0.4  # a folt magassága legalább ennyiszerese a sor magasságának
MIN_LETTER_AREA_SHARE = 0.05  # a folt területe legalább ennyiszerese a sormagasság négyzetének
CHROMA_NAME = 35  # a színes (név) betű magjának telítettsége legalább ennyi
CHROMA_WHITE = 25  # a fehér (szöveg) betűé legfeljebb ennyi; a kettő közötti betű bizonytalan
MIN_NAME_LETTERS = 2
MIN_NAME_CORE_PX = 30
PALETTE_HISTORY = 5  # a paletta nevenként ennyi utolsó mérés mediánját adja: egy rossz mérés nem fordítja át
# A megtanult szín csak ennél nagyobb (csatornánkénti) eltérésre változik: a tömörítés ±1-2-es zaja
# különben minden olvasásnál új színt, így új kiírást és fölösleges újrarajzolást adna
PALETTE_DEADBAND = 6

_Letter = tuple[int, int, np.ndarray, float]  # (bal szél, jobb szél, magpixelek BGR-ben, telítettség)


def speaker_names(text: str) -> list[str]:
    """A szöveg beszélőnevei sorrendben (a nyelvi megjegyzés, pl. „In Spanish”, nem beszélő)."""
    names = (split_speaker(part)[0] for part in split_speakers(text))
    return [name for name in names if name and not _LANGUAGE_NOTE.match(name)]


def _one_sided_kernel(after: bool) -> np.ndarray:
    """Vízszintes erode-mag: a pixel utáni (vagy előtti) OUTLINE_REACH pixel minimuma, a pixel nélkül."""
    kernel = np.zeros((1, 2 * OUTLINE_REACH + 1), np.uint8)
    if after:
        kernel[0, OUTLINE_REACH + 1:] = 1
    else:
        kernel[0, :OUTLINE_REACH] = 1
    return kernel


_RIGHT, _LEFT = _one_sided_kernel(after=True), _one_sided_kernel(after=False)
_BELOW, _ABOVE = _RIGHT.T.copy(), _LEFT.T.copy()


def _glyph_mask(bgr: np.ndarray) -> np.ndarray:
    """Világos pixel, amelyet két szemközti oldalról közelről fekete körvonal fog közre."""
    level = np.maximum(np.maximum(bgr[:, :, 0], bgr[:, :, 1]), bgr[:, :, 2])

    def dark(kernel: np.ndarray) -> np.ndarray:
        # A kép szélén túl nincs körvonal: az erode alapértelmezett szegélye a legnagyobb érték
        return cv2.subtract(level, cv2.erode(level, kernel)) >= EDGE_MIN

    across = dark(_LEFT) & dark(_RIGHT)
    along = dark(_ABOVE) & dark(_BELOW)
    return (level >= V_MIN) & (across | along)


def _median(values: np.ndarray) -> int:
    """A felső medián: betűnként sokszor kell, a np.median ehhez lassú."""
    middle = len(values) // 2
    return int(np.partition(values, middle)[middle])


def _letters(bgr: np.ndarray, row_height: int) -> list[_Letter]:
    """A betűfoltok balról jobbra: szélük, a világosabb felük pixelei és azok telítettsége."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(_glyph_mask(bgr).astype(np.uint8), connectivity=8)
    is_letter = ((stats[:, cv2.CC_STAT_AREA] >= MIN_LETTER_AREA_SHARE * row_height * row_height)
                 & (stats[:, cv2.CC_STAT_HEIGHT] >= MIN_LETTER_HEIGHT_SHARE * row_height))
    is_letter[0] = False  # a 0. címke a háttér
    flat_labels = labels.ravel()
    inside = np.flatnonzero(is_letter[flat_labels])
    if not inside.size:
        return []
    # Egyetlen rendezéssel csoportosítjuk a pixeleket foltonként (foltonkénti maszk helyett)
    order = inside[np.argsort(flat_labels[inside], kind="stable")]
    sorted_labels = flat_labels[order]
    kept = np.flatnonzero(is_letter)
    bounds = np.searchsorted(sorted_labels, np.append(kept, count))
    pixels = bgr.reshape(-1, BGR_CHANNELS)[order].astype(np.int16)
    letters: list[_Letter] = []
    for index, label in enumerate(kept):
        blob = pixels[bounds[index]:bounds[index + 1]]
        brightness = blob.max(axis=1)
        core = blob[brightness >= _median(brightness)]
        chroma = float(_median(core.max(axis=1) - core.min(axis=1)))
        left = int(stats[label, cv2.CC_STAT_LEFT])
        letters.append((left, left + int(stats[label, cv2.CC_STAT_WIDTH]), core, chroma))
    return sorted(letters, key=lambda letter: letter[0])


def _coloured_runs(letters: list[_Letter], max_gap: int) -> list[list[_Letter]]:
    """A színes betűk futamai, amelyeket fehér szöveg követ; a fehér szöveg vagy nagy rés vágja el őket.

    A név két szava közti résben a körvonalak közé szorult fehér háttérfolt egymagában nem vágja ketté
    a nevet: az a fehér betű, amelyet legalább két színes követ, nem zárja le a futamot.
    """
    runs: list[list[_Letter]] = []
    current: list[_Letter] = []
    is_coloured = [letter[3] >= CHROMA_NAME for letter in letters] + [False, False]
    for index, letter in enumerate(letters):
        if is_coloured[index]:
            if current and letter[0] - current[-1][1] > max_gap:
                current = []  # a nagy rés utáni színes betű új futamot kezd; az előzőt nem követte szöveg
            current.append(letter)
        elif letter[3] < CHROMA_WHITE and current and not (is_coloured[index + 1] and is_coloured[index + 2]):
            runs.append(current)
            current = []
    # A betűk közti résben a körvonalak közé szorult színes háttérfolt kicsi: a túl kevés pixelű futam zaj
    return [run for run in runs if len(run) >= MIN_NAME_LETTERS and _core_size(run) >= MIN_NAME_CORE_PX]


def _core_size(run: list[_Letter]) -> int:
    return sum(len(letter[2]) for letter in run)


def _rgb(bgr: np.ndarray) -> RGB:
    blue, green, red = (min(MAX_LEVEL, max(0, round(float(channel)))) for channel in bgr)
    return red, green, blue


def sample_speaker_colors(band: np.ndarray, area: Box, text: str, *boxes: Box) -> SpeakerColors:
    """A szöveg beszélőneveinek színe, ha mindegyiket biztosan meg tudjuk mérni, különben ().

    band: a feliratsáv BGR (vagy BGRA) uint8 képe; area: a sáv helye a képen; text: a felolvasott szöveg
    (0, 1 vagy több beszélővel); boxes: a szöveg képernyősorainak dobozai fentről lefelé, a kép
    koordinátáiban (két sorba tört megszólalásnál a második sorban is állhat név).
    """
    names = speaker_names(text)
    if not names or not boxes or band.ndim != BGR_CHANNELS or band.shape[2] < BGR_CHANNELS:
        return ()
    runs: list[list[_Letter]] = []
    white_letters: list[np.ndarray] = []
    for box in boxes:
        top = max(0, box.top - area.top - ROW_PAD)
        bottom = min(band.shape[0], box.bottom - area.top + ROW_PAD)
        if bottom <= top:
            return ()
        row_height = max(box.height, 1)
        letters = _letters(np.ascontiguousarray(band[top:bottom, :, :BGR_CHANNELS]), row_height)
        runs.extend(_coloured_runs(letters, row_height))
        white_letters.extend(letter[2] for letter in letters if letter[3] < CHROMA_WHITE)
    if len(runs) != len(names) or not white_letters:
        return ()
    white = np.maximum(np.median(np.concatenate(white_letters), axis=0), 1.0)
    samples: list[tuple[str, RGB]] = []
    for name, run in zip(names, runs, strict=True):
        core = np.concatenate([letter[2] for letter in run])
        samples.append((name, _rgb(np.median(core, axis=0) * MAX_LEVEL / white)))
    return tuple(samples)


def _key(name: str) -> str:
    return name.strip().casefold()


class SpeakerPalette:
    """A megtanult névszínek (név → szín); nevenként az utolsó mérések mediánja. Csak egy szálról."""

    def __init__(self, history: int = PALETTE_HISTORY) -> None:
        self._history = history
        self._samples: dict[str, deque[RGB]] = {}
        self._colors: dict[str, RGB] = {}

    def learn(self, samples: SpeakerColors) -> None:
        for name, color in samples:
            key = _key(name)
            if not key:
                continue
            recent = self._samples.setdefault(key, deque(maxlen=self._history))
            recent.append(color)
            red, green, blue = (int(np.median([sample[channel] for sample in recent])) for channel in range(3))
            known = self._colors.get(key)
            if known is None or max(abs(a - b) for a, b in zip(known, (red, green, blue))) > PALETTE_DEADBAND:
                self._colors[key] = (red, green, blue)

    def color_of(self, name: str) -> RGB | None:
        return self._colors.get(_key(name))

    def for_text(self, text: str) -> SpeakerColors:
        """A szöveg beszélői, akiknek már ismerjük a színét: a név a szövegbeli írásmóddal, rendezve."""
        found = {(name, color) for name in speaker_names(text) if (color := self.color_of(name)) is not None}
        return tuple(sorted(found))
