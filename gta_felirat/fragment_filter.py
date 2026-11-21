"""Az OCR-darabok közül csak a valódi feliratot tartjuk meg.

A feliratsávban gyakran más szöveg is látszik: az autó rendszáma, a minitérkép „N” betűje,
táblák. Ezek a fordításba keveredve értelmetlen szöveget adnak („N EEX36H Fuck you!”), és a
takarást is óriásira nyújtják. A GTA-felirat felismerhető:
- fehér betűk (a darab pixeleinek jó része hófehér; a rendszám sötét betű világos alapon),
- a sorai egy oszlopban, középre igazítva, szorosan egymás alatt állnak,
- a kép közepén áll; a minitérkép fölötti értesítés („Moses Bucked You $825”) oldalt van
  (ha a sáv egészét nézzük, whole_band_text, a máshol álló szöveg is számít, de csak ha nincs
  középen felirat, és nem a sáv széle vágja el),
- a párbeszéd sorai fölött végződik a küldetésutasítás („Find Dunk”, „Follow Mia”): az a kép alján,
  külön sorban áll, és nem fordítjuk,
- a kijelző feliratai (csupa nagybetű: „BASEBALL BAT”, vagy csak szám: „100”) nem párbeszéd,
- a megvilágított rendszám világos, ezért a színe alapján feliratnak tűnne: a nagybetű–szám
  kódot („BM472H”) a szövege alapján dobjuk el.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .ocr import Box

WHITE_LEVEL = 200  # mindhárom színcsatorna legalább ennyi: hófehér pixel
MIN_WHITE_FRACTION = 0.04  # a valódi feliratdarabokban 6–18%, a rendszámban ~0%
# A játékbeli tévéműsor felirata halványabb, szürkésfehér (~183), elsötétített kép fölött
DIM_WHITE_LEVEL = 150
DIM_BACKGROUND_LEVEL = 90  # a halvány betű csak ilyen sötét háttéren számít feliratnak
MAX_GREY_SPREAD = 40  # a szürkésfehér betű színcsatornái közel egyformák
SAME_LINE_TOLERANCE_PX = 12
MAX_WORD_GAP_RATIO = 2.0  # egy soron belül ennél nagyobb rés (sormagasság-arány): külön szöveg
MAX_LINE_GAP_RATIO = 0.75  # két feliratsor között legfeljebb ekkora rés (sormagasság-arány)
CENTER_TOLERANCE_RATIO = 0.06  # a párbeszéd közepe legfeljebb ennyire térhet el a kép közepétől (képszélesség-arány)
# A párbeszéd legalsó sorának közepe a képmagasság ~90%-ánál van, a küldetésutasításé ~96%-nál
DIALOGUE_BOTTOM_RATIO = 0.93
MIN_LABEL_LETTERS = 4  # ennél rövidebb csupa nagybetűs darab párbeszéd is lehet („OK.”, „I”)


@dataclass(frozen=True)
class Fragment:
    text: str
    box: Box  # a teljes képen
    whiteness: float  # a hófehér pixelek aránya (white_fraction)


def white_fraction(crop: np.ndarray, box: Box, area: Box) -> float:
    """A darab feliratszínű pixeleinek aránya; crop: a sáv képe, area: a sáv helye a teljes képen.

    Hófehér betűk, vagy sötét háttéren szürkésfehérek (a tévéműsor halványabb felirata).
    A rendszám (sötét betű világos alapon) egyik szerint sem felirat.
    """
    top, left = max(box.top - area.top, 0), max(box.left - area.left, 0)
    patch = crop[top:box.bottom - area.top, left:box.right - area.left]
    if patch.size == 0:
        return 0.0
    darkest, brightest = patch.min(axis=2), patch.max(axis=2)
    white = float((darkest >= WHITE_LEVEL).mean())
    if white >= MIN_WHITE_FRACTION or np.median(brightest) > DIM_BACKGROUND_LEVEL:
        return white
    grey = (darkest >= DIM_WHITE_LEVEL) & (brightest.astype(np.int16) - darkest <= MAX_GREY_SPREAD)
    return float(grey.mean())


def _union(boxes: list[Box]) -> Box:
    return replace(boxes[0], left=min(b.left for b in boxes), top=min(b.top for b in boxes),
                   right=max(b.right for b in boxes), bottom=max(b.bottom for b in boxes))


def _segments(fragments: list[Fragment]) -> list[list[Fragment]]:
    """Soronként, és soron belül a nagy réseknél szétvágva: egy-egy összefüggő szövegdarab."""
    ordered = sorted(fragments, key=lambda f: (f.box.top, f.box.left))
    rows: list[list[Fragment]] = []
    for fragment in ordered:
        if rows and abs(rows[-1][0].box.top - fragment.box.top) <= SAME_LINE_TOLERANCE_PX:
            rows[-1].append(fragment)
        else:
            rows.append([fragment])
    segments = []
    for row in rows:
        row.sort(key=lambda f: f.box.left)
        current = [row[0]]
        for previous, fragment in zip(row, row[1:]):
            gap = fragment.box.left - previous.box.right
            if gap > MAX_WORD_GAP_RATIO * max(previous.box.height, fragment.box.height):
                segments.append(current)
                current = []
            current.append(fragment)
        segments.append(current)
    return segments


def row_pieces(boxes: list[Box]) -> list[Box]:
    """A kereső dobozai összefüggő sordarabokká vonva: egy sordarabot egyszer elég beolvasni."""
    pieces = _segments([Fragment("", box, 1.0) for box in boxes])
    return [_union([fragment.box for fragment in piece]) for piece in pieces]


def _letters(segment: list[Fragment]) -> int:
    return sum(ch.isalpha() for fragment in segment for ch in fragment.text)


def _is_aligned(box: Box, main: Box) -> bool:
    # A feliratsorok középre igazítottak: a közepük közel esik egymáshoz
    shift = abs((box.left + box.right) / 2 - (main.left + main.right) / 2)
    return shift <= max(box.width, main.width) / 2 + main.height


@dataclass(frozen=True)
class FrameCenter:
    x: float  # a teljes kép vízszintes közepe
    frame_width: float
    dialogue_bottom: float = float("inf")  # ennél lejjebb (nagyobb y) már a küldetésutasítás sora áll
    # A figyelt sáv két széle a teljes képen; végtelen: nem tudjuk, semmi sem ér a széléhez
    band_left: float = float("-inf")
    band_right: float = float("inf")
    edge_margin: float = 0.0  # képpont; a hívó adja meg (row_finder.band_edge_margin, a kereső szabálya)
    anywhere: bool = False  # True: középen álló felirat híján a sáv máshol álló szövege is felirat lehet

    def is_near(self, box: Box) -> bool:
        return abs((box.left + box.right) / 2 - self.x) <= CENTER_TOLERANCE_RATIO * self.frame_width

    def is_dialogue_row(self, box: Box) -> bool:
        return (box.top + box.bottom) / 2 <= self.dialogue_bottom

    def touches_band_edge(self, box: Box) -> bool:
        """A sáv széle vágja-e el a szöveget (a széléhez edge_margin-nél közelebb ér)."""
        return reaches_band_edge((box.left, box.right), (self.band_left, self.band_right), self.edge_margin)


def reaches_band_edge(columns: tuple[float, float], band: tuple[float, float], margin: float) -> bool:
    """A sáv széle vágja-e el a szöveget: a (bal, jobb) oszlopai a sáv (bal, jobb) széléhez margin-nál közelebb
    érnek. A sorkereső (row_geometry) és a pipeline ugyanígy, ugyanarra a (kitömött) sordobozra méri."""
    return columns[0] - band[0] < margin or band[1] - columns[1] < margin


def is_hud_label(text: str) -> bool:
    """A kijelző vagy a játékvilág felirata, nem párbeszéd.

    Csak szám („100”), csupa nagybetű („BASEBALL BAT”), vagy nagybetű–szám kód (rendszám: „BM472H”).
    """
    letters = [ch for ch in text if ch.isalpha()]
    has_digit = any(ch.isdigit() for ch in text)
    if not letters:
        return has_digit
    if not all(ch.isupper() for ch in letters):
        return False
    return has_digit or len(letters) >= MIN_LABEL_LETTERS


def select_subtitle(fragments: list[Fragment], center: FrameCenter | None = None) -> list[Fragment]:
    """A felirat darabjai: fehérek, és a leghosszabb szövegdarabbal egy tömbben állnak.

    center: ha meg van adva, csak a kép közepén álló szöveg számít párbeszédnek; ha center.anywhere,
    középen álló szöveg híján a sáv máshol álló, a sáv széléig nem érő szövege is.
    """
    white = [f for f in fragments if f.whiteness >= MIN_WHITE_FRACTION and not is_hud_label(f.text)
             and (center is None or center.is_dialogue_row(f.box))]
    if not white:
        return []
    segments = _segments(white)
    boxes = {id(s): _union([f.box for f in s]) for s in segments}
    central = [s for s in segments if center is None or center.is_near(boxes[id(s)])]
    if not central and center is not None and center.anywhere:
        # A középső felirat mindig elsőbbséget kap; csak nélküle keresünk máshol, és a sáv széle által
        # elvágott szöveg (csonka szavak) akkor sem számít
        segments = [s for s in segments if not center.touches_band_edge(boxes[id(s)])]
        central = segments
    if not central:
        return []
    main = max(central, key=_letters)
    main_box = boxes[id(main)]
    candidates = [(s, boxes[id(s)]) for s in segments if s is not main]
    block = [main_box]
    kept = list(main)
    grew = True
    while grew:  # fölfelé és lefelé is sorról sorra bővítjük a tömböt
        grew = False
        top, bottom = min(b.top for b in block), max(b.bottom for b in block)
        for segment, box in list(candidates):
            gap = max(top - box.bottom, box.top - bottom)
            if gap <= MAX_LINE_GAP_RATIO * main_box.height and _is_aligned(box, main_box):
                block.append(box)
                kept.extend(segment)
                candidates.remove((segment, box))
                grew = True
    return kept
