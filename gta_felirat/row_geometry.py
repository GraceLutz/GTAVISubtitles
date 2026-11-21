"""Feliratsorok kimérése a betűmag-maszkból: hol van sor, mekkora, és biztosan felirat-e.

A maszkot a row_detector készíti (255 a fehér, fekete körvonalú betűk magján). A GTA minden
feliratsort a kép közepére igazít, erre épül a mérés:
1. a kép közepe körüli keskeny ablak sorainak festettségéből sorjelölteket keresünk (seed_parts),
2. a jelöltből a közepétől kifelé, a szóközöket áthidalva megkeressük a sor két végét, majd a teljes
   szélességén a tetejét és az alját (a fel- és lenyúló szárakkal együtt),
3. a méret, a középre igazítás, a festettség és a vonalvastagság dönti el, hogy feliratsor-e, és
   olyan biztosan az-e, hogy olvasás nélkül, első látásra eltakarható (Row.sure).

A sáv többi szövege (find_free_rows, csak ha a teljes sávot nézzük): a játék máshova is írhat szöveget
(oldalra, más sorba), ezért a középső sorokon kívül a sáv minden festett csoportja külön sorjelölt, és
a saját közepéhez mérjük; nincs középre igazítási kapu, tükrös vágás és névkiterjesztés. Az ilyen sor
sosem biztos (olvasás előtt nem takarjuk), és a sáv széle által elvágott szöveget eldobjuk: azt úgysem
lehetne helyesen elolvasni.

A küszöbök a row_detector.Params mért értékei; a sorok koordinátái a sáv saját koordinátái.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .fragment_filter import reaches_band_edge

if TYPE_CHECKING:
    from .row_detector import Measures

MASK_ON = 255  # a maszk bekapcsolt pixelének értéke

Span = tuple[int, int]  # [kezdet, vég): sorok vagy oszlopok a maszkon


@dataclass(frozen=True)
class Row:
    """Egy feliratsor a sáv koordinátáiban."""

    left: int  # a kitömött doboz (a neurális kereső dobozaihoz hasonló)
    top: int
    right: int
    bottom: int
    ink_left: int  # a betűmagok szoros kiterjedése
    ink_top: int
    ink_right: int
    ink_bottom: int
    pixels: int  # magpixelek a szoros kiterjedésen belül
    strokes: float  # a sor közepén húzott vízszintes vonal átlagosan ennyi vonást keresztez
    run: float  # a magpixelek átlagos vízszintes futáshossza (vonalvastagság), 1920 széles képre átszámolva
    center_offset: float  # a festett rész közepe mínusz a kép közepe (px)
    trimmed: bool  # az egyik oldalára ragadt foltot a képközépre vett tükrözéssel levágtuk
    sure: bool  # a szigorú feltételeknek is megfelel
    dim: bool  # a halvány (tévéműsor-felirat) menet találta

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(frozen=True)
class _Ink:
    """Egy sorjelölt festett kiterjedése a maszkon."""

    lines: Span
    columns: Span
    pixels: int
    trimmed: bool
    named: bool = False  # a színes beszélőnévre is kiterjedt


def seed_parts(window_core: np.ndarray, measures: Measures) -> list[Span]:
    """Sorjelöltek fentről lefelé: a középső ablak maszkjának (window_core) elég magas festett sorfutamai.

    A túl magas futamot (folt ragasztott össze két sort) kettévágjuk; azt a jelöltet, amelyből csak
    a párbeszédsorok alsó határa alatti sor lehetne, már itt eldobjuk.
    """
    params = measures.params
    profile = cv2.reduce(window_core, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S).ravel() // MASK_ON
    min_height = measures.px(params.seed_min_height)
    parts: list[Span] = []
    for top, bottom in _runs(profile >= params.seed_min_pixels, measures.px(params.seed_max_hole)):
        if bottom - top >= min_height:
            parts.extend(_split_tall(profile, (top, bottom), measures))
    if measures.dialogue_bottom is None:
        return parts
    return [part for part in parts if _may_hold_dialogue(part, window_core.shape[0], measures)]


def find_rows(core: np.ndarray, parts: list[Span], measures: Measures, dim: bool,
              named: np.ndarray | None = None) -> tuple[Row, ...]:
    """A jelöltekből kimért sorok fentről lefelé; core: a sáv betűmag-maszkja (a jelöltek környékén pontos).

    named: a core a színes (beszélőnév) betűmagokkal kiegészítve; a sor oldalra kiterjedhet rájuk.
    """
    rows: list[Row] = []
    for part in parts:
        ink = _ink_of(core, part, measures, named)
        judged = named if ink is not None and ink.named else core
        row = _judge(judged, ink, measures, dim) if ink is not None else None
        if row is not None and _is_dialogue_row(row, measures.dialogue_bottom):
            rows.append(row)
    return _drop_overlaps(rows)


def find_free_rows(core: np.ndarray, measures: Measures) -> tuple[Row, ...]:
    """A sáv szabad (nem középre igazított) sorai fentről, balról jobbra; core: a teljes sáv betűmag-maszkja,
    a középső sorok helye már kitörölve.

    Egy képsoron több sor is lehet: minden festett oszlopcsoport külön jelölt. Az egész sávra húzott
    sorprofil a különböző magasságban álló szövegeket összemosná, ezért a jelölteket oszlopcsoportonként keressük.
    """
    max_gap = measures.px(measures.params.max_gap)
    rows: list[Row] = []
    for group in _groups(_column_ink(core, (0, core.shape[0])), max_gap):
        for part in seed_parts(core[:, group[0]:group[1]], measures):
            for ink in _free_inks(core, part, group, measures):
                row = _judge(core, ink, measures, dim=False, free=True)
                if (row is not None and _is_dialogue_row(row, measures.dialogue_bottom)
                        and not _is_cut(row, core, measures)):
                    rows.append(row)
    return _drop_overlaps(rows, by_columns=True)


def _is_cut(row: Row, core: np.ndarray, measures: Measures) -> bool:
    """A sáv széle elvágta-e a szabad sort: a kitömött doboza a sáv széléhez edge_margin-nál közelebb ér.

    Ugyanaz a doboz és ugyanaz a mérték, amellyel a pipeline is mér (row_finder.band_edge_margin): amit
    a kereső megtart, azt a pipeline sem dobja el a sáv széle miatt, az olvasás nem megy kárba."""
    return reaches_band_edge((row.left, row.right), (0, core.shape[1]), measures.params.edge_margin * measures.scale)


def _runs(flags: np.ndarray, max_hole: int) -> list[Span]:
    """Az igaz értékek futamai; a legfeljebb max_hole hosszú lyukakat áthidaljuk."""
    index = np.flatnonzero(flags)
    if index.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(index) > max_hole + 1)
    starts = np.concatenate(([index[0]], index[breaks + 1]))
    ends = np.concatenate((index[breaks], [index[-1]])) + 1
    return list(zip(starts.tolist(), ends.tolist()))


def _split_tall(profile: np.ndarray, part: Span, measures: Measures) -> list[Span]:
    """A sornál magasabb jelöltet a középső része legüresebb soránál kettévágjuk (ha kell, többször)."""
    top, bottom = part
    if bottom - top <= measures.px(measures.params.row_max_height):
        return [part]
    min_height = measures.px(measures.params.seed_min_height)
    first, last = top + min_height, bottom - min_height
    if last <= first:
        return [part]
    cut = first + int(np.argmin(profile[first:last]))
    return _split_tall(profile, (top, cut), measures) + _split_tall(profile, (cut + 1, bottom), measures)


def _may_hold_dialogue(part: Span, band_height: int, measures: Measures) -> bool:
    """Hamis, ha a jelöltből kimért sor közepe biztosan a párbeszédsorok alsó határa alá esne.

    A kimért sor a jelölt minden sorát tartalmazza, és legfeljebb row_max_height magasra nő: a doboza
    tehát nem kezdődhet és nem végződhet az itt számoltnál feljebb.
    """
    params, (top, bottom) = measures.params, part
    highest_top = max(0, min(top, bottom - measures.px(params.row_max_height)) - measures.px(params.pad_top))
    highest_bottom = min(band_height, bottom + measures.px(params.pad_bottom))
    return (highest_top + highest_bottom) / 2 <= measures.dialogue_bottom


def _is_dialogue_row(row: Row, dialogue_bottom: float | None) -> bool:
    # Ugyanaz a szabály, mint a fragment_filter.FrameCenter.is_dialogue_row: a doboz közepe dönt
    return dialogue_bottom is None or (row.top + row.bottom) / 2 <= dialogue_bottom


def _column_ink(core: np.ndarray, lines: Span) -> np.ndarray:
    return cv2.reduce(core[lines[0]:lines[1]], 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S).ravel() // MASK_ON


def _central_group(column_ink: np.ndarray, window: Span, max_gap: int) -> Span | None:
    """A festett oszlopok csoportja, amelynek a legtöbb magpixele esik az ablakba (a kis rések zárva)."""
    best, best_ink = None, 0
    for left, right in _runs(column_ink > 0, max_gap):
        start, end = max(left, window[0]), min(right, window[1])
        if end > start:
            ink = int(column_ink[start:end].sum())
            if ink > best_ink:
                best, best_ink = (left, right), ink
    return best


def _groups(column_ink: np.ndarray, max_gap: int) -> list[Span]:
    """Minden festett oszlopcsoport balról jobbra (a legfeljebb max_gap széles réseket áthidalva)."""
    return _runs(column_ink > 0, max_gap)


def _full_height(core: np.ndarray, part: Span, extent: Span, max_height: int) -> Span:
    """A sor teteje és alja a teljes szélességén: a középső ablakon kívüli fel- és lenyúló szárakkal együtt."""
    top, bottom = part
    first, last = max(0, top - max_height), min(core.shape[0], bottom + max_height)
    # A REDUCE_SUM ezen a tengelyen ~40× gyorsabb a REDUCE_MAX-nál (130 µs → 4 µs)
    inked = cv2.reduce(core[first:last, extent[0]:extent[1]], 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S).ravel() > 0
    while top > first and inked[top - 1 - first]:
        top -= 1
    while bottom < last and inked[bottom - first]:
        bottom += 1
    return (top, bottom) if bottom - top <= max_height else part


def _symmetric_trim(column_ink: np.ndarray, extent: Span, center: float, slack: int) -> Span | None:
    """A kiterjedés az egyik oldalára ragadt folt nélkül: a szöveg ott ér véget, ahol a másik végének tükörképe.

    Csak akkor vágunk, ha ott (slack pixelen belül) rés van; különben None.
    """
    left, right = extent
    half = min(center - left, right - center)
    if half <= 0:
        return None
    if right - center > center - left:  # a jobb oldal a hosszabb
        cut = int(round(center + half))
        start = max(left, cut - slack)
        gap = np.flatnonzero(column_ink[start:min(right, cut + slack + 1)] == 0)
        if gap.size == 0:
            return None
        inked = np.flatnonzero(column_ink[left:start + int(gap[0])])
        return (left, left + int(inked[-1]) + 1) if inked.size else None
    cut = int(round(center - half))
    start = max(left, cut - slack)
    gap = np.flatnonzero(column_ink[start:min(right, cut + slack + 1)] == 0)
    if gap.size == 0:
        return None
    first = start + int(gap[-1]) + 1
    inked = np.flatnonzero(column_ink[first:right])
    return (first + int(inked[0]), right) if inked.size else None


def _with_name(named: np.ndarray, white_ink: np.ndarray, lines: Span, extent: Span,
               measures: Measures) -> Span | None:
    """A fehér kiterjedés a hozzá tapadó színes névvel, ha a sor így közelebb kerül a kép közepéhez.

    A valódi névvel a teljes sor középre igazított; a mellé ragadt színes háttérfolt épp elhúzza a középtől.
    A név után mindig fehér szöveg jön: a sor jobb vége fehér betű (a sor közepén álló név fölött átérünk
    a következő beszélő szövegéig), a jobbra tapadó színes folt (bőr, háttér) nem tartozik a sorhoz.
    white_ink: a fehér betűmag oszloponkénti mennyisége a sor képsoraiban.
    """
    params = measures.params
    wider = _central_group(_column_ink(named, lines), extent, measures.px(params.max_gap))
    if (wider is None or wider[0] > extent[0] or wider[1] < extent[1]
            or wider[1] - wider[0] < measures.px(params.named_min_width)):
        return None
    white_after = np.flatnonzero(white_ink[extent[1]:wider[1]])
    wider = (wider[0], extent[1] + int(white_after[-1]) + 1 if white_after.size else extent[1])
    if wider == extent:
        return None
    before = abs((extent[0] + extent[1]) / 2.0 - measures.center)
    after = abs((wider[0] + wider[1]) / 2.0 - measures.center)
    return wider if after + params.sure_offset * measures.scale < before else None


def _ink_of(core: np.ndarray, part: Span, measures: Measures, named: np.ndarray | None = None) -> _Ink | None:
    """A jelölt sorának festett kiterjedése; None, ha nincs benne középre eső szöveg, vagy túl kicsi."""
    params = measures.params
    max_gap = measures.px(params.max_gap)
    column_ink = _column_ink(core, part)
    extent = _central_group(column_ink, measures.window, max_gap)
    if extent is None:
        return None
    lines = _full_height(core, part, extent, measures.px(params.row_max_height))
    if lines != part:
        column_ink = _column_ink(core, lines)
        extent = _central_group(column_ink, extent, max_gap) or extent
    trimmed = False
    wider = _with_name(named, column_ink, lines, extent, measures) if named is not None else None
    if wider is not None:  # a színes név a sorhoz tartozik: a magpixeleket is vele együtt számoljuk
        column_ink, extent = _column_ink(named, lines), wider
    is_off_centre = abs((extent[0] + extent[1]) / 2.0 - measures.center) > params.sure_offset * measures.scale
    if params.trims_off_centre and is_off_centre:
        cut = _symmetric_trim(column_ink, extent, measures.center, measures.px(params.trim_slack))
        if cut is not None:
            extent, trimmed = cut, True
    if (lines[1] - lines[0] < measures.px(params.row_min_height)
            or extent[1] - extent[0] < measures.px(params.row_min_width)):
        return None
    return _Ink(lines, extent, int(column_ink[extent[0]:extent[1]].sum()), trimmed, wider is not None)


def _free_inks(core: np.ndarray, part: Span, within: Span, measures: Measures) -> list[_Ink]:
    """A jelölt képsorain a within oszlopaiba eső minden festett csoport, külön kimérve."""
    column_ink = _column_ink(core, part)
    inks = (_free_ink_of(core, part, group, measures)
            for group in _groups(column_ink, measures.px(measures.params.max_gap))
            if group[1] > within[0] and group[0] < within[1])
    return [ink for ink in inks if ink is not None]


def _free_ink_of(core: np.ndarray, part: Span, extent: Span, measures: Measures) -> _Ink | None:
    """Egy szabad csoport festett kiterjedése (mint az _ink_of, név és tükrös vágás nélkül); None, ha túl kicsi.

    A sáv széle által elvágott szöveget a kész sor dobozán szűrjük ki (_is_cut)."""
    params = measures.params
    lines = _full_height(core, part, extent, measures.px(params.row_max_height))
    column_ink = _column_ink(core, lines)
    if lines != part:
        extent = _central_group(column_ink, extent, measures.px(params.max_gap)) or extent
    if (lines[1] - lines[0] < measures.px(params.row_min_height)
            or extent[1] - extent[0] < measures.px(params.row_min_width)):
        return None
    return _Ink(lines, extent, int(column_ink[extent[0]:extent[1]].sum()), trimmed=False)


def _run_starts(zone: np.ndarray) -> int:
    """Hány vízszintes pixelfutam kezdődik a kivágásban (a bal szélén kezdődők nélkül)."""
    if zone.shape[1] < 2:
        return 0
    return cv2.countNonZero(cv2.subtract(zone[:, 1:], zone[:, :-1]))


def _strokes(core: np.ndarray, ink: _Ink) -> float:
    """Átlagosan hány vonást keresztez egy vízszintes vonal a sor középső felében."""
    top, bottom = ink.lines
    quarter = (bottom - top) // 4
    first = top + quarter
    last = max(first + 1, bottom - quarter)
    return _run_starts(core[first:last, ink.columns[0]:ink.columns[1]]) / float(last - first)


def _judge(core: np.ndarray, ink: _Ink, measures: Measures, dim: bool, free: bool = False) -> Row | None:
    """A kimért jelölt sorként; None, ha a helye, a festettsége vagy a vonalvastagsága nem feliraté.

    free: a sáv szabad sora; a helyét nem a kép közepéhez mérjük, és sosem biztos (olvasás előtt nem takarjuk).
    """
    params, scale, px = measures.params, measures.scale, measures.px
    (top, bottom), (left, right) = ink.lines, ink.columns
    height, width = bottom - top, right - left
    offset = (left + right) / 2.0 - measures.center
    if not free and abs(offset) > params.accept_offset * scale:
        return None
    fill = ink.pixels / float(height * width)
    strokes = _strokes(core, ink)
    density = strokes / width * scale
    run = ink.pixels / float(max(1, _run_starts(core[top:bottom, left:right]))) / scale
    if not params.min_fill <= fill <= params.max_fill or density < params.min_stroke_density or run > params.max_run:
        return None
    sure = (not free and abs(offset) <= params.sure_offset * scale and width >= px(params.sure_min_width)
            and px(params.sure_min_height) <= height <= px(params.sure_max_height)
            and params.sure_min_fill <= fill <= params.sure_max_fill
            and density >= params.sure_min_stroke_density and run <= params.sure_max_run)
    return Row(left=max(0, left - px(params.pad_left)), top=max(0, top - px(params.pad_top)),
               right=min(core.shape[1], right + px(params.pad_right)),
               bottom=min(core.shape[0], bottom + px(params.pad_bottom)),
               ink_left=left, ink_top=top, ink_right=right, ink_bottom=bottom, pixels=ink.pixels,
               strokes=strokes, run=run, center_offset=offset, trimmed=ink.trimmed, sure=sure, dim=dim)


def _drop_overlaps(rows: list[Row], by_columns: bool = False) -> tuple[Row, ...]:
    """Két ütköző sorból a több magpixelű marad; az eredmény fentről lefelé (azonos tetőnél balról jobbra) rendezett.

    A középső sorok nem osztozhatnak képsorokon. by_columns: a szabad sorok csak akkor ütköznek, ha az
    oszlopaik is fedik egymást, hiszen egy képsoron több szöveg is állhat egymás mellett.
    """
    kept: list[Row] = []
    for row in sorted(rows, key=lambda r: -r.pixels):
        if not any(overlaps(row, other, by_columns) for other in kept):
            kept.append(row)
    return tuple(sorted(kept, key=lambda r: (r.top, r.left)))


def overlaps(row: Row, other: Row, by_columns: bool = True) -> bool:
    """Fedik-e egymást a két sor festett részei (by_columns=False: elég, ha a képsoraik fedik egymást)."""
    if row.ink_bottom <= other.ink_top or row.ink_top >= other.ink_bottom:
        return False
    return not by_columns or (row.ink_left < other.ink_right and other.ink_left < row.ink_right)
