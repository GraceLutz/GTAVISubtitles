"""Klasszikus (nem neurális) feliratsor-kereső: csak OpenCV és numpy, sávonként ~1–3 ms.

    detection = detect(band, band_left, frame_width)   # Detection(rows, core, level)
    rows = detect_rows(band, band_left, frame_width)    # csak a sorok, fentről lefelé

A GTA felirata fehér betű vékony fekete körvonallal, és minden sora a kép közepére igazított:
1. betűmag-maszk: a pixel „fehér” (mindhárom csatornája eléri a szintet), vékony alakzat része
   (eltűnik, ha egy betűvonalnál nagyobb négyzettel „kinyitjuk”: a fehér póló, az ég, a napos talaj
   így a szélével együtt eltűnik), és néhány pixelen belül sötét (körvonal) pixel van mellette;
2. a maszkot előbb csak a kép közepe körüli keskeny ablakban számoljuk ki: sorjelölt nélkül a sáv
   többi részéhez hozzá sem nyúlunk, jelölttel pedig csak az általa elérhető sorokon készül el
   a teljes szélességű maszk;
3. a jelöltekből a row_geometry méri ki a sorokat, és dönt róluk (sor-e, és biztosan az-e).
A beszélő neve („Lucia:”) színes, fekete körvonalú betű a fehér sor elején vagy közepén: a fehér sort
oldalra kiterjesztjük a színes betűkre, ha a sor így a kép közepére kerül (a GTA a teljes sort, nevestül
igazítja középre). Sorjelöltet csak fehér szöveg adhat, a sor magassága is a fehér betűkből jön.
Előbb a fehér szöveget keressük; a tévéműsor halvány, szürke feliratát (a legvilágosabb pixelekhez
igazított szinttel, mint a frosted_cover._text_level) csak akkor, ha nincs biztos fehér sor.

A játék máshova is írhat szöveget (oldalra, más sorba, több helyre egyszerre). Params.whole_band esetén a
fenti, változatlan középső keresés után egy második, „szabad” menet a sáv többi részét is átnézi (csak fehér
betűre): ott minden festett csoport külön sor, a saját közepéhez mérve (row_geometry.find_free_rows). Ezek
a sorok sosem biztosak, tehát olvasás előtt nem takarjuk őket; a középső sorok bármelyik módban ugyanazok.

A küldetésutasítás („Find Dunk”) ugyanilyen betű, de a párbeszédsorok alatt, külön sorban áll; nem
takarjuk és nem fordítjuk, ezért a dialogue_bottom alatti közepű sort a kereső nem adja vissza.

A küszöbök 1920 képpont széles képen mért értékek, más szélességnél arányosan nőnek. Kisebb ablakban
(kb. 1600 képpont alatt) a kicsinyített videón a betűk halványabbak, és a kereső sorokat hagyhat ki.

A detect tiszta függvény: bármelyik szálról hívható, az eredménye nem módosítható. Az OpenCV saját
szálai ilyen kis műveleteknél csak a processzort égetik (mérve: ~6× processzoridő, és lassabb is):
a programnak induláskor érdemes cv2.setNumThreads(1)-et hívnia.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import cv2
import numpy as np

from .row_geometry import Row, Span, find_free_rows, find_rows, overlaps, seed_parts

__all__ = ["DEFAULT_PARAMS", "NOTHING", "REFERENCE_WIDTH", "Detection", "Params", "Row", "detect", "detect_rows",
           "text_extent", "to_frame"]

REFERENCE_WIDTH = 1920.0  # a mért pixelméretek ekkora képszélességre szólnak
BGR_CHANNELS = 3  # a negyedik (alfa) csatornát nem nézzük: a BGRA kép (GetDIBits, mss) is jó
MAX_LEVEL = 255
MIN_STROKE_PX = 3  # kis képen se legyen a „vékony alakzat” határa ennél kisebb: a betűvonal ott is 1–2 px
# A halvány szintet a sáv minden nyolcadik pixeléből becsüljük: vízszintesen 4, függőlegesen 2 a lépés
LEVEL_SAMPLE_STEP_X = 4
LEVEL_SAMPLE_STEP_Y = 2
KERNEL_CACHE_SIZE = 16  # képszélességenként két szerkesztőelem kell; az ablak mérete ritkán változik


@dataclass(frozen=True)
class Params:
    """A kereső mért küszöbei; a pixelben megadott méretek 1920 széles képre szólnak."""

    white_level: int = 200  # a betű magja: mindhárom csatorna legalább ennyi (frosted_cover.TEXT_LEVEL)
    # Második fehér menet, ha az elsőben nincs biztos sor: világos háttér előtt a tömörített kép fehér betűje
    # 180–195 (rendszám előtt mérve), a 200-as menet csak a sor töredékeit látja
    soft_white_level: int = 180
    outline_level: int = 150  # körvonal vagy sötét háttér: mindhárom csatorna legfeljebb ennyi (OUTLINE_LEVEL)
    outline_reach: float = 4.0  # ... és ilyen közel van a maghoz (OUTLINE_REACH_PX)
    stroke_max: float = 8.0  # az ennél nagyobb négyzetet tartalmazó világos folt nem betű (póló, ég, napos talaj)
    # A tévéműsor halvány, szürkésfehér felirata (~183, elsötétített képen): csak ha nincs biztos fehér sor
    dim_level: int = 150
    dim_margin: int = 25  # a halvány szint ennyivel van a legvilágosabb pixelek alatt
    dim_top_fraction: float = 0.005  # „legvilágosabb”: a sáv pixeleinek felső 0,5%-a (99,5. percentilis)
    dim_grey_spread: int = 40  # a halvány betű szürke: a színcsatornái legfeljebb ennyire térnek el
    dim_min_grey_share: float = 0.7  # a halvány sor magpixeleinek legalább ekkora része szürke
    dim_outline_level: int = 90  # a halvány betű csak igazán sötét pixel mellett számít
    # Sorkeresés
    center_half: float = 110.0  # a kép közepe körüli ablak félszélessége: a sorjelölteket ebben keressük
    seed_min_pixels: int = 2  # az ablak egy sora ennyi magpixeltől „festett”
    seed_max_hole: float = 2.0  # ennyi üres sor még belefér egy sorjelöltbe
    seed_min_height: float = 9.0  # a betűtípus x-magassága ~12 px
    max_gap: float = 18.0  # soron belül ekkora vízszintes rést zárunk (a legnagyobb mért szóköz 13 px)
    row_max_height: float = 32.0  # a valódi sor festett magassága 16–28 px; a magasabb jelöltet kettévágjuk
    row_min_height: float = 13.0
    row_min_width: float = 26.0
    trim_slack: float = 6.0  # a sor végének tükörképétől ennyire keresünk rést, ahol a ráragadt folt levágható
    # A középtől félre álló sort a középre tükrözve vágjuk (a ráragadt foltot levágja); a küldetéssor fehér része
    # a színes kulcsszó miatt áll félre, ott a vágás egy szóra szűkítené, a sor középre állását utólag mérjük
    trims_off_centre: bool = True
    # Elfogadás (laza): érdemes-e a sort odaadni a szövegolvasónak
    accept_offset: float = 45.0  # |sorközép − képközép|: a valódi sor ±2 px-en belül van, ha nem ragadt rá semmi
    min_fill: float = 0.07  # magpixel / festett doboz: valódi sor 0,09–0,21 (fehér pólón 0,28)
    max_fill: float = 0.30
    min_stroke_density: float = 0.062  # keresztezett vonások a szélesség egy pixelére: valódi sor 0,073–0,15
    max_run: float = 3.2  # a magpixelek átlagos vízszintes futáshossza = vonalvastagság: valódi sor 1,55–2,7 px
    # Bizonyosság (szigorú): olvasás nélkül, első látásra eltakarható
    sure_offset: float = 3.0
    sure_min_width: float = 34.0  # a legkeskenyebb mért valódi sor 35 px („No.”)
    sure_min_height: float = 15.0  # valódi sor: 16 (fel- és lenyúló szár nélkül) – 28
    sure_max_height: float = 29.0
    sure_min_fill: float = 0.085
    sure_max_fill: float = 0.22
    sure_min_stroke_density: float = 0.08
    sure_max_run: float = 2.5
    # A doboz ennyivel nagyobb a festett résznél: így hasonlít a neurális kereső (kitömött) dobozaira
    pad_left: float = 7.0
    pad_right: float = 5.0
    pad_top: float = 8.0
    pad_bottom: float = 5.0
    # A beszélő színes neve: a legvilágosabb csatornája legalább ennyi, és a csatornái legalább ennyire
    # eltérnek (a fehér és szürke betűtől így elválik); szintetikus, körvonalas betűkön mérve
    color_level: int = 120
    color_min_spread: int = 30
    # A névvel kiegészített sor legalább ilyen széles: a „Név: szó.” is az; a rendszám, egy színes folt nem
    named_min_width: float = 80.0
    # A teljes sáv szövege: a középső sorok mellett a sáv többi fehér szövege is sor (lásd a modul leírását)
    whole_band: bool = False
    # A szabad sor kitömött doboza legalább ennyire marad el a sáv bal és jobb szélétől: a szélhez érőt
    # a sáv széle elvágta (a telefon üzenetkártyája a jobb szélen), az úgysem olvasható helyesen.
    # A pipeline is ezt az értéket használja (row_finder.band_edge_margin)
    edge_margin: float = 12.0
    # A középső sor képsorain álló szabad sor legalább ekkora réssel kezdődik a középső sor mellett: a közelebbi
    # a középső sorról a tükrös vágással levágott vég (valós kockákon 5–25 px), a sor része, nem külön szöveg
    side_gap: float = 40.0


DEFAULT_PARAMS = Params()


@dataclass(frozen=True)
class Measures:
    """A küszöbök egy adott képre és sávra vonatkoztatva."""

    params: Params
    scale: float  # képszélesség / REFERENCE_WIDTH
    center: float  # a kép vízszintes közepe a sáv koordinátáiban
    window: Span  # a középső ablak oszlopai
    dialogue_bottom: float | None  # az ennél lejjebb (nagyobb y) lévő közepű sor a küldetésutasítás

    def px(self, size: float, minimum: int = 1) -> int:
        """Az 1920 széles képre mért méret ezen a képen, egész pixelben."""
        return _px(size, self.scale, minimum)


@dataclass(frozen=True, eq=False)  # eq=False: a numpy maszkok == összehasonlítása hibát dobna
class Detection:
    """A kereső eredménye egy sávra."""

    rows: tuple[Row, ...]  # fentről lefelé
    core: np.ndarray | None  # a betűmag-maszk, amelyen a sorokat találtuk (uint8, 0/255, csak olvasható)
    level: int  # a betűmag szintje: white_level, halvány szövegnél alacsonyabb


NOTHING = Detection((), None, 0)  # nincs sor: maszk sem készült


@dataclass(frozen=True)
class _Tone:
    """Egy menet betűszíne."""

    level: int  # a betű magja: mindhárom csatorna legalább ennyi
    outline_level: int  # a közelében legfeljebb ilyen világos pixel kell
    dim: bool  # a halvány (tévéműsor-felirat) menet


def detect(band_bgr: np.ndarray, band_left: int, frame_width: float, params: Params = DEFAULT_PARAMS, *,
           dialogue_bottom: float | None = None) -> Detection:
    """A sáv feliratsorai fentről lefelé, és a betűmag-maszk, amelyen találtuk őket.

    band_bgr: a feliratsáv (BGR vagy BGRA, uint8); band_left: a sáv bal széle a teljes képen;
    frame_width: a teljes kép szélessége. dialogue_bottom: a párbeszédsorok alsó határa a sáv
    koordinátáiban (ocr.frame_center(...).dialogue_bottom mínusz a sáv teteje); None: nincs határ.
    """
    if band_bgr.ndim != 3 or band_bgr.shape[2] not in (BGR_CHANNELS, BGR_CHANNELS + 1) or band_bgr.dtype != np.uint8:
        raise ValueError(f"band must be an 8-bit BGR or BGRA image, got shape {band_bgr.shape}, {band_bgr.dtype}")
    if frame_width <= 0:
        raise ValueError(f"frame_width must be positive, got {frame_width}")
    if band_bgr.size == 0:
        return NOTHING
    measures = _measures(params, band_bgr.shape[1], frame_width / 2.0 - band_left, frame_width, dialogue_bottom)
    centred = _centred(band_bgr, measures)
    # A szabad menet a középső döntés után fut: így a középső sorok (és a halvány menet választása) pontosan
    # ugyanazok, mint nélküle
    return _with_free_rows(band_bgr, centred, measures) if params.whole_band else centred


def _centred(band: np.ndarray, measures: Measures) -> Detection:
    """A kép közepére igazított sorok, és a biztos sorok mellett álló szürke közbeszólás-sorok."""
    found = _centred_rows(band, measures)
    return _with_grey_neighbours(band, found, measures) if any(row.sure for row in found.rows) else found


def _centred_rows(band: np.ndarray, measures: Measures) -> Detection:
    """A kép közepére igazított sorok: előbb fehér, biztos fehér sor nélkül halvány (tévéfelirat) betűvel, végül
    a lágy fehér menet, ha az egy sort egészebben lát (lásd soft_white_level)."""
    params = measures.params
    white = _scan(band, _white(params), measures)
    if any(row.sure for row in white.rows):
        return white
    found = _white_or_dim(band, white, measures)
    # A tömörített képen világos háttér (rendszám, napos fal) előtt a fehér betű magja a hófehér szint alá esik:
    # a sornak csak töredékeit látjuk, a halvány menet pedig a színes nevet nem. Ugyanazzal a körvonallal és
    # névvel, kicsit sötétebb szinten a sor még egyben megtalálható
    soft = _scan(band, _Tone(params.soft_white_level, params.outline_level, dim=False), measures)
    if sees_more(soft.rows, found.rows):
        return soft
    return _with_soft_neighbours(band, found, soft, measures)


def _white_or_dim(band: np.ndarray, white: Detection, measures: Measures) -> Detection:
    """Biztos fehér sor nélkül halvány szöveget is keresünk (fehér folt ne takarhassa el a tévéfeliratot)."""
    params = measures.params
    level = _text_level(band, params)
    if level >= params.white_level:
        return white  # világos jelenet: a fehér menet már mindent megnézett
    dim = _scan(band, _Tone(level, params.dim_outline_level, dim=True), measures)
    if dim_replaces_white(white.rows, dim.rows):
        return dim
    kept = keep_white_rows(white.rows, dim.rows)
    return white if kept == white.rows else replace(white, rows=kept)


def sees_more(soft: tuple[Row, ...], found: tuple[Row, ...]) -> bool:
    """A lágy fehér menetnek van-e biztos sora, amelyet a többi menet nem lát legalább ilyen szélesen, biztosként.

    A sötét jeleneten álló tévéfeliratot a halvány menet ugyanúgy megtalálja: ott az marad (dim sorként)."""
    return any(row.sure and not any(other.sure and _lies_within(row, other, WIDER_TOLERANCE_PX) for other in found)
               for row in soft)


def _with_grey_neighbours(band: np.ndarray, white: Detection, measures: Measures) -> Detection:
    """A biztos fehér feliratsor mellé a közvetlenül fölötte vagy alatta álló szürke sor is (a játék a
    névtelen közbeszólást, pl. „Lu! I thought…”, szürkén írja a megszólalás alá); a fehér sorok érintetlenek."""
    params = measures.params
    level = _text_level(band, params)
    if level >= params.white_level:
        return white  # világos jelenet: szürke betű nem válik el a háttértől
    # A közbeszólás szürkébb a tévéfeliratnál; csak a biztos sor mellett keressük, így a mélyebb szint is elég szigorú
    tone = _Tone(min(level, GREY_NEIGHBOUR_LEVEL), params.dim_outline_level, dim=True)
    dim = _scan(band, tone, measures)
    extra = grey_neighbours(white.rows, dim.rows)
    if not extra or white.core is None or dim.core is None:
        return white
    merged = white.core.copy()
    for row in extra:  # a sorjelhez (row_signature) a szürke sor betűi is a közös maszkba kerülnek
        lines, columns = slice(row.ink_top, row.ink_bottom), slice(row.ink_left, row.ink_right)
        merged[lines, columns] = dim.core[lines, columns]
    merged.setflags(write=False)
    return Detection(tuple(sorted(white.rows + extra, key=lambda r: (r.top, r.left))), merged, white.level)


def _with_soft_neighbours(band: np.ndarray, found: Detection, soft: Detection, measures: Measures) -> Detection:
    """A talált sorok mellé a közvetlenül fölöttük vagy alattuk álló szürke sor is, ha a lágy fehér menet látja.

    Napos háttér előtt a szürke közbeszólás („Locals: Uh, no…”) betűi a hófehér szint alatt vannak, és a
    halvány menet sem látja (a szint a háttér miatt magas); a lágy fehér menetben megvan, de nem biztos sorként.
    Önállóan nem fogadjuk el (napos felület hamis sort adna), csak egy már megtalált sor közvetlen
    szomszédjaként; a talált sorok érintetlenek. A lágy menet a sor szélét levágja (a bal vége a világos
    aszfalton elvész, a tükrös vágás a jobb végét is megrövidíti): a sor teljes szélességét lásd _widened."""
    if not found.rows or found.core is None or soft.core is None:
        return found
    params = measures.params
    narrow = tuple(row for row in soft_neighbours(found.rows, soft.rows)
                   if _grey_share(band, soft.core, row, params.dim_grey_spread) >= params.dim_min_grey_share)
    if not narrow:
        return found
    wide = _scan(band, _Tone(SOFT_NEIGHBOUR_LEVEL, params.outline_level, dim=False), measures)
    merged = found.core.copy()
    extra = []
    for row in narrow:
        row, core = _widened(row, soft.core, wide, found.rows)
        extra.append(replace(row, dim=True))
        lines, columns = slice(row.ink_top, row.ink_bottom), slice(row.ink_left, row.ink_right)
        merged[lines, columns] = core[lines, columns]  # a sorjelhez (row_signature) a sor betűi a közös maszkba kerülnek
    merged.setflags(write=False)
    return Detection(tuple(sorted(found.rows + tuple(extra), key=lambda r: (r.top, r.left))), merged, found.level)


def _widened(row: Row, core: np.ndarray, wide: Detection, found: tuple[Row, ...]) -> tuple[Row, np.ndarray]:
    """A szomszédsor a SOFT_NEIGHBOUR_LEVEL szinten kimért, szélesebb változatával és annak maszkjával (az a
    sor, amelynek oszlopai a szomszédsort tartalmazzák, és amely nem fedi a talált sorokat); ha nincs ilyen,
    a sor és a maszkja változatlan."""
    if wide.core is not None:
        for other in wide.rows:
            if (_vertical_gap(row, other) < 0 and other.ink_left <= row.ink_left <= row.ink_right <= other.ink_right
                    and not any(_vertical_gap(other, kept) < 0 for kept in found)):
                return other, wide.core
    return row, core


def soft_neighbours(found: tuple[Row, ...], soft: tuple[Row, ...]) -> tuple[Row, ...]:
    """A lágy menet sorai, amelyek egy talált sor közvetlen szomszédjai (a játék sorközénél nem nagyobb
    réssel), de egyik talált sor képsoraira sem esnek (azok a talált sorok újratalálásai)."""
    def is_neighbour(row: Row) -> bool:
        return any(_vertical_gap(row, other) <= NEIGHBOUR_GAP_ROWS * (other.ink_bottom - other.ink_top)
                   for other in found)

    return tuple(row for row in soft
                 if is_neighbour(row) and not any(_vertical_gap(row, other) < 0 for other in found))


def grey_neighbours(white: tuple[Row, ...], dim: tuple[Row, ...]) -> tuple[Row, ...]:
    """A halvány menet sorai, amelyek egy biztos fehér sor közvetlen szomszédjai (a játék sorközénél nem nagyobb
    réssel), de egyik fehér sor képsoraira sem esnek (azok a fehér sorok halvány újratalálásai)."""
    sure = [row for row in white if row.sure]

    def is_neighbour(row: Row) -> bool:
        return any(_vertical_gap(row, other) <= NEIGHBOUR_GAP_ROWS * (other.ink_bottom - other.ink_top)
                   for other in sure)

    return tuple(row for row in dim
                 if is_neighbour(row) and not any(_vertical_gap(row, other) < 0 for other in white))


def _vertical_gap(first: Row, second: Row) -> int:
    """A két sor betűi közötti függőleges rés; negatív, ha képsoraik fedik egymást."""
    return max(first.ink_top - second.ink_bottom, second.ink_top - first.ink_bottom)


def dim_replaces_white(white: tuple[Row, ...], dim: tuple[Row, ...]) -> bool:
    """A halvány menet sorai váltják-e a fehér menetéit.

    Igen, ha fehér sor nincs, vagy a halvány menet biztos sort talált, amely nem egy fehér sor szűkebb
    újratalálása. A színes nevet a halvány menet nem látja: a névvel együtt megtalált fehér sort név nélkül,
    a képközépre tükrözve levágva találná meg újra, és akkor a név balra, a mondat vége jobbra kilógna a takarásból.
    """
    if not dim:
        return False
    if not white:
        return True
    return (any(row.sure for row in dim)
            and not any(_lies_within(inner, outer) for inner in dim for outer in white))


def keep_white_rows(white: tuple[Row, ...], dim: tuple[Row, ...]) -> tuple[Row, ...]:
    """A megtartott fehér sorok; amelyikben a halvány menet biztos sort talált, az is biztos (olvasás előtt
    takarható): ugyanaz a sor, csak a fehér menet a színes névvel együtt, teljes szélességben látta."""
    return tuple(replace(row, sure=True) if not row.sure and any(other.sure and _lies_within(other, row)
                                                                    for other in dim) else row
                 for row in white)


# A szürke szomszédsor legfeljebb ennyi sormagasságnyira áll a fehér sortól: a játék sorköze ~1 sormagasság,
# a küldetésutasítás ~2-re van a párbeszéd alatt
NEIGHBOUR_GAP_ROWS = 1.5
# A szürke közbeszólás betűmagja ~110-150 (720p-s felvételen mérve); a tévéfelirat halvány szintje (dim_level) ennél magasabb
GREY_NEIGHBOUR_LEVEL = 95
# A napos háttér előtti szürke közbeszólás betűmagja ~160 felett marad (a 180-as lágy szinten a bal vége elvész)
SOFT_NEIGHBOUR_LEVEL = 160
SAME_LINE_TOLERANCE_PX = 4  # a két menet ugyanannak a sornak a tetejét és alját ennyin belül méri
# A lágy fehér menet ennyivel szélesebben is ugyanazt a sort látja (más szint, más élsimítás); a beszélőnév ~70 px
WIDER_TOLERANCE_PX = 8


def _lies_within(inner: Row, outer: Row, slack: int = 0) -> bool:
    """inner ugyanaz a sor (teteje és alja egyezik), és oszlopai outer festett kiterjedésén belül vannak
    (slack: ennyi képponttal oldalanként túl is érhet rajta)."""
    same_line = (abs(inner.ink_top - outer.ink_top) <= SAME_LINE_TOLERANCE_PX
                 and abs(inner.ink_bottom - outer.ink_bottom) <= SAME_LINE_TOLERANCE_PX)
    return same_line and outer.ink_left - slack <= inner.ink_left and inner.ink_right <= outer.ink_right + slack


def _white(params: Params) -> _Tone:
    return _Tone(params.white_level, params.outline_level, dim=False)


def _with_free_rows(band: np.ndarray, centred: Detection, measures: Measures) -> Detection:
    """A középső eredmény kiegészítve a sáv többi fehér szövegével (a középső sorok érintetlenek).

    Csak fehér betűt keresünk: a halvány szinttel a sáv szürke kijelzőszövegei (HUD, tévé) is sorok lennének.
    """
    tone = _white(measures.params)
    core = _core_mask(band, tone, measures)
    for row in centred.rows:  # a középső sorok helye (a kitömött dobozuk) már foglalt
        core[row.top:row.bottom, row.left:row.right] = 0
    gap = measures.px(measures.params.side_gap)
    free = tuple(row for row in find_free_rows(core, measures)
                 if not any(_is_cut_off(row, other, gap) for other in centred.rows))
    if not free:
        return centred
    merged = np.zeros(band.shape[:2], np.uint8) if centred.core is None else centred.core.copy()
    for row in free:  # a sorjelhez (row_signature) a szabad sor betűi is a közös maszkba kerülnek
        lines, columns = slice(row.ink_top, row.ink_bottom), slice(row.ink_left, row.ink_right)
        merged[lines, columns] = core[lines, columns]
    merged.setflags(write=False)
    rows = tuple(sorted(centred.rows + free, key=lambda r: (r.top, r.left)))
    return Detection(rows, merged, centred.level if centred.rows else tone.level)


def _is_cut_off(free: Row, centred: Row, gap: int) -> bool:
    """A szabad sor a középső sor képsorain, gap-nél közelebb áll mellette: a középső sor levágott vége."""
    if not overlaps(free, centred, by_columns=False):
        return False
    return max(centred.ink_left - free.ink_right, free.ink_left - centred.ink_right) < gap


def text_extent(band_bgr: np.ndarray, row: Row, band_left: int, frame_width: float, max_gap: float,
                params: Params = DEFAULT_PARAMS) -> Span:
    """A sor képsorain álló fehér és színes betűk összefüggő oszloptartománya [left, right) a sáv koordinátáiban.

    A tartomány a sor festett részéből indul, és a legfeljebb max_gap (1920-as képre mért) képpontos réseken
    át terjed (pl. a küldetésutasítás „Hold [gomb] to …” gombikonja fölött); a színes kulcsszót is tartalmazza."""
    measures = _measures(params, band_bgr.shape[1], frame_width / 2.0 - band_left, frame_width, None)
    margin = _margin(measures)
    top, bottom = max(0, row.ink_top - margin), min(band_bgr.shape[0], row.ink_bottom + margin)
    image = np.ascontiguousarray(band_bgr[top:bottom])
    letters = cv2.bitwise_or(_core_mask(image, _white(params), measures), _coloured_mask(image, measures))
    lines = letters[row.ink_top - top:row.ink_bottom - top]
    inked = cv2.reduce(lines, 0, cv2.REDUCE_MAX).ravel() > 0
    inked[row.ink_left:row.ink_right] = True
    left, right = row.ink_left, row.ink_right
    reach = measures.px(max_gap)
    while left > 0 and inked[max(0, left - reach):left].any():
        left = max(0, left - reach) + int(np.flatnonzero(inked[max(0, left - reach):left])[0])
    while right < inked.size and inked[right:right + reach].any():
        right = right + int(np.flatnonzero(inked[right:right + reach])[-1]) + 1
    return left, right


def detect_rows(band_bgr: np.ndarray, band_left: int, frame_width: float, params: Params = DEFAULT_PARAMS, *,
                dialogue_bottom: float | None = None) -> tuple[Row, ...]:
    """Csak a sorok: detect(...).rows."""
    return detect(band_bgr, band_left, frame_width, params, dialogue_bottom=dialogue_bottom).rows


def to_frame(row: Row, band_left: int, band_top: int) -> Row:
    """Ugyanaz a sor a teljes kép koordinátáiban; band_left, band_top: a sáv bal felső sarka a képen."""
    return replace(row, left=row.left + band_left, right=row.right + band_left, top=row.top + band_top,
                   bottom=row.bottom + band_top, ink_left=row.ink_left + band_left,
                   ink_right=row.ink_right + band_left, ink_top=row.ink_top + band_top,
                   ink_bottom=row.ink_bottom + band_top)


def _px(size: float, scale: float, minimum: int = 1) -> int:
    return max(minimum, int(round(size * scale)))


def _measures(params: Params, band_width: int, center: float, frame_width: float,
              dialogue_bottom: float | None) -> Measures:
    scale = frame_width / REFERENCE_WIDTH
    half = _px(params.center_half, scale)
    window = (max(0, int(center) - half), min(band_width, int(center) + half))
    return Measures(params, scale, center, window, dialogue_bottom)


def _scan(band: np.ndarray, tone: _Tone, measures: Measures) -> Detection:
    """Egy menet: sorjelöltek a középső ablakból, majd sorok a jelöltek környékének teljes szélességű maszkjából."""
    parts = _central_parts(band, tone, measures)
    if not parts:
        return NOTHING
    core = _strip_mask(band, parts, tone, measures)
    # A színes beszélőnév betűi a sor két végének kiméréséhez számítanak (a halvány tévéfeliratnál nincs név)
    named = None if tone.dim else _with_coloured_names(band, core, parts, measures)
    rows = find_rows(core, parts, measures, tone.dim, named)
    if tone.dim:
        params = measures.params
        rows = tuple(row for row in rows
                     if _grey_share(band, core, row, params.dim_grey_spread) >= params.dim_min_grey_share)
    return Detection(rows, core, tone.level) if rows else NOTHING


def _central_parts(band: np.ndarray, tone: _Tone, measures: Measures) -> list[Span]:
    """Olcsó kapu: sorjelöltek csak a középső ablak maszkjából (a morfológiához kellő margóval kivágva)."""
    left, right = measures.window
    if right <= left:
        return []
    margin = _margin(measures)
    crop_left, crop_right = max(0, left - margin), min(band.shape[1], right + margin)
    core = _core_mask(band[:, crop_left:crop_right], tone, measures)
    return seed_parts(core[:, left - crop_left:right - crop_left], measures)


def _with_coloured_names(band: np.ndarray, core: np.ndarray, parts: list[Span], measures: Measures) -> np.ndarray:
    """A fehér betűmag-maszk a színes (beszélőnév) betűmagokkal kiegészítve, ugyanazokon a sorokon."""
    top, bottom = _strip_lines(band.shape[0], parts, measures)
    coloured = np.zeros(band.shape[:2], np.uint8)
    coloured[top:bottom] = _coloured_mask(band[top:bottom], measures)
    _clear_cut_edges(coloured, top, bottom, measures)
    return cv2.bitwise_or(core, coloured)


def _strip_lines(height: int, parts: list[Span], measures: Measures) -> Span:
    """A jelöltek által elérhető képsorok [top, bottom)."""
    reach = measures.px(measures.params.row_max_height) + _margin(measures)
    return max(0, parts[0][0] - reach), min(height, parts[-1][1] + reach)


def _clear_cut_edges(mask: np.ndarray, top: int, bottom: int, measures: Measures) -> None:
    """A mesterséges vágásszél melletti sorok nem pontosak: ott semmit se találjunk."""
    margin = _margin(measures)
    if top > 0:
        mask[top:top + margin] = 0
    if bottom < mask.shape[0]:
        mask[bottom - margin:bottom] = 0


def _strip_mask(band: np.ndarray, parts: list[Span], tone: _Tone, measures: Measures) -> np.ndarray:
    """Teljes szélességű betűmag-maszk, csak a jelöltek által elérhető sorokon kiszámolva (máshol 0)."""
    top, bottom = _strip_lines(band.shape[0], parts, measures)
    core = np.zeros(band.shape[:2], np.uint8)
    core[top:bottom] = _core_mask(band[top:bottom], tone, measures)
    _clear_cut_edges(core, top, bottom, measures)
    core.setflags(write=False)
    return core


def _core_mask(image: np.ndarray, tone: _Tone, measures: Measures) -> np.ndarray:
    """255 a betűk magján: vékony világos alakzat, amelyhez közel sötét pixel van (csatornabontás nélkül)."""
    alpha = image.shape[2] - BGR_CHANNELS
    white = cv2.inRange(image, (tone.level,) * BGR_CHANNELS + (0,) * alpha, (MAX_LEVEL,) * image.shape[2])
    dark = cv2.inRange(image, (0,) * image.shape[2], (tone.outline_level,) * BGR_CHANNELS + (MAX_LEVEL,) * alpha)
    thin = cv2.subtract(white, cv2.morphologyEx(white, cv2.MORPH_OPEN, _square(_stroke_px(measures) + 1)))
    return cv2.bitwise_and(thin, cv2.dilate(dark, _square(2 * _reach_px(measures) + 1)))


def _coloured_mask(image: np.ndarray, measures: Measures) -> np.ndarray:
    """255 a színes betűk magján: világos, telített, vékony alakzat, amelyhez közel sötét (körvonal) pixel van."""
    params = measures.params
    blue, green, red = cv2.split(image)[:BGR_CHANNELS]
    brightest = cv2.max(cv2.max(blue, green), red)
    darkest = cv2.min(cv2.min(blue, green), red)
    _, bright = cv2.threshold(brightest, params.color_level - 1, MAX_LEVEL, cv2.THRESH_BINARY)
    _, saturated = cv2.threshold(cv2.subtract(brightest, darkest), params.color_min_spread - 1, MAX_LEVEL,
                                 cv2.THRESH_BINARY)
    coloured = cv2.bitwise_and(bright, saturated)
    _, dark = cv2.threshold(brightest, params.outline_level, MAX_LEVEL, cv2.THRESH_BINARY_INV)
    thin = cv2.subtract(coloured, cv2.morphologyEx(coloured, cv2.MORPH_OPEN, _square(_stroke_px(measures) + 1)))
    return cv2.bitwise_and(thin, cv2.dilate(dark, _square(2 * _reach_px(measures) + 1)))


def _stroke_px(measures: Measures) -> int:
    return measures.px(measures.params.stroke_max, MIN_STROKE_PX)


def _reach_px(measures: Measures) -> int:
    return measures.px(measures.params.outline_reach)


def _margin(measures: Measures) -> int:
    """Mesterséges vágásszéltől ennyi soron/oszlopon belül a maszk eltérhet a teljes sáv maszkjától."""
    return _stroke_px(measures) + _reach_px(measures) + 1


@lru_cache(maxsize=KERNEL_CACHE_SIZE)
def _square(side: int) -> np.ndarray:
    """Négyzet alakú szerkesztőelem (közös, gyorsítótárazott példány: csak olvasható)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (side, side))
    kernel.setflags(write=False)
    return kernel


def _text_level(band: np.ndarray, params: Params) -> int:
    """A betűmag szintje halvány feliratnál: a sáv legvilágosabb pixelei alatt (frosted_cover._text_level)."""
    height, width = band.shape[:2]
    size = (max(1, width // LEVEL_SAMPLE_STEP_X), max(1, height // LEVEL_SAMPLE_STEP_Y))
    blue, green, red = cv2.split(cv2.resize(band, size, interpolation=cv2.INTER_NEAREST))[:BGR_CHANNELS]
    darkest = cv2.min(cv2.min(blue, green), red)
    histogram = cv2.calcHist([darkest], [0], None, [MAX_LEVEL + 1], [0, MAX_LEVEL + 1]).ravel()
    at_least = np.cumsum(histogram[::-1])  # ennyi pixel értéke legalább 255, 254, ...
    top = MAX_LEVEL - int(np.searchsorted(at_least, darkest.size * params.dim_top_fraction, side="left"))
    return int(min(params.white_level, max(params.dim_level, top - params.dim_margin)))


def _grey_share(band: np.ndarray, core: np.ndarray, row: Row, spread: int) -> float:
    """A sor magpixeleinek mekkora része szürke vagy fehér (közel egyforma színcsatornák), nem színes fény."""
    crop = band[row.ink_top:row.ink_bottom, row.ink_left:row.ink_right]
    blue, green, red = cv2.split(crop)[:BGR_CHANNELS]
    brightest = cv2.max(cv2.max(blue, green), red)
    darkest = cv2.min(cv2.min(blue, green), red)
    _, grey = cv2.threshold(cv2.subtract(brightest, darkest), spread, MAX_LEVEL, cv2.THRESH_BINARY_INV)
    mask = core[row.ink_top:row.ink_bottom, row.ink_left:row.ink_right]
    total = cv2.countNonZero(mask)
    return cv2.countNonZero(cv2.bitwise_and(mask, grey)) / float(total) if total else 0.0
