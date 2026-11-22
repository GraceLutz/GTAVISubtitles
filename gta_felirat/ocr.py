"""Az angol felirat kiolvasása a képkocka alsó sávjából (RapidOCR).

Két lépésben olvasunk: előbb megkeressük, hol van feliratszínű szöveg a sáv közepén (keresés),
aztán csak ezeket a sorokat olvassuk be (olvasás). Így a kijelző többi szövegére nem megy idő,
és egy már ismert helyű sort keresés nélkül is újra lehet olvasni (read_rows).
A kép közepétől távoli szöveget (whole_band_text) csak a soralapú út keresi (row_detector): az egész
sáv a neurális keresőnek túl nagy (lekicsinyítve a középső feliratsorok elveszhetnének), ezért itt a
keresés a kapcsolótól függetlenül a sáv középső részét nézi.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

import cv2
import numpy as np

from .config import Region
from .fragment_filter import (DIALOGUE_BOTTOM_RATIO, MIN_WHITE_FRACTION, Fragment, FrameCenter, row_pieces,
                              select_subtitle, white_fraction)
from .speaker_color import SpeakerColors, sample_speaker_colors

BRIGHT_PIXEL_LEVEL = 160  # a tévéműsor halványabb felirata (~183) is számítson
# Sötét jelenetben egy rövid sor („It's me.”) a sáv ~0,1%-a; 0,2%-os küszöbbel be sem olvastuk
MIN_BRIGHT_FRACTION = 0.0003
LINE_MERGE_TOLERANCE_PX = 12
# A kereső a kép szélességének ezt a középső részét nézi: a párbeszéd középen áll, a kijelző nem
DETECT_SPAN = (0.22, 0.78)
EDGE_MARGIN_PX = 6  # ennél közelebb a kivágás széléhez: a sor valószínűleg kilóg belőle
CUT_OFF_MIN_WIDTH_RATIO = 0.7  # a kivágás széléig érő, középre igazított sor legalább ilyen széles
ROW_PAD_X_AT_1080P = 8  # az olvasáshoz a sort oldalt ennyivel bővítjük (a kereső a záró pontokat gyakran levágja)
REFERENCE_FRAME_HEIGHT = 1080
# Zajszűrés: HUD-töredékek, táblák ("OD", "k Mall") ne menjenek fordításra
MIN_LETTERS_PLAIN = 8
MIN_LETTERS_SENTENCE = 2
SENTENCE_END = (".", "!", "?", "…")
MAX_LEADING_FRAGMENT_LETTERS = 3  # pl. "oes" egy levágott szó vége a beszélő előtt
MAX_ICON_CHARS = 2  # gomb-ikon vagy HUD-maradék a mondat után: „8”, „C8”, „X”, „TS”
# Ha a szavak ennyi része ékezetes, az nem angol felirat (pl. a PS5 magyar menüje)
MAX_ACCENTED_WORD_RATIO = 0.4
LATIN_SCRIPT_LIMIT = 0x24F  # a Latin bővített-B blokk vége: fölötte nem angol betű
# HUD-értesítés jele: előjeles érték- vagy képességváltozás („Focus +10%”, „-5”, „+$250”)
STAT_CHANGE_TOKEN = re.compile(r"^[+\-−]\$?\d+(?:[.,]\d+)?%?$")
# HUD-értesítés jele: a szavak többsége csupa nagybetűs („Focus SMALL SALTW BONUS”)
MIN_SHOUTED_WORD_LETTERS = 2  # az „I” és az „A” nem számít
MAX_SHOUTED_WORD_RATIO = 0.5
# Beszélő neve: csak betűk (számjegy nem, így a hívás-időzítő „Jason 00:04” nem beszélő)
SPEAKER_PATTERN = re.compile(r"^[A-Z](?:[^\W\d_]|[ .'&-]){0,30}:\s*\S")
CUT_SPEAKER_NAME = re.compile(r"^[a-z][\w'’ -]{0,23}:\s")  # a név eleje lemaradt („ocal: Bruh.”)
# Világos háttér előtt a név utáni kettőspontot az OCR pontosvesszőnek olvassa („Cal; Alright…”); legfeljebb
# kétszavas név után ez beszélő, nem mondatközi pontosvessző
MISREAD_SPEAKER_COLON = re.compile(r"^([A-Z][^\W\d_]*(?:[ .'-][A-Za-z][^\W\d_]*)?);(?=\s*\S)")
# Az OCR betűhibái, amelyek angolul nem lehetnek szavak, és félrefordítást okoznak (a fordítás a hibás szóból
# találgat: „On my od” → „Az én véleményem szerint”):
# - kisbetűs „l” az „I” helyén az összevonásban („l'm”, „l'll”);
LOWER_L_AS_I = re.compile(r"(?<![\w'’])l(?=['’](?:m|ll|ve|d)\b)")
# - a felkiáltójel „i”-nek olvasva az ismételt szó előtt („Movei Move it!”);
EXCLAMATION_AS_I = re.compile(r"\b([A-Za-z]{2,})i (?=(?i:\1)\b)")
# - a „god” első betűje elveszett („On my od”)
GOD_WITHOUT_G = re.compile(r"\b([Mm]y) od\b")
# - a „girl” utolsó „l”-je „i”-nek olvasva („homegiri”, „giri”)
GIRL_AS_GIRI = re.compile(r"(?<![^\W\d_])((?:[Hh]ome)?[Gg]ir)i\b")
# A mondatvégi írásjel után ottmaradt kósza függőleges vonás („…to eat? l.”): az OCR betűnek olvassa
STRAY_STROKE_AT_END = re.compile(r"(?<=[?!.])\s+[l|]\.*$", re.MULTILINE)
# - az „I” egybeolvasva a mögötte álló igével („When Isay all that?” → „Mikor mondom…”); csak gyakori igék,
#   mert az I-vel kezdődő valódi szavakat (It, Is, Into, Inside) nem szabad szétvágni
I_GLUED_TO_VERB = re.compile(r"\bI(say|said|want|know|knew|need|think|thought|got|was|can|could|will|would|"
                             r"did|do|have|had|like|love|mean|just|feel|told|gotta|swear|ain't)\b")
# Egész sort kitevő zárójeles hang- vagy zenefelirat; a hangjegyet az OCR néha „J”-nek olvassa, de a zárójel marad
# A beúszó vagy elvágott hangfelirat zárójele még nem látszik („[frustrat”): az is hangfelirat, nem szöveg
CAPTION_ROW = re.compile(r"^[\[(][^\])]*[\])]?$|[♪♫]")
# A dal sora a hangjegyek között; a hangjegyet az OCR S, J, 5 vagy ∫ jelnek olvassa („DWNPLY: S Never sell out J”).
# A ∫ és a magában álló J a sor végén angol szövegben nem fordul elő; az S és az 5 csak párban, a sor két végén
# Egyetlen hangjegy is elég, ha angolul nem lehet betű: a sor elején magában álló S vagy J nagybetűs szó előtt
# („Lucia: S Big ol titties!”), magában álló J a sor közepén („…! J [laughs]”), vagy kisbetűhöz tapadó J a végén
# („Ass fatJ”)
SONG_ROW = re.compile(r"^(?:[A-Z][\w.'&\- ]{0,23}:\s+)?(?:[SJ5∫]\s.*\s[SJ5∫]$|[SJ∫]\s+[A-Z])"
                      r"|\s[J∫](?:\s|$)|[a-z][J∫]$")
# A sor elején magában álló kettőspont vagy pontosvessző (a beszélő neve nélkül maradt)
LEADING_ORPHAN_PUNCTUATION = re.compile(r"^\s*[:;]+\s*")
# A telefonhívás kijelzője a sor végén: „… alright? Jason 00:05 D”
# A felirat háttérdobozának széle a mondatvégre tapadva: „huh?3”, „heard?湖”
GLUED_TAIL = re.compile(r"([.!?…][\"'”’)]*)(?:\d|[^\x00-\u024F]){1,%d}$" % MAX_ICON_CHARS)
CALL_TIMER_TAIL = re.compile(r"\s+(?:[A-Z][\w'.-]*\s+){0,2}\d{1,2}:\d{2}(?:\s+\S{1,2})*\s*$")


@dataclass(frozen=True)
class Box:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(frozen=True)
class OcrLine:
    text: str
    box: Box
    rows: tuple[Box, ...] = ()  # a megszólalás képernyősorai (kettő, ha a mondat két sorba tört)


@dataclass(frozen=True)
class OcrResult:
    text: str  # a megszólalások sortöréssel elválasztva
    box: Box | None
    lines: tuple[OcrLine, ...] = ()  # megszólalásonként a szöveg és a helye
    speaker_colors: SpeakerColors = ()  # a beszélőnevek mért színe


EMPTY_RESULT = OcrResult(text="", box=None)


def _speaker_colors(crop: np.ndarray, area: Box, lines: Sequence[OcrLine]) -> SpeakerColors:
    """A megszólalások beszélőneveinek színe; a két sorba tört megszólalás mindkét sorát mérjük."""
    return tuple(sample for line in lines
                 for sample in sample_speaker_colors(crop, area, line.text, *(line.rows or (line.box,))))


def frame_center(region: Region, area: Box) -> FrameCenter:
    """A teljes kép közepe és a párbeszédsorok alsó határa; area: a feliratsáv helye a képen."""
    frame_width = area.width / (region.right - region.left)
    frame_height = area.height / (region.bottom - region.top)
    frame_top = area.top - region.top * frame_height
    return FrameCenter(area.left + (0.5 - region.left) * frame_width, frame_width,
                       frame_top + DIALOGUE_BOTTOM_RATIO * frame_height)


def region_to_box(region: Region, frame_width: int, frame_height: int) -> Box:
    return Box(
        left=int(region.left * frame_width),
        top=int(region.top * frame_height),
        right=int(region.right * frame_width),
        bottom=int(region.bottom * frame_height),
    )


def has_bright_text(crop: np.ndarray) -> bool:
    """Gyors előszűrés: a GTA-felirat fehér, ha nincs világos pixel, nincs felirat."""
    # cv2-vel 0,6 ms a teljes sávon; a numpy max(axis=2) ugyanerre ~10 ms
    gray = cv2.max(cv2.max(crop[:, :, 0], crop[:, :, 1]), crop[:, :, 2]) if crop.ndim == 3 else crop
    bright = cv2.countNonZero(cv2.compare(gray, BRIGHT_PIXEL_LEVEL, cv2.CMP_GE))
    return bright >= MIN_BRIGHT_FRACTION * gray.size


def _is_stray_glyph(token: str) -> bool:
    return any(ch.isalpha() for ch in token) and all(
        not ch.isalpha() or ord(ch) > LATIN_SCRIPT_LIMIT for ch in token)


def strip_stray_glyphs(text: str) -> str:
    """Kiveszi az OCR által a HUD-ból vagy élsimításból kitalált nem latin jeleket (pl. „大”)."""
    return " ".join(token for token in text.split() if not _is_stray_glyph(token))


def drop_leading_fragment(text: str) -> str:
    """Levágja az elején álló, más szövegből becsúszott kisbetűs töredéket („oes Staff Member: …”)."""
    # A színes beszélőnév kimaradt az olvasásból, csak az utána álló kettőspont maradt: „: Hey, …”
    text = LEADING_ORPHAN_PUNCTUATION.sub("", text)
    tokens = text.split()
    if len(tokens) >= 2:
        first, second = tokens[0], tokens[1]
        if (first.isalpha() and first.islower() and len(first) <= MAX_LEADING_FRAGMENT_LETTERS
                and second[:1].isupper()):
            return " ".join(tokens[1:])
    # Gomb-ikon (X, R1…), HUD-szám vagy csupa nagybetűs tábla („TOPAWN SEL”) a beszélő neve előtt
    sign_count = 0
    while sign_count < len(tokens) - 1 and _is_icon_or_sign(tokens[sign_count]):
        sign_count += 1
    for start in range(sign_count, 0, -1):  # a leghosszabb levágástól indulunk
        rest = " ".join(tokens[start:])
        if SPEAKER_PATTERN.match(rest):
            return rest
    return text


def _is_icon_or_sign(token: str) -> bool:
    return len(token) == 1 or token.isdigit() or (token.isalpha() and token.isupper())


def drop_call_timer(text: str) -> str:
    """Levágja a sor végéről a telefonhívás nevét és időzítőjét, ha előtte lezárt mondat áll."""
    match = CALL_TIMER_TAIL.search(text)
    if match and text[:match.start()].rstrip().endswith(SENTENCE_END):
        return text[:match.start()].rstrip()
    return text


def drop_trailing_icons(text: str) -> str:
    """Levágja a lezárt mondat utáni rövid ikon-maradékokat („Menni fog, haver. 8”)."""
    tokens = text.split()
    while (len(tokens) >= 2 and len(tokens[-1]) <= MAX_ICON_CHARS
           and not tokens[-1].endswith(SENTENCE_END) and tokens[-2].endswith(SENTENCE_END)):
        tokens.pop()
    return " ".join(tokens)


def drop_glued_tail(text: str) -> str:
    """Levágja a lezárt mondat végére szóköz nélkül tapadt számjegyet vagy nem latin jelet („huh?3”)."""
    return GLUED_TAIL.sub(r"\1", text)


def fix_speaker_colon(text: str) -> str:
    """A beszélő neve utáni, pontosvesszőnek olvasott kettőspont visszaállítása."""
    return MISREAD_SPEAKER_COLON.sub(r"\1:", text)


def fix_letter_misreads(text: str) -> str:
    """Az OCR jellegzetes betűhibái, amelyek angolul nem lehetnek szavak (l'm, „Movei Move”, „my od”)."""
    text = LOWER_L_AS_I.sub("I", text)
    text = EXCLAMATION_AS_I.sub(r"\1! ", text)
    text = I_GLUED_TO_VERB.sub(r"I \1", text)
    text = STRAY_STROKE_AT_END.sub("", text)
    text = GIRL_AS_GIRI.sub(r"\1l", text)
    return GOD_WITHOUT_G.sub(r"\1 god", text)


def clean_ocr_text(text: str) -> str:
    text = drop_leading_fragment(fix_letter_misreads(fix_speaker_colon(strip_stray_glyphs(text))))
    return drop_glued_tail(drop_trailing_icons(drop_call_timer(text)))


def is_spoken_line(text: str) -> bool:
    """Kimondott szöveg: „Név: szöveg” alakú párbeszédsor (nem tábla, küldetésszöveg vagy menü)."""
    return bool(SPEAKER_PATTERN.match(text.strip()))


# Beszédre valló név nélküli sor (a tévéműsor felirata): kérdés vagy felkiáltás a végén, vagy első személy.
# A küldetésszöveg és a súgó felszólító, első személy nélkül, legfeljebb pont a végén („Go to the Vice City Port.”)
_SPOKEN_ENDING = re.compile(r"[?!][\"'”’)\]]*$")
# Kisbetűvel (az „I” mindig nagy): a menü „About Us” nagybetűs szava nem beszéd
_FIRST_PERSON = re.compile(r"\b(?:I(?:['’](?:m|ll|ve|d))?|[Mm]e|[Mm]y|mine|[Ww]e(?:['’](?:re|ll|ve))?|us|[Oo]ur)\b")
# Az „I” alakjai: sortörés után is nagybetűsek, a mondat folytatódhat velük
_FIRST_PERSON_SINGULAR = re.compile(r"[Ii](?:['’](?:m|ll|ve|d))?")


def sounds_spoken(text: str) -> bool:
    """Név nélkül is beszédnek hangzó sor: kérdés, felkiáltás vagy első személyű mondat."""
    stripped = text.strip()
    return bool(_SPOKEN_ENDING.search(stripped) or _FIRST_PERSON.search(stripped))


def is_mostly_accented(text: str) -> bool:
    """Az angolban ritka az ékezet; ha sok szóban van, az más nyelvű szöveg (menü, értesítés)."""
    words = [word for word in text.split() if any(ch.isalpha() for ch in word)]
    if not words:
        return False
    accented = sum(any(ch.isalpha() and not ch.isascii() for ch in word) for word in words)
    return accented / len(words) >= MAX_ACCENTED_WORD_RATIO


def is_hud_notification(text: str) -> bool:
    """Képesség-/értékváltozás vagy nagybetűs HUD-értesítés, ami a feliratsávba lóg."""
    tokens = text.split()
    if any(STAT_CHANGE_TOKEN.match(token) for token in tokens):
        return True
    words = [token for token in tokens if sum(ch.isalpha() for ch in token) >= MIN_SHOUTED_WORD_LETTERS]
    if not words:
        return False
    shouted = sum(all(ch.isupper() for ch in word if ch.isalpha()) for word in words)
    return shouted / len(words) > MAX_SHOUTED_WORD_RATIO


def looks_like_subtitle(text: str) -> bool:
    """Igaz, ha a szöveg angol feliratsornak tűnik, nem HUD-töredéknek, táblának vagy menünek."""
    text = text.strip()
    if is_mostly_accented(text):
        return False
    letters = sum(ch.isalpha() for ch in text)
    if SPEAKER_PATTERN.match(text):
        return True
    if is_hud_notification(text):
        return False  # a „Név:” nélküli sorban: játékértesítés, nem párbeszéd
    if letters and all(ch.isupper() for ch in text if ch.isalpha()):
        return False  # csupa nagybetű: tábla vagy HUD-felirat, nem párbeszéd
    if text.endswith(SENTENCE_END):
        return letters >= MIN_LETTERS_SENTENCE
    return letters >= MIN_LETTERS_PLAIN


def _rows(fragments: list[tuple[str, Box]]) -> list[OcrLine]:
    """A felismert darabok képernyősoronként (fentről le, balról jobbra) összefűzve."""
    ordered = sorted(fragments, key=lambda item: (item[1].top, item[1].left))
    rows: list[list[tuple[str, Box]]] = []
    for text, box in ordered:
        if rows and abs(rows[-1][0][1].top - box.top) <= LINE_MERGE_TOLERANCE_PX:
            rows[-1].append((text, box))
        else:
            rows.append([(text, box)])
    lines = []
    for row in rows:
        box = union_box([b for _, b in row])
        lines.append(OcrLine(" ".join(t for t, _ in sorted(row, key=lambda i: i[1].left)), box, (box,)))
    return lines


def join_lines(fragments: list[tuple[str, Box]]) -> str:
    """A felismert darabokat soronként (fentről le, balról jobbra) fűzi össze."""
    return " ".join(row.text for row in _rows(fragments)).strip()


def is_caption_row(text: str) -> bool:
    """Zárójeles hang- vagy zenefelirat („[♪ Money For Nothing ♪]”, „[music playing]”): nem kimondott szöveg."""
    stripped = text.strip()
    return bool(CAPTION_ROW.search(stripped) or SONG_ROW.search(stripped))


def _ends_with_misread_period(previous: str, row: str) -> bool:
    """A sor végi „,” valójában pont: a játék betűjében a kettő képről képre összetéveszthető.

    A sortörés utáni folytatás kisbetűvel kezdődik (kivéve az „I” alakjait); nagybetűs szó után új mondat jön.
    """
    first_word = row.lstrip("\"'“‘(").split(" ", 1)[0]
    return (previous.endswith(",") and first_word[:1].isupper()
            and not _FIRST_PERSON_SINGULAR.fullmatch(first_word.rstrip(",.!?…")))


def spoken_lines(fragments: list[tuple[str, Box]]) -> tuple[OcrLine, ...]:
    """Megszólalásonként egy sor: a lezáratlan mondat a következő képernyősorban folytatódik,
    kivéve ha az a sor új beszélővel kezdődik. A zárójeles hang- és zenefeliratot kihagyjuk."""
    lines: list[OcrLine] = []
    for row in _rows(fragments):
        if is_caption_row(row.text):
            continue
        starts_new_speaker = bool(SPEAKER_PATTERN.match(clean_ocr_text(row.text)))
        if lines and _ends_with_misread_period(lines[-1].text.rstrip(), row.text):
            previous = lines.pop()
            lines.append(replace(previous, text=previous.text.rstrip()[:-1] + "."))
        is_continuation = not lines[-1].text.endswith(SENTENCE_END) or _starts_lowercase(row.text) if lines else False
        if lines and not starts_new_speaker and is_continuation:
            previous = lines.pop()
            row = OcrLine(f"{previous.text} {row.text}", union_box([previous.box, row.box]),
                          previous.rows + row.rows)
        lines.append(row)
    cleaned = (OcrLine(clean_ocr_text(line.text), line.box, line.rows) for line in lines)
    return tuple(line for line in cleaned if line.text)


def _starts_lowercase(text: str) -> bool:
    """Kisbetűvel kezdődő sor: az előző mondat folytatása, akkor is, ha az előző sor végén pont áll („…outta
    here.. / there won't be any more trouble”). A levágott beszélőnév („ocal: Bruh.”) nem folytatás."""
    stripped = text.strip()
    return stripped[:1].islower() and not CUT_SPEAKER_NAME.match(stripped)


def union_box(boxes: list[Box]) -> Box:
    return Box(
        left=min(b.left for b in boxes),
        top=min(b.top for b in boxes),
        right=max(b.right for b in boxes),
        bottom=max(b.bottom for b in boxes),
    )


def _polygon_to_box(points, offset: Box) -> Box:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return Box(
        left=int(min(xs)) + offset.left,
        top=int(min(ys)) + offset.top,
        right=int(max(xs)) + offset.left,
        bottom=int(max(ys)) + offset.top,
    )


class TextEngine(Protocol):
    """Amit a szövegkeresőtől és -olvasótól várunk (ocr_engine.OcrEngine; a tesztekben hamisítvány)."""

    def detect(self, image: np.ndarray) -> list[np.ndarray]: ...

    def recognise(self, patches: list[np.ndarray]) -> list[tuple[str, float]]: ...


class SubtitleReader:
    def __init__(self, region: Region, min_score: float, engine: TextEngine | None = None) -> None:
        if engine is None:
            from .ocr_engine import OcrEngine  # lassú betöltés (modellek), ezért csak itt

            engine = OcrEngine()
        self._engine = engine
        self._region = region
        self._min_score = min_score

    def set_region(self, region: Region) -> None:
        """Új figyelt sáv; a munkaszálak a következő olvasástól ezt használják (egyetlen értékadás, szálbiztos)."""
        self._region = region

    def read(self, frame: np.ndarray) -> OcrResult:
        height, width = frame.shape[:2]
        area = region_to_box(self._region, width, height)
        return self.read_band(frame[area.top:area.bottom, area.left:area.right], area)

    def read_band(self, crop: np.ndarray, area: Box) -> OcrResult:
        """Már kivágott feliratsáv olvasása; area: a sáv helye a teljes képen."""
        if crop.size == 0 or not has_bright_text(crop):
            return EMPTY_RESULT
        rows = self.detect_rows(crop, area)
        if not rows:
            return EMPTY_RESULT
        boxes = [(text, box) for box, (text, score) in zip(rows, self.read_rows(crop, area, rows))
                 if score >= self._min_score and text]
        # Rendszám, minitérkép-betű és társai ne keveredjenek a feliratba
        fragments = select_subtitle([Fragment(text, box, white_fraction(crop, box, area)) for text, box in boxes],
                                    frame_center(self._region, area))
        if not fragments:
            return EMPTY_RESULT
        lines = spoken_lines([(f.text, f.box) for f in fragments])
        if not looks_like_subtitle(" ".join(line.text for line in lines)):
            return EMPTY_RESULT
        return OcrResult(text="\n".join(line.text for line in lines), box=union_box([line.box for line in lines]),
                         lines=lines, speaker_colors=_speaker_colors(crop, area, lines))

    def detect_rows(self, crop: np.ndarray, area: Box) -> list[Box]:
        """A feliratszínű szövegdarabok dobozai (a teljes képen), képernyősoronként; olvasás nélkül (~25 ms)."""
        left, right = self._detect_columns(area)
        rows = self._detect(crop, area, left, right)
        if (left, right) != (0, area.width) and any(self._is_cut_off(box, area, left, right) for box in rows):
            rows = self._detect(crop, area, 0, area.width)  # egy sor kilóg a középső részből: az egész sávot nézzük
        return rows

    def _detect_columns(self, area: Box) -> tuple[int, int]:
        """A sáv oszlopai, amelyeket a kereső megnéz (a kép szélességének középső része)."""
        frame_width = area.width / (self._region.right - self._region.left)
        frame_left = area.left - self._region.left * frame_width
        left = max(int(frame_left + DETECT_SPAN[0] * frame_width) - area.left, 0)
        right = min(int(frame_left + DETECT_SPAN[1] * frame_width) - area.left, area.width)
        if left >= right:
            return 0, area.width  # a sáv kívül esik a kép középső részén: üres kivágás helyett az egész sávot
        return left, right

    @staticmethod
    def _is_cut_off(box: Box, area: Box, left: int, right: int) -> bool:
        touches = box.left - area.left - left <= EDGE_MARGIN_PX or right - (box.right - area.left) <= EDGE_MARGIN_PX
        return touches and box.width >= CUT_OFF_MIN_WIDTH_RATIO * (right - left)

    def _detect(self, crop: np.ndarray, area: Box, left: int, right: int) -> list[Box]:
        offset = Box(area.left + left, area.top, area.left + right, area.bottom)
        boxes = [_polygon_to_box(points, offset) for points in self._engine.detect(crop[:, left:right])]
        # A nem feliratszínű darabot a szűrő úgyis eldobná: ne fizessünk darabonként 20–90 ms-ot az olvasásáért
        white = [box for box in boxes if white_fraction(crop, box, area) >= MIN_WHITE_FRACTION]
        return row_pieces(white)

    def read_rows(self, crop: np.ndarray, area: Box, rows: list[Box]) -> list[tuple[str, float]]:
        """(szöveg, bizonyosság) a megadott sordobozokhoz (a teljes képen); keresés nélkül (~25–65 ms soronként)."""
        frame_height = area.height / (self._region.bottom - self._region.top)
        pad = max(1, round(ROW_PAD_X_AT_1080P * frame_height / REFERENCE_FRAME_HEIGHT))
        patches = []
        for row in rows:
            top, bottom = max(row.top - area.top, 0), min(row.bottom - area.top, crop.shape[0])
            left, right = max(row.left - area.left - pad, 0), min(row.right - area.left + pad, crop.shape[1])
            patches.append(crop[top:bottom, left:right])
        return self._engine.recognise(patches)
