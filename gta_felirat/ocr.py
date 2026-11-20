"""Az angol felirat kiolvasása a képkocka alsó sávjából (RapidOCR)."""

import re
from dataclasses import dataclass

import numpy as np

from .config import Region

BRIGHT_PIXEL_LEVEL = 200
MIN_BRIGHT_FRACTION = 0.002
LINE_MERGE_TOLERANCE_PX = 12
DET_MAX_SIDE_PX = 960
# Zajszűrés: HUD-töredékek, táblák ("OD", "k Mall") ne menjenek fordításra
MIN_LETTERS_PLAIN = 8
MIN_LETTERS_SENTENCE = 2
SENTENCE_END = (".", "!", "?", "…")
MAX_LEADING_FRAGMENT_LETTERS = 3  # pl. "oes" egy levágott szó vége a beszélő előtt
MAX_ICON_CHARS = 2  # gomb-ikon vagy HUD-maradék a mondat után: „8”, „C8”, „X”, „TS”
# Ha a szavak ennyi része ékezetes, az nem angol felirat (pl. a PS5 magyar menüje)
MAX_ACCENTED_WORD_RATIO = 0.4
LATIN_SCRIPT_LIMIT = 0x24F  # a Latin bővített-B blokk vége: fölötte nem angol betű
# Beszélő neve: csak betűk (számjegy nem, így a hívás-időzítő „Jason 00:04” nem beszélő)
SPEAKER_PATTERN = re.compile(r"^[A-Z](?:[^\W\d_]|[ .'-]){0,30}:\s*\S")
# A telefonhívás kijelzője a sor végén: „… alright? Jason 00:05 D”
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
class OcrResult:
    text: str
    box: Box | None


EMPTY_RESULT = OcrResult(text="", box=None)


def region_to_box(region: Region, frame_width: int, frame_height: int) -> Box:
    return Box(
        left=int(region.left * frame_width),
        top=int(region.top * frame_height),
        right=int(region.right * frame_width),
        bottom=int(region.bottom * frame_height),
    )


def has_bright_text(crop: np.ndarray) -> bool:
    """Gyors előszűrés: a GTA-felirat fehér, ha nincs világos pixel, nincs felirat."""
    gray = crop.max(axis=2) if crop.ndim == 3 else crop
    bright_fraction = float((gray >= BRIGHT_PIXEL_LEVEL).mean())
    return bright_fraction >= MIN_BRIGHT_FRACTION


def _is_stray_glyph(token: str) -> bool:
    return any(ch.isalpha() for ch in token) and all(
        not ch.isalpha() or ord(ch) > LATIN_SCRIPT_LIMIT for ch in token)


def strip_stray_glyphs(text: str) -> str:
    """Kiveszi az OCR által a HUD-ból vagy élsimításból kitalált nem latin jeleket (pl. „大”)."""
    return " ".join(token for token in text.split() if not _is_stray_glyph(token))


def drop_leading_fragment(text: str) -> str:
    """Levágja az elején álló, más szövegből becsúszott kisbetűs töredéket („oes Staff Member: …”)."""
    tokens = text.split()
    if len(tokens) >= 2:
        first, second = tokens[0], tokens[1]
        if (first.isalpha() and first.islower() and len(first) <= MAX_LEADING_FRAGMENT_LETTERS
                and second[:1].isupper()):
            return " ".join(tokens[1:])
        rest = " ".join(tokens[1:])
        is_icon_or_hud = (len(first) == 1 or first.isdigit()
                          or (first.isalpha() and first.isupper() and len(first) <= MAX_LEADING_FRAGMENT_LETTERS))
        if is_icon_or_hud and SPEAKER_PATTERN.match(rest):
            return rest  # gomb-ikon (X, R1…), HUD-szám vagy -felirat („SEL”) a beszélő neve előtt
    return text


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


def clean_ocr_text(text: str) -> str:
    return drop_trailing_icons(drop_call_timer(drop_leading_fragment(strip_stray_glyphs(text))))


def is_spoken_line(text: str) -> bool:
    """Kimondott szöveg: „Név: szöveg” alakú párbeszédsor (nem tábla, küldetésszöveg vagy menü)."""
    return bool(SPEAKER_PATTERN.match(text.strip()))


def is_mostly_accented(text: str) -> bool:
    """Az angolban ritka az ékezet; ha sok szóban van, az más nyelvű szöveg (menü, értesítés)."""
    words = [word for word in text.split() if any(ch.isalpha() for ch in word)]
    if not words:
        return False
    accented = sum(any(ch.isalpha() and not ch.isascii() for ch in word) for word in words)
    return accented / len(words) >= MAX_ACCENTED_WORD_RATIO


def looks_like_subtitle(text: str) -> bool:
    """Igaz, ha a szöveg angol feliratsornak tűnik, nem HUD-töredéknek, táblának vagy menünek."""
    text = text.strip()
    if is_mostly_accented(text):
        return False
    letters = sum(ch.isalpha() for ch in text)
    if SPEAKER_PATTERN.match(text):
        return True
    if letters and all(ch.isupper() for ch in text if ch.isalpha()):
        return False  # csupa nagybetű: tábla vagy HUD-felirat, nem párbeszéd
    if text.endswith(SENTENCE_END):
        return letters >= MIN_LETTERS_SENTENCE
    return letters >= MIN_LETTERS_PLAIN


def join_lines(fragments: list[tuple[str, Box]]) -> str:
    """A felismert darabokat soronként (fentről le, balról jobbra) fűzi össze."""
    ordered = sorted(fragments, key=lambda item: (item[1].top, item[1].left))
    lines: list[list[tuple[str, Box]]] = []
    for text, box in ordered:
        if lines and abs(lines[-1][0][1].top - box.top) <= LINE_MERGE_TOLERANCE_PX:
            lines[-1].append((text, box))
        else:
            lines.append([(text, box)])
    rendered = [" ".join(t for t, _ in sorted(line, key=lambda i: i[1].left)) for line in lines]
    return " ".join(rendered).strip()


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


class SubtitleReader:
    def __init__(self, region: Region, min_score: float) -> None:
        # Lassú import, ezért csak itt töltjük be
        from rapidocr import RapidOCR

        # Az alapbeállítás a keskeny feliratsávot ~5000 px szélesre nagyítaná (1,3 s/olvasás);
        # max 960 px-re korlátozva ~75 ms ugyanazzal a pontossággal
        self._engine = RapidOCR(params={
            "Global.log_level": "error",
            "Global.use_cls": False,
            "Det.limit_type": "max",
            "Det.limit_side_len": DET_MAX_SIDE_PX,
        })
        self._region = region
        self._min_score = min_score
        self._warm_up()

    def _warm_up(self) -> None:
        # Az első hívás pár másodperc (modellbetöltés); ne játék közben fusson le
        dummy = np.zeros((64, 256, 3), np.uint8)
        dummy[24:40, 16:240] = 255
        self._engine(dummy)

    def read(self, frame: np.ndarray) -> OcrResult:
        height, width = frame.shape[:2]
        area = region_to_box(self._region, width, height)
        return self.read_band(frame[area.top:area.bottom, area.left:area.right], area)

    def read_band(self, crop: np.ndarray, area: Box) -> OcrResult:
        """Már kivágott feliratsáv olvasása; area: a sáv helye a teljes képen."""
        if crop.size == 0 or not has_bright_text(crop):
            return EMPTY_RESULT
        output = self._engine(crop)
        if output is None or not getattr(output, "txts", None):
            return EMPTY_RESULT
        fragments = [
            (text.strip(), _polygon_to_box(points, area))
            for text, score, points in zip(output.txts, output.scores, output.boxes)
            if score >= self._min_score and text.strip()
        ]
        if not fragments:
            return EMPTY_RESULT
        text = clean_ocr_text(join_lines(fragments))
        if not looks_like_subtitle(text):
            return EMPTY_RESULT
        return OcrResult(text=text, box=union_box([b for _, b in fragments]))
