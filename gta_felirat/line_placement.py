"""A lefordított sorok elhelyezése az angol sorok aktuális helyére.

A játék görgeti a feliratot: az új sor alul jelenik meg, a régebbi feljebb csúszik. Ezért a
fordítást soronként tartjuk nyilván, és minden olvasásnál oda tesszük, ahol az angol párja
éppen áll. Amelyik angol sornak még nincs fordítása, azt rövid ideig üresre takarjuk, hogy
ne villanjon fel angolul; ha a fordítás késik, az angol sor újra látszik.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from .layer_plan import Shown, ShownLine
from .subtitle_tracker import normalize, similarity

if TYPE_CHECKING:
    from .ocr import Box, OcrLine
    from .overlay import SpeakerColors

MATCH_THRESHOLD = 0.7  # ennyire kell hasonlítania az olvasott sornak a lefordított sor angol eredetijére
ANCHOR_JITTER_PX = 3  # ennél kisebb elmozdulás OCR-zaj: a szöveg ne remegjen
# Világos, mozgó háttéren a sor két széle olvasásonként ennyit is vándorol (a sor maga középre igazítva áll)
ANCHOR_WIDTH_JITTER_PX = 12
MIN_SHARED_ROW_RATIO = 0.5  # egy másik sor magasságának ekkora része lóg a helyre: azt a sort már ő foglalja
PENDING_HIDE_SECONDS = 1.5  # eddig takarjuk a még le nem fordított angol sort; utána újra látszik


@dataclass(frozen=True)
class TranslatedLine:
    source: str  # az angol sor
    text: str  # a fordítása


def pair_lines(source_text: str, translated_text: str) -> tuple[TranslatedLine, ...]:
    """Soronként párosítja az angolt a fordítással; üres, ha a sorok száma nem egyezik."""
    sources, texts = source_text.split("\n"), translated_text.split("\n")
    if len(sources) != len(texts):
        return ()
    return tuple(TranslatedLine(source, text) for source, text in zip(sources, texts))


def _is_same_place(first: Box, second: Box) -> bool:
    edges = ((first.left, second.left), (first.top, second.top),
             (first.right, second.right), (first.bottom, second.bottom))
    return all(abs(a - b) <= ANCHOR_JITTER_PX for a, b in edges)


def _steady_anchor(text: str, box: Box, others: list[Box], previous: Shown | None) -> Box:
    """A fordított sor helye: az előző marad, amíg az olvasás csak pontatlanabb lett, a sor nem mozdult.

    others: a többi most olvasott sor doboza; ami közülük az előző helyre lóg, az már az ő soruk.
    """
    if previous is None:
        return box
    for line in previous.lines:
        if line.text != text:
            continue
        if _is_within(box, line.anchor) and not _is_shared(line.anchor, box, others):
            return line.anchor  # szűkebben olvastuk (pl. kimaradt a mondat második sora): a hely marad
        if _is_same_row(line.anchor, box):
            return _widened(line.anchor, box)  # a széle kijjebb van: a takarás nőjön, de ne ugráljon
    return box


def _is_within(box: Box, anchor: Box) -> bool:
    return (box.left >= anchor.left - ANCHOR_JITTER_PX and box.right <= anchor.right + ANCHOR_JITTER_PX
            and box.top >= anchor.top - ANCHOR_JITTER_PX and box.bottom <= anchor.bottom + ANCHOR_JITTER_PX)


def _is_same_row(anchor: Box, box: Box) -> bool:
    return (abs(anchor.top - box.top) <= ANCHOR_JITTER_PX and abs(anchor.bottom - box.bottom) <= ANCHOR_JITTER_PX
            and abs(anchor.left - box.left) <= ANCHOR_WIDTH_JITTER_PX
            and abs(anchor.right - box.right) <= ANCHOR_WIDTH_JITTER_PX)


def _widened(anchor: Box, box: Box) -> Box:
    return replace(anchor, left=min(anchor.left, box.left), right=max(anchor.right, box.right))


def _is_shared(anchor: Box, box: Box, others: list[Box]) -> bool:
    """Áll-e más sor az előző hely azon részén, amelyet a mostani olvasat már nem fed?"""
    return any(_row_overlap(other, anchor) >= MIN_SHARED_ROW_RATIO > _row_overlap(other, box) for other in others)


def _row_overlap(row: Box, area: Box) -> float:
    """A sor magasságának mekkora része esik a terület sorai közé."""
    if row.height <= 0:
        return 0.0
    return max(0, min(row.bottom, area.bottom) - max(row.top, area.top)) / row.height


def _steady_box(box: Box, previous: Shown | None) -> Box:
    if previous is not None:
        for old in previous.hidden:
            if _is_same_place(old, box):
                return old
    return box


def _match(line: TranslatedLine, candidates: list[OcrLine]) -> OcrLine | None:
    best = max(candidates, key=lambda candidate: similarity(normalize(candidate.text), line.source), default=None)
    if best is None or similarity(normalize(best.text), line.source) < MATCH_THRESHOLD:
        return None
    return best


def match_lines(translated: tuple[TranslatedLine, ...],
                live: tuple[OcrLine, ...]) -> tuple[list[tuple[TranslatedLine, OcrLine]], list[OcrLine]]:
    """(fordítás–élő sor párok, a fordítás nélkül maradt élő sorok)."""
    free = list(live)
    pairs = []
    for line in translated:
        match = _match(line, free)
        if match is not None:
            free.remove(match)
            pairs.append((line, match))
    return pairs, free


def _without(boxes: list[Box], own: Box) -> list[Box]:
    return [box for box in boxes if box is not own]


def _union(boxes: list[Box]) -> Box:
    return type(boxes[0])(left=min(b.left for b in boxes), top=min(b.top for b in boxes),
                          right=max(b.right for b in boxes), bottom=max(b.bottom for b in boxes))


def build_shown(pairs: list[tuple[TranslatedLine, OcrLine]], hidden: list[Box],
                previous: Shown | None, *, speaker_colors: SpeakerColors = ()) -> Shown | None:
    """A kiírandó állapot: fordított sorok a helyükön, plusz a csak eltakart angol sorok.

    speaker_colors: a beszélőnevek mintázott színe; ha változik, új Shown készül (újrarajzolás).
    """
    live_boxes = [live.box for _line, live in pairs] + list(hidden)
    placed = sorted((ShownLine(line.text, _steady_anchor(line.text, live.box, _without(live_boxes, live.box),
                                                         previous))
                     for line, live in pairs), key=lambda line: line.anchor.top)
    hidden_boxes = tuple(_steady_box(box, previous) for box in hidden)
    if not placed and not hidden_boxes:
        return None
    anchor = _union([line.anchor for line in placed] or list(hidden_boxes))
    shown = Shown("\n".join(line.text for line in placed), anchor, tuple(placed), hidden_boxes, speaker_colors)
    return previous if shown == previous else shown


def place_lines(translated: tuple[TranslatedLine, ...], live: tuple[OcrLine, ...],
                previous: Shown | None) -> Shown | None:
    """A fordított sorok az élő angol sorok helyén; None, ha egyik angol párja sem látszik."""
    pairs, _untranslated = match_lines(translated, live)
    return build_shown(pairs, [], previous)


class PendingLines:
    """Mióta látszik egy még le nem fordított angol sor: rövid ideig takarjuk, utána nem."""

    def __init__(self) -> None:
        self._since: dict[str, float] = {}

    def boxes_to_hide(self, untranslated: list[OcrLine], now: float) -> list[Box]:
        seen: dict[str, float] = {}
        boxes = []
        for line in untranslated:
            key = normalize(line.text)
            since = next((start for known, start in self._since.items()
                          if similarity(known, key) >= MATCH_THRESHOLD), now)
            seen[key] = since
            if now - since < PENDING_HIDE_SECONDS:
                boxes.append(line.box)
        self._since = seen  # ami már nem látszik, azt elfelejtjük
        return boxes

    def clear(self) -> None:
        self._since = {}
