"""Mit kell a feliratrétegre rajzolni, és hová: a kirajzolás terve.

A terv csak geometria (takarások, panel, szövegek helye), képpontokhoz nem nyúl. Így ugyanazt
a tervet kétféleképp lehet kirajzolni: a capture card képkockájára (lebegőpontos keverés),
vagy gyorsan, helyben az átlátszó ablak képpontjaiba (layer_compositor).
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import AppConfig
from .frosted_cover import SIDE_FEATHER_RATIO, cover_area
from .ocr import Box
from .overlay import (MAX_TEXT_WIDTH_RATIO, RenderedSubtitle, beside_original_position, centered_position,
                      hex_to_rgb, render_subtitle, subtitle_position)

MIN_FONT_PX = 12
ORIGINAL_COVER_OPACITY = 255
LINE_COVER_FEATHER_PX = 6  # soronkénti takarásnál keskeny a szél: a szomszéd angol sorba ne lógjon bele
LINE_MAX_WIDTH_RATIO = 0.9  # a soronkénti fordítás ne törjön két sorba: rálógna a szomszédjára
MULTI_ROW_HEIGHT_RATIO = 0.055  # ennél magasabb angol sor (a képmagassághoz mérve) két képernyősort foglal
MAX_OPACITY = 255.0
MIN_COVER_FEATHER_PX = 8  # a takarás lágy szélének szélessége legalább ennyi
COVER_PADDING_PX = 4  # a lágy szél a betűkön kívül kezdődjön


@dataclass(frozen=True)
class ShownLine:
    text: str
    anchor: Box  # az angol sor helye, amelynek ez a fordítása


@dataclass(frozen=True)
class Shown:
    text: str
    anchor: Box  # az eredeti angol felirat helye
    lines: tuple[ShownLine, ...] = ()  # soronkénti fordítás; ha üres, a text egy tömbben áll az anchor helyén
    hidden: tuple[Box, ...] = ()  # még le nem fordított angol sorok: csak eltakarjuk, hogy ne villanjanak fel


@dataclass(frozen=True)
class LayerView:
    """Amit a tervhez a képről és a panel pillanatnyi állapotáról tudni kell."""
    width: int
    height: int
    hide_original: bool
    backdrop_area: Box | None = None  # a takarás nélküli játékkép helye; None: nincs ilyen kép
    custom_center: tuple[int, int] | None = None  # a felhasználó ide húzta a panelt
    is_dragging: bool = False
    is_preview: bool = False  # mintaszöveg: az angolt nem takarjuk


@dataclass(frozen=True)
class CoverJob:
    """Az angol betűk eltüntetése a játékkép elmosott másával."""
    area: Box
    feather: int


@dataclass(frozen=True)
class BoxJob:
    """Egyszínű sötét doboz (a képre vágva)."""
    box: Box
    opacity: int


@dataclass(frozen=True)
class TextJob:
    rendered: RenderedSubtitle
    x: int
    y: int


@dataclass(frozen=True)
class LayerPlan:
    covers: tuple[CoverJob | BoxJob, ...] = ()
    cover_strength: float = 1.0  # a panel átlátszóságát követi (frosted_layer strength)
    panel: BoxJob | None = None  # a soronkénti szövegek közös sötét panelje
    texts: tuple[TextJob, ...] = ()
    last_rect: tuple[int, int, int, int] | None = None  # a kirajzolt panel (x, y, szélesség, magasság)
    last_cover: Box | None = None  # az eltakart terület
    outline: Box | None = None  # húzás közben a panel kerete


def union(boxes: list[Box]) -> Box:
    return Box(min(b.left for b in boxes), min(b.top for b in boxes),
               max(b.right for b in boxes), max(b.bottom for b in boxes))


def clip(box: Box, width: int, height: int) -> Box:
    return Box(max(box.left, 0), max(box.top, 0), min(box.right, width), min(box.bottom, height))


def make_plan(config: AppConfig, view: LayerView, shown: Shown | None) -> LayerPlan | None:
    """A kirajzolás terve; None, ha nincs mit rajzolni."""
    if shown is None:
        return None
    covers, bounds = ((), [])
    if view.hide_original and not view.is_preview:
        covers, bounds = _cover_jobs(view, shown)
    strength = config.panel_opacity / MAX_OPACITY
    last_cover = union(bounds) if bounds else None
    if not shown.text:
        return LayerPlan(covers, strength, last_cover=last_cover)  # csak takarás: a fordítás még úton van
    panel, texts, rect = _text_jobs(config, view, shown)
    outline = Box(rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]) if view.is_dragging else None
    return LayerPlan(covers, strength, panel, texts, rect, last_cover, outline)


def _cover_jobs(view: LayerView, shown: Shown) -> tuple[tuple[CoverJob | BoxJob, ...], list[Box]]:
    """A takarások és az általuk érintett területek (a last_cover-hez)."""
    anchors: list[tuple[Box, int | None]] = []
    if not shown.lines and shown.text:
        anchors.append((shown.anchor, None))
    # Soronként, keskeny széllel: a szomszéd sorba nem lóg bele
    anchors += [(line.anchor, LINE_COVER_FEATHER_PX) for line in shown.lines]
    anchors += [(box, LINE_COVER_FEATHER_PX) for box in shown.hidden]
    jobs: list[CoverJob | BoxJob] = []
    bounds: list[Box] = []
    for anchor, line_feather in anchors:
        job = _cover_job(view, anchor, line_feather)
        jobs.append(job)
        bounds.append(job.area if isinstance(job, CoverJob) else anchor)
    return tuple(jobs), bounds


def _cover_job(view: LayerView, anchor: Box, line_feather: int | None) -> CoverJob | BoxJob:
    backdrop = view.backdrop_area
    if backdrop is not None:
        # Átméretezés után a korábbi sáv kilóghat a képből: csak a közös részt takarjuk
        limits = Box(max(backdrop.left, 0), max(backdrop.top, 0),
                     min(backdrop.right, view.width), min(backdrop.bottom, view.height))
        feather = line_feather or max(MIN_COVER_FEATHER_PX, anchor.height // 2)
        area = cover_area(anchor, feather + COVER_PADDING_PX, limits,
                          side_margin=feather * SIDE_FEATHER_RATIO + COVER_PADDING_PX)
        if area.width > 0 and area.height > 0:
            return CoverJob(area, feather)
    return BoxJob(clip(anchor, view.width, view.height), ORIGINAL_COVER_OPACITY)


def _text_jobs(config: AppConfig, view: LayerView,
               shown: Shown) -> tuple[BoxJob | None, tuple[TextJob, ...], tuple[int, int, int, int]]:
    font_size = max(MIN_FONT_PX, int(view.height * config.font_size_ratio))
    if view.custom_center is None and view.hide_original and shown.lines:
        return _line_by_line_jobs(config, view, shown, font_size)
    rendered, (x, y) = _block_text(config, view, shown, font_size)
    return None, (TextJob(rendered, x, y),), (x, y, rendered.width, rendered.height)


def _line_by_line_jobs(config: AppConfig, view: LayerView, shown: Shown,
                       font_size: int) -> tuple[BoxJob, tuple[TextJob, ...], tuple[int, int, int, int]]:
    """Minden fordított sor a saját angol sora helyére kerül; a panel egy közös sötét tömb."""
    texts = []
    for line in shown.lines:
        # Ha az angol mondat két képernyősorban állt, a fordítás is törhet: nem fut ki a kép szélére
        is_multi_row = line.anchor.height > view.height * MULTI_ROW_HEIGHT_RATIO
        max_width = MAX_TEXT_WIDTH_RATIO if is_multi_row else LINE_MAX_WIDTH_RATIO
        rendered = render_subtitle(line.text, config.font_path, font_size, int(view.width * max_width), (0, 0), 0,
                                   hex_to_rgb(config.speaker_color))
        texts.append(TextJob(rendered, *subtitle_position(rendered, line.anchor, view.width, view.height)))
    left, top = min(job.x for job in texts), min(job.y for job in texts)
    right = max(job.x + job.rendered.width for job in texts)
    bottom = max(job.y + job.rendered.height for job in texts)
    panel = BoxJob(clip(Box(left, top, right, bottom), view.width, view.height), config.panel_opacity)
    return panel, tuple(texts), (left, top, right - left, bottom - top)


def _block_text(config: AppConfig, view: LayerView, shown: Shown,
                font_size: int) -> tuple[RenderedSubtitle, tuple[int, int]]:
    """A teljes szöveg egy panelben: a felhasználó által választott helyen, vagy az angol feliratnál."""
    speaker = hex_to_rgb(config.speaker_color)
    max_width = int(view.width * MAX_TEXT_WIDTH_RATIO)
    if view.custom_center is not None:
        rendered = render_subtitle(shown.text, config.font_path, font_size, max_width, (0, 0),
                                   config.panel_opacity, speaker)
        return rendered, centered_position(rendered, *view.custom_center, view.width, view.height)
    anchor = shown.anchor
    min_panel = (anchor.width + font_size, anchor.height + font_size // 2) if view.hide_original else (0, 0)
    rendered = render_subtitle(shown.text, config.font_path, font_size, max_width, min_panel,
                               config.panel_opacity, speaker)
    if view.hide_original:
        return rendered, subtitle_position(rendered, anchor, view.width, view.height)
    return rendered, beside_original_position(rendered, anchor, view.width, view.height, gap=font_size // 2)
