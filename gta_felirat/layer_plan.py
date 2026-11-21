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
from .overlay import (LINE_SPACING_RATIO, PANEL_PADDING_RATIO, RenderedSubtitle, SpeakerColors, beside_original_position,
                      centered_position, hex_to_rgb, render_subtitle, subtitle_position)

MIN_FONT_PX = 12
ORIGINAL_COVER_OPACITY = 255
LINE_COVER_FEATHER_PX = 6  # soronkénti takarásnál keskeny a szél: a szomszéd angol sorba ne lógjon bele
LINE_MAX_WIDTH_RATIO = 0.9  # a soronkénti fordítás ne törjön két sorba: rálógna a szomszédjára
MULTI_ROW_HEIGHT_RATIO = 0.055  # ennél magasabb angol sor (a képmagassághoz mérve) két képernyősort foglal
MAX_OPACITY = 255.0
MIN_COVER_FEATHER_PX = 8  # a takarás lágy szélének szélessége legalább ennyi
COVER_PADDING_PX = 4  # a lágy szél a betűkön kívül kezdődjön
PREVIEW_ROWS = 3  # a Minta módban ennyi sor magas a panel: így látszik és fogható meg a teljes területe
CLICKABLE_ALPHA = 1  # az átlátszó rétegen a teljesen átlátszó képpontra kattintás átmegy a játékra
# A mintapanel ennyire áll a sávtól: a panel alja és a sáv teteje külön-külön megfogható maradjon
PREVIEW_GAP_PX = 40


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
    speaker_colors: SpeakerColors = ()  # (név, RGB) a játékból mintázva; üres: mindenki a config színével


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
    is_framed = view.is_dragging or view.is_preview  # a Minta módban is látszik, meddig ér a panel
    outline = Box(rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]) if is_framed else None
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
    if view.is_preview:
        rendered, (x, y) = _preview_text(config, view, shown, font_size)
        return None, (TextJob(rendered, x, y),), (x, y, rendered.width, rendered.height)
    # Az angol helyére írt fordítás soronként a saját angol sorára kerül, akkor is, ha a panelt
    # korábban elhúzták: különben a fordítás a panelen, a takarás az angol sorokon állna, két helyen
    if view.hide_original and shown.lines:
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
        max_width = (config.panel_width_ratio if is_multi_row
                     else max(LINE_MAX_WIDTH_RATIO, config.panel_width_ratio))
        rendered = render_subtitle(line.text, config.font_path, font_size,
                                   _wrap_width(int(view.width * max_width), font_size), (0, 0), 0,
                                   hex_to_rgb(config.speaker_color), shown.speaker_colors)
        texts.append(TextJob(rendered, *subtitle_position(rendered, line.anchor, view.width, view.height)))
    # A szegély miatt a szomszéd sorok kirajzolása kicsit fedheti egymást; ennél többet nem
    texts = _stacked(texts, view.height, allowed_overlap=font_size // 2)
    left, top = min(job.x for job in texts), min(job.y for job in texts)
    right = max(job.x + job.rendered.width for job in texts)
    bottom = max(job.y + job.rendered.height for job in texts)
    top, bottom = _grown_to(top, bottom, _min_height(config, view), view.height)
    panel = BoxJob(clip(Box(left, top, right, bottom), view.width, view.height), config.panel_opacity)
    return panel, tuple(texts), (left, top, right - left, bottom - top)


def _stacked(texts: list[TextJob], frame_height: int, allowed_overlap: int) -> list[TextJob]:
    """Több beszélő sora ne lógjon egymásra: a lejjebb álló sor kerül az előző alá.

    Minden sor a saját angol sora helyén indul; csak akkor mozdul, ha a fölötte álló fordítás (pl. két
    sorba tört magyar mondat) rálógna. Ha így a kép aljáról kicsúszna, az egész csoport feljebb kerül.
    """
    ordered = sorted(texts, key=lambda job: job.y)
    placed: list[TextJob] = []
    for job in ordered:
        if placed:
            previous = placed[-1]
            lowest_top = previous.y + previous.rendered.height - allowed_overlap
            job = TextJob(job.rendered, job.x, max(job.y, lowest_top))
        placed.append(job)
    for index in range(len(placed) - 1, -1, -1):
        job = placed[index]
        highest_top = frame_height - job.rendered.height
        if index + 1 < len(placed):
            following = placed[index + 1]
            highest_top = min(highest_top, following.y - job.rendered.height + allowed_overlap)
        placed[index] = TextJob(job.rendered, job.x, max(min(job.y, highest_top), 0))
    return placed


def _block_text(config: AppConfig, view: LayerView, shown: Shown,
                font_size: int) -> tuple[RenderedSubtitle, tuple[int, int]]:
    """A teljes szöveg egy panelben: a felhasználó által választott helyen, vagy az angol feliratnál."""
    speaker = hex_to_rgb(config.speaker_color)
    max_width = _wrap_width(int(view.width * config.panel_width_ratio), font_size)
    if view.custom_center is not None:
        rendered = render_subtitle(shown.text, config.font_path, font_size, max_width,
                                   (0, _min_height(config, view)), config.panel_opacity, speaker,
                                   shown.speaker_colors)
        return rendered, centered_position(rendered, *view.custom_center, view.width, view.height)
    anchor = shown.anchor
    min_panel = (anchor.width + font_size, anchor.height + font_size // 2) if view.hide_original else (0, 0)
    min_panel = (min_panel[0], max(min_panel[1], _min_height(config, view)))
    rendered = render_subtitle(shown.text, config.font_path, font_size, max_width, min_panel,
                               config.panel_opacity, speaker, shown.speaker_colors)
    if view.hide_original:
        return rendered, subtitle_position(rendered, anchor, view.width, view.height)
    return rendered, beside_original_position(rendered, anchor, view.width, view.height, gap=font_size // 2)


def _preview_text(config: AppConfig, view: LayerView, shown: Shown,
                  font_size: int) -> tuple[RenderedSubtitle, tuple[int, int]]:
    """A Minta mód panelje a beállított teljes szélességben és néhány sor magasan, hogy látszódjon, meddig ér
    a fordítás, és bárhol megfogható legyen (átlátszó panelnél is: alig látható, de kattintható alappal)."""
    width = int(view.width * config.panel_width_ratio)
    height = _min_height(config, view) or int(font_size * (1 + LINE_SPACING_RATIO)) * PREVIEW_ROWS
    rendered = render_subtitle(shown.text, config.font_path, font_size, _wrap_width(width, font_size), (width, height),
                               max(config.panel_opacity, CLICKABLE_ALPHA), hex_to_rgb(config.speaker_color))
    if view.custom_center is not None:
        return rendered, centered_position(rendered, *view.custom_center, view.width, view.height)
    # A sáv mellé, nem a közepére: így a sáv belseje is megfogható, és az OCR is látja a sávot
    return rendered, beside_original_position(rendered, shown.anchor, view.width, view.height,
                                              gap=max(font_size // 2, PREVIEW_GAP_PX))


def _wrap_width(panel_width: int, font_size: int) -> int:
    """Ilyen szélesen törhet a szöveg, hogy a belső margóval együtt se legyen szélesebb a panelnél."""
    return max(panel_width - 2 * int(font_size * PANEL_PADDING_RATIO), font_size)


def _min_height(config: AppConfig, view: LayerView) -> int:
    """A beállított legkisebb panelmagasság képpontban (0: a szöveghez igazodik)."""
    return int(view.height * config.panel_height_ratio) if config.panel_height_ratio else 0


def _grown_to(top: int, bottom: int, min_height: int, frame_height: int) -> tuple[int, int]:
    """A sáv (top, bottom) legalább min_height magasra nő, a közepe körül, a képen belül."""
    if bottom - top >= min_height:
        return top, bottom
    new_top = max(min((top + bottom) // 2 - min_height // 2, frame_height - min_height), 0)
    return new_top, min(new_top + min_height, frame_height)
