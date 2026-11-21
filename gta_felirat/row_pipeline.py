"""A feliratfeldolgozás soralapú (képpontokból vezérelt) útja.

A régi út (SubtitlePipeline.submit_band / poll) minden olvasásnál az egész sávban keres és olvas
(~300 ms). Itt a sávot egy gyors sorkereső figyeli (row_tracker), és:
- az újonnan megjelent, feliratszerű sort azonnal eltakarjuk, még mielőtt elolvastuk volna
  (az olvasásig, legfeljebb UNREAD_HIDE_SECONDS-ig);
- csak az új képpontú sort olvassuk be; második olvasás helyett az a megerősítés, hogy a sor
  képpontjai STABLE_CAPTURES rögzítésen át nem változtak;
- a lefordított sor fordítása olvasás nélkül követi a sort, amikor a játék feljebb görgeti.
Hogy melyik sor felirat, hogyan fordítjuk, hová kerül a fordítás, és meddig takarjuk a még le nem
fordított sort, az ugyanaz, mint a régi úton.

A sáv egészét figyelve (whole_band_text) a kép közepétől távoli szöveg is felirat lehet: a játék máshová
is írhat szöveget, és egyszerre több helyre is. Az ilyen szöveget helyenként (tömbönként: a két sorba
tördelt mondat egy megszólalás) a középen álló felirattól függetlenül szűrjük, ugyanazzal a
szövegszűrő-lánccal, de csak „Név: szöveg” megszólalás lehet (a kijelző oldalt álló szövege, pl. helynév
vagy hívásértesítés, nem az), és csak stabil képpontokkal vagy álló képen kerül a felirat mellé: így a
bizonytalansága nem tartja vissza a középső felirat fordítását. Ha átmegy, a saját angol sorára fordítjuk,
akár a középső felirat mellé is. Olvasás előtt sosem takarjuk: a kereső sosem biztos benne (row.sure False).

Szálak: a RowPipeline minden metódusát a megjelenítési (Tk) szál hívja, és csak ez a szál írja az
állapotát. A munkaszálak (handoff.BandWatcher, handoff.RowRecognizer) megváltoztathatatlan
eredményeket tesznek a handoff.Inbox-ba; a Tk-szál kiüríti, és átadja a receive()-nek. Más szál
egyedül a konstruktornak adott `wake`-et hívja (a fordító szál jelzi vele, hogy kész a fordítás):
annak szálbiztosnak kell lennie, és nem blokkolhat (a handoff.TkWaker.wake ilyen).

Idő: a `now` és a pillanatképek ideje (captured_at, first_seen) ugyanabból az órából való
(time.monotonic()).

A rögzített sávban nem látszhat a saját feliratrétegünk (az ablak saját tartalmát rögzítjük),
ezért itt nincs „a saját fordításunkat olvastuk vissza” védelem. A régi út ugyanezen a példányon
megmarad tartaléknak, de a kettő nem megy egyszerre: a régire váltás előtt suspend_rows() kell.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from .config import AppConfig
from .fragment_filter import DIALOGUE_BOTTOM_RATIO, MIN_WHITE_FRACTION, Fragment, FrameCenter, select_subtitle
from .line_placement import PENDING_HIDE_SECONDS, build_shown, match_lines
from .objective_line import ObjectiveLine
from .ocr import (Box, OcrLine, SubtitleReader, clean_ocr_text, is_spoken_line, looks_like_subtitle, spoken_lines,
                  union_box)
from .row_finder import band_edge_margin
from .row_tracker import STABLE_CAPTURES, BandSnapshot, RowObs, RowText, is_same_place, overlaps
from .subtitle_pipeline import SubtitlePipeline
from .subtitle_tracker import NewSubtitle, SubtitleCleared
from .translation_service import TranslationService

EAGER_HIDE_SECONDS = 0.35  # az előző sortól örökölt (még nem olvasott) szöveg eddig érvényes
# Az olvasatlan, biztos párbeszédsort az olvasásig takarjuk, de legfeljebb ameddig a le nem fordított sort
# is (a felismerés terhelt gépen 0,5-1 mp is lehet; ennyi ideig nem villanhat ki az angol szöveg)
UNREAD_HIDE_SECONDS = PENDING_HIDE_SECONDS
# Időfüggő állapotnál (késő fordítás takarása, eltűnés utáni várakozás) ilyen sűrűn nézünk rá, ha a kép nem változik
RECHECK_SECONDS = 0.1
# Ha ennyi ideje nem jött újabb kép, a mostani áll (a rögzítés csak változáskor ad képet): az egyetlen képen
# látott sort sem kell tovább megerősíteni. A leglassabb képforrás két képe között ~0,07 mp telik el
STILL_SECONDS = 0.15
NEVER = float("inf")
PLACE_SLACK_PX = 6  # ennyit ingadozhat a kereső doboza ugyanazon a soron
# Az örökölt szöveg ennyi ideig érvényes a lánc legutóbbi valódi olvasása után: mozgó háttér előtt a sor szinte
# minden képen új azonosítót kap, a szövegét a késve érkező olvasások frissítik (lásd _take_texts)
CHAIN_READ_SECONDS = 1.0

# (az elfogadott feliratsorok, stabilak-e a képpontjaik, az olvasatlan sorok: azonosító, hely, előre takarható-e)
_State = tuple[tuple[OcrLine, ...], bool, tuple[tuple[int, Box, bool], ...]]

logger = logging.getLogger(__name__)


@dataclass
class RowPathStats:
    """Számlálók méréshez és hibakereséshez; csak a Tk-szál írja."""

    state_updates: int = 0  # ennyiszer változott a sorok vagy a szövegeik állása
    idle_updates: int = 0  # ennyi érkezés (pillanatkép, szöveg) nem változtatott semmin
    stale_texts: int = 0  # közben eltűnt sor szövege: eldobtuk
    stale_snapshots: int = 0  # más ablakméretre készült pillanatkép: eldobtuk
    inherited_texts: int = 0  # ennyi új sor örökölte az előtte ugyanott álló sor szövegét


def _lost_its_name(text: str) -> bool:
    """A sor olvasata kettősponttal kezdődik: a beszélő neve lemaradt a dobozból."""
    return text.lstrip().startswith(":")


def _plain(text: str) -> str:
    """Összevetéshez: kisbetűs, egyszeres szóközökkel (a sortörés is szóköz)."""
    return " ".join(text.lower().split())


def _lies_within(inner: Box, outer: Box) -> bool:
    """A belső doboz (néhány pixel eltéréssel) a külsőn belül áll."""
    return (inner.left >= outer.left - PLACE_SLACK_PX and inner.right <= outer.right + PLACE_SLACK_PX
            and inner.top >= outer.top - PLACE_SLACK_PX and inner.bottom <= outer.bottom + PLACE_SLACK_PX)


def _is_stable(row: RowObs) -> bool:
    """A sor képpontjai már elég rögzítésen át nem változtak: a szövege második olvasás nélkül is végleges."""
    return row.stable_captures >= STABLE_CAPTURES


def free_blocks(fragments: list[Fragment], kept: list[Fragment], center: FrameCenter) -> list[list[Fragment]]:
    """A kép közepétől távoli, a középső feliratba nem tartozó szöveg helyenként, tömbönként.

    Ugyanazok a szűrők, mint a select_subtitle-ben (feliratszín, kijelző-felirat, a küldetésutasítás
    zónája), és a sáv széle által elvágott szöveg (csonka szavak) kimarad. Egy tömb egy helyen álló szöveg:
    a két sorba tördelt mondat egy tömb, két helyen álló szöveg két tömb. A középen álló, de a középső
    tömbbe nem tartozó sort ugyanúgy elhagyjuk, mint eddig.
    """
    taken = {id(fragment) for fragment in kept}
    anywhere = replace(center, anywhere=True)
    pool = [fragment for fragment in fragments if id(fragment) not in taken
            and not center.is_near(fragment.box) and not center.touches_band_edge(fragment.box)]
    blocks = []
    while block := select_subtitle(pool, anywhere):  # minden kör legalább egy darabot kivesz: véges
        blocks.append(block)
        chosen = {id(fragment) for fragment in block}
        pool = [fragment for fragment in pool if id(fragment) not in chosen]
    return blocks


class RowPipeline(SubtitlePipeline):
    """SubtitlePipeline, amelyet a sorkövető pillanatképei hajtanak; a régi út (submit_band / poll) is megmarad."""

    def __init__(self, config: AppConfig, reader: SubtitleReader, translation: TranslationService, *,
                 wake: Callable[[], None] | None = None) -> None:
        """wake: szálbiztos, nem blokkoló jelzés a Tk-szálnak; a fordító szál hívja, amikor kész a fordítás."""
        super().__init__(config, reader, translation)
        self._wake = wake
        self._expected_size: tuple[int, int] | None = None
        self._snapshot: BandSnapshot | None = None  # None: a soralapú út most nem megy
        self._rows: tuple[RowObs, ...] = ()
        self._texts: dict[int, RowText] = {}
        # Eltűnt sor → a sor, amelyik a helyére került (lásd _texts_for); a késve érkező olvasás az utódé lesz
        self._heirs: dict[int, int] = {}
        self._read_at: dict[int, float] = {}  # sor → mikor érkezett az a valódi olvasás, amelyből a szövege van
        self._white_rows: set[int] = set()  # a feliratszínű sorok és utódaik (lásd _whiteness)
        self._last_state: _State | None = None
        self._is_on_screen = False  # a legutóbbi kép szerint látszik a felirat
        self._next_deadline = NEVER
        # A küldetésutasítás sora: a párbeszédtől függetlenül, a saját sorára fordítjuk (translate_objectives)
        self._objective = ObjectiveLine(self._translation.translate, self._translate_pool,
                                        config.similarity_threshold, wake=wake)
        self.stats = RowPathStats()

    @property
    def snapshot(self) -> BandSnapshot | None:
        """A legutóbb elfogadott pillanatkép: a mostani `shown` ebből készült, a takarás háttere is ez legyen."""
        return self._snapshot

    def update_config(self, config: AppConfig) -> None:
        super().update_config(config)
        self._last_state = None  # a szűrők változhattak: a következő kiértékelés mindent újra alkalmaz

    def set_frame_size(self, frame_size: tuple[int, int]) -> None:
        """A játékablak mostani mérete (a Tk-szál tudja): a más méretre készült pillanatképet eldobjuk (amíg nem
        hívják, mindet elfogadjuk). Méretváltáskor a sorok helyét elfelejtjük; a felirat szövege és a kész
        fordítás megmarad, így semmit sem fordítunk újra."""
        self._expected_size = frame_size
        if self._snapshot is not None and self._snapshot.frame_size != frame_size:
            self._forget_places()

    def suspend_rows(self) -> None:
        """A soralapú út leáll (pl. elakadt a figyelő): a sorok helyét elfelejtjük, a felirat és a fordítása marad.

        Ezután a régi, teljes olvasású út (submit_band / poll) ugyanezen a példányon folytathatja; a következő
        elfogadott pillanatkép újraindítja a soralapú utat. A már felismert sorszövegek megmaradnak: a felismerő
        nem olvassa újra azt a sort, amelyik a visszatéréskor még a képen van."""
        self._snapshot, self._rows, self._heirs, self._read_at, self._white_rows = None, (), {}, {}, set()
        self._last_state, self._is_on_screen, self._next_deadline = None, False, NEVER
        self._objective.see(None, 0.0)  # a régi út nem keres küldetéssort
        if self._block is None:  # a takarásokat csak a soralapú út venné le: ami kell, azt a régi út újra kiteszi
            self.shown = build_shown(match_lines(self._translated, self._live_lines)[0], [], self.shown,
                                     speaker_colors=self._live_speaker_colors())

    def receive(self, snapshot: BandSnapshot | None, texts: Sequence[RowText], now: float) -> bool:
        """A munkaszálak eredményei (handoff.Inbox.drain()) és az idő múlása. True, ha a kiírt felirat megváltozott.

        Előbb a pillanatkép, aztán a szövegek: a soroknak előbb kell létezniük, mint a szövegüknek.
        """
        before = self.shown
        if self._is_on_screen:
            self._tracker.touch(now)  # újabb képig a legutóbbi érvényes: a felirat eddig a pillanatig látszott
        has_news = self._take_snapshot(snapshot)
        has_news = self._take_texts(texts, now) or has_news
        current = self._snapshot
        if current is not None and (has_news or self._last_state is None or now >= self._next_deadline):
            self._refresh(current.captured_at, now)
        self._collect_translation(now)
        if self._objective.collect() and self._snapshot is not None:
            self._follow_live_lines(now)
        return self.shown is not before

    def pump(self, now: float) -> bool:
        """Csak az idő múlt (a hívó rendes időzítője): lejáró takarások, eltűnt felirat, kész fordítás."""
        return self.receive(None, (), now)

    # --- ami a munkaszálaktól érkezik ---------------------------------------------------
    def _take_snapshot(self, snapshot: BandSnapshot | None) -> bool:
        if snapshot is None:
            return False
        if self._expected_size is not None and snapshot.frame_size != self._expected_size:
            self.stats.stale_snapshots += 1
            logger.debug("snapshot for %s dropped: the window is %s now", snapshot.frame_size, self._expected_size)
            return False
        if self._snapshot is not None and snapshot.frame_size != self._snapshot.frame_size:
            self._forget_places()
        self._texts = self._texts_for(snapshot.rows)
        alive = {row.row_id for row in snapshot.rows}
        self._read_at = {row_id: moment for row_id, moment in self._read_at.items() if row_id in alive}
        self._snapshot, self._rows, self._frame_size = snapshot, snapshot.rows, snapshot.frame_size
        return True

    def _texts_for(self, rows: tuple[RowObs, ...]) -> dict[int, RowText]:
        """A megmaradt sorok szövegei; az eltűnt sor pontos helyén megjelent új sor a felismerésig örökli a szövegét.

        Ilyenkor többnyire ugyanaz a szöveg kapott új azonosítót (megváltozott mögötte a háttér): ne tűnjön el miatta
        egy pillanatra a fordítása, és ne takarjuk el újra azt a sort, amelyről már tudjuk, hogy nem felirat."""
        alive = {row.row_id for row in rows}
        known = {row.row_id for row in self._rows}
        texts = {row_id: text for row_id, text in self._texts.items() if row_id in alive}
        orphans = [row for row in self._rows if row.row_id not in alive]
        heirs = {}
        for row in rows:
            if row.row_id in known:
                continue
            orphan = next((old for old in orphans if is_same_place(old.box, row.box)), None)
            if orphan is None:
                continue
            heirs[orphan.row_id] = row.row_id
            left_behind = self._texts.get(orphan.row_id)
            if left_behind is not None:
                texts[row.row_id] = replace(left_behind, row_id=row.row_id, provisional=True)
                if orphan.row_id in self._read_at:
                    self._read_at[row.row_id] = self._read_at[orphan.row_id]
                self.stats.inherited_texts += 1
        # Az öröklési lánc csak addig érdekes, amíg a vége látszik: a régebbi sorok olvasása elavult
        chains = {**self._heirs, **heirs}
        self._heirs = {old: new for old, new in chains.items() if self._heir_of(old, chains) in alive}
        # Világos, mozgó háttér előtt a fehér betűk aránya képről képre 1-10% között ugrál: ami ugyanazon a helyen
        # egyszer már feliratszínű volt, az utódja is az marad (különben a felirat sorai ki-be esnének belőle)
        self._white_rows = ({row.row_id for row in rows if row.whiteness >= MIN_WHITE_FRACTION}
                            | {row_id for row_id in self._white_rows if row_id in alive}
                            | {new for old, new in heirs.items() if old in self._white_rows})
        return texts

    def _whiteness(self, row: RowObs) -> float:
        """A sor feliratszínűsége, az elődjeitől örökölve (lásd _texts_for)."""
        return max(row.whiteness, MIN_WHITE_FRACTION) if row.row_id in self._white_rows else row.whiteness

    def _heir_of(self, row_id: int, heirs: dict[int, int] | None = None) -> int:
        """Az a sor, amelyik (akár több lépésben) az eltűnt sor helyére került; ha nincs ilyen, maga a sor."""
        heirs = self._heirs if heirs is None else heirs
        seen = {row_id}
        while row_id in heirs and heirs[row_id] not in seen:
            row_id = heirs[row_id]
            seen.add(row_id)
        return row_id

    def _take_texts(self, texts: Sequence[RowText], now: float) -> bool:
        alive = {row.row_id for row in self._rows}
        # A színes beszélőnév világos háttér előtt egy-egy képen kimarad a sor dobozából: a név nélküli olvasat
        # („: He's too…”) nem a sor szövege, a sor olvasatlan marad (takarva), amíg a nevével együtt meg nem találjuk
        texts = [item for item in texts if not _lost_its_name(item.text)]
        fresh = {item.row_id: item for item in texts if item.row_id in alive}
        for item in texts:
            heir = self._heir_of(item.row_id)
            if item.row_id in alive or heir not in alive or not item.is_legible(self._config.min_text_score):
                continue
            current = fresh.get(heir) or self._texts.get(heir)
            if current is None or current.provisional:
                # Mozgó háttér előtt a sor szinte minden képen új azonosítót kap, és az olvasása mire elkészül, a
                # sor már eltűnt: a helyére került utód kapja (ideiglenesen, a saját olvasásáig)
                fresh[heir] = replace(item, row_id=heir, provisional=True)
        stale = len([item for item in texts if item.row_id not in alive])
        if stale:  # a sor közben eltűnt (vagy más méretű lett a kép): az eredménye elavult
            self.stats.stale_texts += stale
            logger.debug("%d row text(s) dropped: the rows are gone", stale)
        for item in fresh.values():
            self._palette.learn(item.speaker_colors)
            self._read_at[item.row_id] = now
        self._texts = {**self._texts, **fresh}
        return bool(fresh)

    def _forget_places(self) -> None:
        """Más méretű lett a kép: a régi sorok és dobozok sehová sem mutatnak."""
        self._live_lines, self._last_box = (), None
        self.suspend_rows()

    # --- sorokból felirat ---------------------------------------------------------------
    def _refresh(self, pictured_at: float, now: float) -> None:
        """A sorok és szövegeik mostani állásából felirat lesz; pictured_at: a legutóbbi kép ideje."""
        self._texts = self._unexpired_texts(now)
        is_still = now - pictured_at >= STILL_SECONDS
        lines, used, is_free_waiting = self._assemble(is_still)
        if self._is_partly_unread(now):
            lines = self._live_lines  # a felirat egyik sora még olvasatlan: a maradékából nem lesz új felirat
        confirmed = is_still or all(_is_stable(row) for row in used)
        unread = tuple((row.row_id, row.box, self._can_hide_early(row))
                       for row in self._rows if row.row_id not in self._texts)
        state = (lines, confirmed, unread)
        has_new_objective = self._objective.see(self._objective_line(), now)
        if state != self._last_state:
            self._last_state = state
            self._apply_rows(lines, confirmed, now)
        else:
            self._let_time_pass(bool(lines), now)
        if has_new_objective:
            self._follow_live_lines(now)
        self._is_on_screen = bool(lines)  # csak a most is látszó sorokból áll (lásd _assemble)
        deadline = self._deadline(state, pictured_at, now, is_free_waiting)
        # A semmit sem változtató képkocka (pl. 30 kép/mp-es videóablak álló jeleneten) nem tolhatja
        # későbbre a még esedékes ellenőrzést, különben az eltűnt felirat fordítása sosem tűnne el
        is_pending = now < self._next_deadline
        self._next_deadline = min(deadline, self._next_deadline) if is_pending else deadline

    def _unexpired_texts(self, now: float) -> dict[int, RowText]:
        """Az örökölt szöveg csak addig érvényes, ameddig egy olvasatlan sort eltakarnánk, vagy amíg az
        öröklési lánc legutóbbi olvasása friss."""
        expired = {row.row_id for row in self._rows if now >= self._inherited_until(row)}
        return {row_id: text for row_id, text in self._texts.items()
                if not (text.provisional and row_id in expired)}

    def _inherited_until(self, row: RowObs) -> float:
        """Eddig érvényes a sor örökölt szövege."""
        return max(row.first_seen + EAGER_HIDE_SECONDS,
                   self._read_at.get(row.row_id, float("-inf")) + CHAIN_READ_SECONDS)

    def _assemble(self, is_still: bool) -> tuple[tuple[OcrLine, ...], tuple[RowObs, ...], bool]:
        """(az elfogadott feliratsorok, a sorok, amelyekből lettek, vár-e máshol álló szöveg a stabilitásra):
        a SubtitleReader.read_band szűrőivel.

        Előbb a középen álló felirat (mint eddig), aztán ha a sáv egészét figyeljük, a máshol álló szöveg
        tömbönként: ez a középső felirat mellé, a saját helyére kerül. is_still: áll a kép (lásd _refresh)."""
        min_score = self._config.min_text_score
        # Csak a most is látszó sor számít: a türelmi időben csak emlékezett sor (missing > 0) nem tartja
        # életben a feliratot (ha utána nem jön több kép, mert áll a jelenet, a fordítás örökre kint maradna).
        # A szövegét megtartjuk: ha a sor visszatér, nem olvassuk újra
        legible = {row.box: row for row in self._rows
                   if row.missing == 0 and row.row_id in self._texts
                   and self._texts[row.row_id].is_legible(min_score)}
        fragments = [Fragment(self._texts[row.row_id].text.strip(), box, self._whiteness(row))
                     for box, row in legible.items()]
        center = self._frame_center()
        kept = select_subtitle(fragments, center)
        lines = self._subtitle_lines(kept, is_tv_text=bool(kept) and all(legible[part.box].dim for part in kept))
        used = list(kept) if lines else []
        is_free_waiting = False
        if self._config.whole_band_text:
            for block in free_blocks(fragments, kept, center):
                free_lines = self._subtitle_lines(block, must_be_spoken=True)
                if not free_lines:
                    continue
                if not (is_still or all(_is_stable(legible[fragment.box]) for fragment in block)):
                    is_free_waiting = True  # a következő stabil képen, vagy ha áll a kép, kerül ki
                    continue
                lines += free_lines
                used += block
        return lines, tuple(legible[fragment.box] for fragment in used), is_free_waiting

    def _is_partly_unread(self, now: float) -> bool:
        """A kiírt felirat egy sora átmenetileg nem olvasható: új dobozt kapott (világos, mozgó háttér előtt a
        kereső másképp méri, ezért a szövegét sem örökli) és még olvasatlan, vagy egy-két képen nem látszott
        (türelmi idejét tölti); a többi sora viszont változatlanul ott áll.

        Ilyenkor a látható maradék (pl. a második sor: „most evenings.”) nem új felirat: addig a kiírt felirat
        marad, az olvasatlan sor takarva. Ha semmi sem maradt a régiből, a szokásos szabály dönt."""
        current = _plain(self._tracker.current)
        if not current or not self._live_lines:
            return False
        nearby = [row for row in self._rows if any(overlaps(row.box, live.box) for live in self._live_lines)]
        read = [row for row in nearby if row.missing == 0 and row.row_id in self._texts]
        gaps = [row for row in nearby if row.missing > 0 or (
            row.row_id not in self._texts and self._can_hide_early(row) and now - row.first_seen < UNREAD_HIDE_SECONDS)]
        if not read or not gaps:
            return False
        # A már olvasott sorok a kiírt felirat részei, és a régi helyükön állnak (nem görgette őket a játék)
        return all(_plain(self._texts[row.row_id].text) in current
                   and any(_lies_within(row.box, live.box) for live in self._live_lines) for row in read)

    def _objective_line(self) -> OcrLine | None:
        """A párbeszédsorok alatt, középen álló, feliratszínű, mondatszerű sor (küldetésutasítás), ha fordítjuk."""
        if not self._config.translate_objectives or self._snapshot is None:
            return None
        center = self._frame_center()
        min_score = self._config.min_text_score
        rows = sorted((row for row in self._rows
                       if row.missing == 0 and not center.is_dialogue_row(row.box) and center.is_near(row.box)
                       and row.whiteness >= MIN_WHITE_FRACTION and row.row_id in self._texts
                       and self._texts[row.row_id].is_legible(min_score)), key=lambda row: row.box.top)
        text = " ".join(clean_ocr_text(self._texts[row.row_id].text.strip()) for row in rows).strip()
        if not rows or not looks_like_subtitle(text):
            return None
        return OcrLine(text, union_box([row.box for row in rows]))

    def _subtitle_lines(self, fragments: list[Fragment], *, must_be_spoken: bool = False,
                        is_tv_text: bool = False) -> tuple[OcrLine, ...]:
        """A darabok megszólalásokká fűzve, ha a szövegük feliratnak tűnik; különben üres.

        must_be_spoken: csak „Név: szöveg” megszólalás lehet, az only_spoken_lines beállítástól függetlenül.
        is_tv_text: a tévéműsor halvány felirata: kimondott szöveg, bár nincs előtte név."""
        lines = spoken_lines([(fragment.text, fragment.box) for fragment in fragments])
        if not lines or not looks_like_subtitle(" ".join(line.text for line in lines)):
            return ()
        must_be_spoken = must_be_spoken or (self._config.only_spoken_lines and not is_tv_text)
        if must_be_spoken and not is_spoken_line("\n".join(line.text for line in lines)):
            return ()  # küldetésszöveg, tábla, menü: nem kimondott szöveg, nem fordítjuk
        return lines

    def _frame_center(self) -> FrameCenter:
        """A kép közepe, a párbeszédsorok alsó határa és a sáv két széle, a pillanatkép saját képméretéből."""
        width, height = self._frame_size
        snapshot = self._snapshot
        return FrameCenter(width / 2, width, DIALOGUE_BOTTOM_RATIO * height,
                           band_left=snapshot.area.left if snapshot is not None else float("-inf"),
                           band_right=snapshot.area.right if snapshot is not None else float("inf"),
                           edge_margin=band_edge_margin(width))

    def _apply_rows(self, lines: tuple[OcrLine, ...], confirmed: bool, now: float) -> None:
        """Megváltozott a sorok állása: ugyanaz történik, mint a régi úton egy olvasás után, de stabil
        képpontoknál a felirat azonnal végleges, és amint más sor áll a régi helyén, a régi fordítás eltűnik."""
        self.stats.state_updates += 1
        text = "\n".join(line.text for line in lines)
        self._prefetch(text)
        event = self._tracker.update(text, now, confirmed=confirmed)
        if lines or self._is_place_taken():
            self._live_lines = lines
        if isinstance(event, SubtitleCleared):
            self._forget_subtitle()
        self._follow_live_lines(now)
        if isinstance(event, NewSubtitle):
            self._last_box = union_box([line.box for line in lines])
            self._start_translation(event.text)

    def _is_place_taken(self) -> bool:
        """Áll-e most látható sor ott, ahol az eddigi feliratsorok álltak (ha nem, a fordítás még ott maradhat)."""
        return any(overlaps(row.box, line.box)
                   for line in self._live_lines for row in self._rows if row.missing == 0)

    def _let_time_pass(self, has_lines: bool, now: float) -> None:
        """Semmi sem változott a képen: megnézzük, lejárt-e valamelyik várakozás."""
        if now < self._next_deadline:
            self.stats.idle_updates += 1
            return
        if not has_lines and isinstance(self._tracker.update("", now), SubtitleCleared):
            self._forget_subtitle()
        self._follow_live_lines(now)

    def _deadline(self, state: _State, pictured_at: float, now: float, is_free_waiting: bool) -> float:
        """Mikor kell akkor is ránézni az állapotra, ha semmi sem érkezik.

        is_free_waiting: máshol álló szöveg vár arra, hogy a képpontjai stabilak legyenek (lásd _assemble)."""
        lines, confirmed, _unread = state
        moments = [self._inherited_until(row) for row in self._rows  # lejár az örökölt szöveg
                   if row.row_id in self._texts and self._texts[row.row_id].provisional
                   and now < self._inherited_until(row)]
        moments += [row.first_seen + UNREAD_HIDE_SECONDS for row in self._rows  # lejár az olvasás előtti takarás
                    if row.row_id not in self._texts and now - row.first_seen < UNREAD_HIDE_SECONDS]
        if (lines and not confirmed) or is_free_waiting:
            moments.append(pictured_at + STILL_SECONDS)  # ha addig nem jön újabb kép, a mostani áll
        if (self.shown is not None and self.shown.hidden) or (not lines and self._tracker.current):
            moments.append(now + RECHECK_SECONDS)  # késő fordítás takarása, eltűnés utáni várakozás
        objective_deadline = self._objective.deadline(now)
        if objective_deadline is not None:
            moments.append(objective_deadline)  # lejár a fordításra váró küldetéssor takarása
        return min(moments, default=NEVER)

    # --- kiírás ---------------------------------------------------------------------------
    def _follow_live_lines(self, now: float) -> None:
        if self._snapshot is None:
            super()._follow_live_lines(now)  # a soralapú út most nem megy: a régi út szabályai
            return
        if self._block is not None:
            return  # egy tömbben álló üzenet látszik (hibaüzenet, eltérő sorszám)
        pairs, untranslated = match_lines(self._translated, self._live_lines)
        hidden = self._pending.boxes_to_hide(untranslated, now) + self._eager_boxes(now) + self._objective.hidden(now)
        objective = self._objective.pair()
        if objective is not None:
            pairs.append(objective)
        self.shown = build_shown(pairs, hidden, self.shown, speaker_colors=self._live_speaker_colors())

    def _eager_boxes(self, now: float) -> list[Box]:
        """Az új, még olvasatlan feliratszerű sorok: rövid ideig már olvasás előtt eltakarjuk őket."""
        return [row.box for row in self._rows if row.row_id not in self._texts and self._can_hide_early(row)
                and now - row.first_seen < UNREAD_HIDE_SECONDS]

    def _can_hide_early(self, row: RowObs) -> bool:
        """Olvasás előtt csak azt takarjuk, ami szinte biztosan párbeszédsor: a kereső biztos benne,
        feliratszínű, és nem a párbeszédsorok alatt áll (a küldetésutasításhoz sosem nyúlunk)."""
        return (row.sure and row.missing == 0 and self._whiteness(row) >= MIN_WHITE_FRACTION
                and self._frame_center().is_dialogue_row(row.box))

    def _has_line_places(self) -> bool:
        if self._snapshot is None:
            return super()._has_line_places()
        # A sorok helyét a képpontokból ismerjük: a soronként párosítható fordítás akkor kerül ki, amikor az
        # angol párja látszik; ha az közben eltűnt, nem írjuk ki egy tömbben egy másik sor fölé
        return bool(self._translated)

    def _start_translation(self, text: str) -> None:
        super()._start_translation(text)
        wake, future = self._wake, self._translate_future
        if wake is not None and future is not None:
            future.add_done_callback(lambda _done: wake())  # a fordító szálon fut: csak jelez a Tk-szálnak
