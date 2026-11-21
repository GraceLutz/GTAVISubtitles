"""A feliratfeldolgozás soralapú (képpontokból vezérelt) útja.

A régi út (SubtitlePipeline.submit_band / poll) minden olvasásnál az egész sávban keres és olvas
(~300 ms). Itt a sávot egy gyors sorkereső figyeli (row_tracker), és:
- az újonnan megjelent, feliratszerű sort azonnal eltakarjuk, még mielőtt elolvastuk volna
  (legfeljebb EAGER_HIDE_SECONDS-ig);
- csak az új képpontú sort olvassuk be; második olvasás helyett az a megerősítés, hogy a sor
  képpontjai STABLE_CAPTURES rögzítésen át nem változtak;
- a lefordított sor fordítása olvasás nélkül követi a sort, amikor a játék feljebb görgeti.
Hogy melyik sor felirat, hogyan fordítjuk, hová kerül a fordítás, és meddig takarjuk a még le nem
fordított sort, az ugyanaz, mint a régi úton.

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
from .line_placement import build_shown, match_lines
from .ocr import Box, OcrLine, SubtitleReader, is_spoken_line, looks_like_subtitle, spoken_lines, union_box
from .row_tracker import STABLE_CAPTURES, BandSnapshot, RowObs, RowText, is_same_place, overlaps
from .subtitle_pipeline import SubtitlePipeline
from .subtitle_tracker import NewSubtitle, SubtitleCleared
from .translation_service import TranslationService

EAGER_HIDE_SECONDS = 0.35  # olvasás előtt legfeljebb eddig takarunk el egy új, feliratszerű sort
# Időfüggő állapotnál (késő fordítás takarása, eltűnés utáni várakozás) ilyen sűrűn nézünk rá, ha a kép nem változik
RECHECK_SECONDS = 0.1
# Ha ennyi ideje nem jött újabb kép, a mostani áll (a rögzítés csak változáskor ad képet): az egyetlen képen
# látott sort sem kell tovább megerősíteni. A leglassabb képforrás két képe között ~0,07 mp telik el
STILL_SECONDS = 0.15
NEVER = float("inf")

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
        self._last_state: _State | None = None
        self._is_on_screen = False  # a legutóbbi kép szerint látszik a felirat
        self._next_deadline = NEVER
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
        self._snapshot, self._rows = None, ()
        self._last_state, self._is_on_screen, self._next_deadline = None, False, NEVER
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
        has_news = self._take_texts(texts) or has_news
        current = self._snapshot
        if current is not None and (has_news or self._last_state is None or now >= self._next_deadline):
            self._refresh(current.captured_at, now)
        self._collect_translation(now)
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
        self._snapshot, self._rows, self._frame_size = snapshot, snapshot.rows, snapshot.frame_size
        return True

    def _texts_for(self, rows: tuple[RowObs, ...]) -> dict[int, RowText]:
        """A megmaradt sorok szövegei; az eltűnt sor pontos helyén megjelent új sor a felismerésig örökli a szövegét.

        Ilyenkor többnyire ugyanaz a szöveg kapott új azonosítót (megváltozott mögötte a háttér): ne tűnjön el miatta
        egy pillanatra a fordítása, és ne takarjuk el újra azt a sort, amelyről már tudjuk, hogy nem felirat."""
        alive = {row.row_id for row in rows}
        known = {row.row_id for row in self._rows}
        texts = {row_id: text for row_id, text in self._texts.items() if row_id in alive}
        orphans = [(row.box, self._texts[row.row_id]) for row in self._rows
                   if row.row_id not in alive and row.row_id in self._texts]
        for row in rows:
            left_behind = next((text for box, text in orphans if is_same_place(box, row.box)), None)
            if row.row_id not in known and left_behind is not None:
                texts[row.row_id] = replace(left_behind, row_id=row.row_id, provisional=True)
                self.stats.inherited_texts += 1
        return texts

    def _take_texts(self, texts: Sequence[RowText]) -> bool:
        alive = {row.row_id for row in self._rows}
        fresh = {item.row_id: item for item in texts if item.row_id in alive}
        stale = len(texts) - len(fresh)
        if stale:  # a sor közben eltűnt (vagy más méretű lett a kép): az eredménye elavult
            self.stats.stale_texts += stale
            logger.debug("%d row text(s) dropped: the rows are gone", stale)
        for item in fresh.values():
            self._palette.learn(item.speaker_colors)
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
        lines, used = self._assemble()
        is_still = now - pictured_at >= STILL_SECONDS
        confirmed = is_still or all(row.stable_captures >= STABLE_CAPTURES for row in used)
        unread = tuple((row.row_id, row.box, self._can_hide_early(row))
                       for row in self._rows if row.row_id not in self._texts)
        state = (lines, confirmed, unread)
        if state != self._last_state:
            self._last_state = state
            self._apply_rows(lines, confirmed, now)
        else:
            self._let_time_pass(bool(lines), now)
        self._is_on_screen = bool(lines)  # csak a most is látszó sorokból áll (lásd _assemble)
        deadline = self._deadline(state, pictured_at, now)
        # A semmit sem változtató képkocka (pl. 30 kép/mp-es videóablak álló jeleneten) nem tolhatja
        # későbbre a még esedékes ellenőrzést, különben az eltűnt felirat fordítása sosem tűnne el
        is_pending = now < self._next_deadline
        self._next_deadline = min(deadline, self._next_deadline) if is_pending else deadline

    def _unexpired_texts(self, now: float) -> dict[int, RowText]:
        """Az örökölt szöveg csak addig érvényes, ameddig egy olvasatlan sort eltakarnánk."""
        expired = {row.row_id for row in self._rows if now - row.first_seen >= EAGER_HIDE_SECONDS}
        return {row_id: text for row_id, text in self._texts.items()
                if not (text.provisional and row_id in expired)}

    def _assemble(self) -> tuple[tuple[OcrLine, ...], tuple[RowObs, ...]]:
        """(az elfogadott feliratsorok, a sorok, amelyekből lettek): a SubtitleReader.read_band szűrőivel."""
        min_score = self._config.min_text_score
        # Csak a most is látszó sor számít: a türelmi időben csak emlékezett sor (missing > 0) nem tartja
        # életben a feliratot (ha utána nem jön több kép, mert áll a jelenet, a fordítás örökre kint maradna).
        # A szövegét megtartjuk: ha a sor visszatér, nem olvassuk újra
        legible = {row.box: row for row in self._rows
                   if row.missing == 0 and row.row_id in self._texts
                   and self._texts[row.row_id].is_legible(min_score)}
        fragments = [Fragment(self._texts[row.row_id].text.strip(), box, row.whiteness)
                     for box, row in legible.items()]
        kept = select_subtitle(fragments, self._frame_center())
        lines = spoken_lines([(fragment.text, fragment.box) for fragment in kept])
        if not lines or not looks_like_subtitle(" ".join(line.text for line in lines)):
            return (), ()
        if self._config.only_spoken_lines and not is_spoken_line("\n".join(line.text for line in lines)):
            return (), ()  # küldetésszöveg, tábla, menü: nem kimondott szöveg, nem fordítjuk
        return lines, tuple(legible[fragment.box] for fragment in kept)

    def _frame_center(self) -> FrameCenter:
        """A kép közepe és a párbeszédsorok alsó határa, a pillanatkép saját képméretéből."""
        width, height = self._frame_size
        return FrameCenter(width / 2, width, DIALOGUE_BOTTOM_RATIO * height)

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

    def _deadline(self, state: _State, pictured_at: float, now: float) -> float:
        """Mikor kell akkor is ránézni az állapotra, ha semmi sem érkezik."""
        lines, confirmed, _unread = state
        waiting = [row for row in self._rows if row.row_id not in self._texts or self._texts[row.row_id].provisional]
        moments = [row.first_seen + EAGER_HIDE_SECONDS for row in waiting
                   if now - row.first_seen < EAGER_HIDE_SECONDS]  # lejár az olvasás előtti takarás
        if lines and not confirmed:
            moments.append(pictured_at + STILL_SECONDS)  # ha addig nem jön újabb kép, a mostani áll
        if (self.shown is not None and self.shown.hidden) or (not lines and self._tracker.current):
            moments.append(now + RECHECK_SECONDS)  # késő fordítás takarása, eltűnés utáni várakozás
        return min(moments, default=NEVER)

    # --- kiírás ---------------------------------------------------------------------------
    def _follow_live_lines(self, now: float) -> None:
        if self._snapshot is None:
            super()._follow_live_lines(now)  # a soralapú út most nem megy: a régi út szabályai
            return
        if self._block is not None:
            return  # egy tömbben álló üzenet látszik (hibaüzenet, eltérő sorszám)
        pairs, untranslated = match_lines(self._translated, self._live_lines)
        hidden = self._pending.boxes_to_hide(untranslated, now) + self._eager_boxes(now)
        self.shown = build_shown(pairs, hidden, self.shown, speaker_colors=self._live_speaker_colors())

    def _eager_boxes(self, now: float) -> list[Box]:
        """Az új, még olvasatlan feliratszerű sorok: rövid ideig már olvasás előtt eltakarjuk őket."""
        return [row.box for row in self._rows if row.row_id not in self._texts and self._can_hide_early(row)
                and now - row.first_seen < EAGER_HIDE_SECONDS]

    def _can_hide_early(self, row: RowObs) -> bool:
        """Olvasás előtt csak azt takarjuk, ami szinte biztosan párbeszédsor: a kereső biztos benne,
        feliratszínű, és nem a párbeszédsorok alatt áll (a küldetésutasításhoz sosem nyúlunk)."""
        return (row.sure and row.missing == 0 and row.whiteness >= MIN_WHITE_FRACTION
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
