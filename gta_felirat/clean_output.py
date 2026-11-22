"""Tiszta kép a nézőknek: a játékablak képe késleltetve, saját ablakban, rajta csak a kész fordítással.

A feliratréteg élőben dolgozik, ezért az új angol sor első pillanata átcsúszhat, és amíg a fordítás megérkezik, csak
takarás látszik. Itt a kép késik (config.clean_output_delay_seconds): mire egy képkocka kikerül, a hozzá tartozó
felirat fordítása már kész (clean_history). A takarás háttere ugyanaz a képkocka, így pontosan illeszkedik.

Saját szál: rögzít (Windows.Graphics.Capture, a teljes kliensterület), rajzol és megjelenít (OpenCV-ablak).
Az OpenCV ablakkezelése szálhoz kötött, ezért a nyitás, a frissítés és a bezárás is ezen a szálon fut.
"""

from __future__ import annotations

import ctypes
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .band_capture import fits
from .band_feed import client_box_in_frame
from .clean_frame import FrameAnalysis, analyse, with_erased_rows, without_covers_over
from .clean_history import HISTORY_SLACK_SECONDS, FrameBuffer, ShownHistory
from .frosted_cover import Backdrop
from .i18n import t
from .layer_plan import CoverJob, LayerPlan
from .ocr import Box
from .subtitle_layer import SubtitleLayer, draw_plan
from .window_stream import WindowStream, WindowStreamError

if TYPE_CHECKING:
    from .config import AppConfig

FPS = 30.0
POLL_SECONDS = 0.004  # ilyen sűrűn nézzük, jött-e új képkocka
RETRY_SECONDS = 2.0  # ha a rögzítés nem indul vagy megszakad, ennyi múlva próbáljuk újra
WINDOW_FLAGS = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
# Az OpenCV a címet ANSI-ként írja ki (az ékezet elromlik), ezért csak ASCII belső nevet kap;
# a látható címet utána Unicode-dal állítjuk be
WINDOW_NAME = "GTA6Felirat-clean"
ANALYSER_THREADS = 3  # sorkeresés és betűtörlés: képkockánként 30–120 ms, a 30 kép/mp-hez több szál kell
# Ha a gép a játékkal van elfoglalva, nem halmozunk: ennyi függő elemzés fölött a képkocka elemzés nélkül
# megy ki (elmosott takarással), és megjelenítéskor sem várunk tovább ennél
MAX_PENDING_ANALYSES = 2 * ANALYSER_THREADS
ANALYSIS_WAIT_SECONDS = 0.05
THREAD_PRIORITY_BELOW_NORMAL = -1

logger = logging.getLogger(__name__)


def _lower_priority() -> None:
    """Az elemző szálak a játék és a stream mögé sorolódnak (Windows-szálprioritás)."""
    windll = getattr(ctypes, "windll", None)
    if windll is not None:
        windll.kernel32.SetThreadPriority(windll.kernel32.GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL)


def _skipped() -> Future[FrameAnalysis]:
    future: Future[FrameAnalysis] = Future()
    future.set_result(FrameAnalysis(None))
    return future


@dataclass(frozen=True)
class CapturedFrame:
    image: np.ndarray
    # A képkocka sorai és kitörölt változata, a rögzítéskor a háttérben (mire a kép sorra kerül, kész);
    # None: megjelenítéskor készül
    analysis: Future[FrameAnalysis] | None = None


class CleanOutput:
    """start()/stop()/update_config() a Tk-szálról; a history-t a Tk-szál tölti (lásd RemotePlayApp)."""

    def __init__(self, hwnd: int, config: AppConfig) -> None:
        self._hwnd = hwnd
        self._config = config
        self.history = ShownHistory(config.clean_output_delay_seconds + HISTORY_SLACK_SECONDS)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._analysers: ThreadPoolExecutor | None = None
        self._pending = 0  # a rögzítő szál számolja; a kész elemzés visszahívása csökkenti
        self._pending_lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def update_config(self, config: AppConfig) -> None:
        self._config = config  # a szál a következő képnél már ezzel rajzol

    def start(self) -> None:
        if self.is_running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="clean-output", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)

    def _run(self) -> None:
        title = WINDOW_NAME
        layer = SubtitleLayer(self._config, on_change=lambda _changes: None)
        frames = FrameBuffer(self._config.clean_output_delay_seconds + HISTORY_SLACK_SECONDS)
        self._analysers = ThreadPoolExecutor(max_workers=ANALYSER_THREADS, thread_name_prefix="clean-rows",
                                             initializer=_lower_priority)
        cv2.namedWindow(title, WINDOW_FLAGS)
        _show_title(title, t("clean.window_title"))
        try:
            while not self._stop.is_set():
                try:
                    self._stream_until_stopped(title, layer, frames)
                except (WindowStreamError, OSError, ValueError) as err:
                    logger.warning(t("clean.capture_failed", error=err))
                    frames.clear()
                    self._stop.wait(RETRY_SECONDS)
        finally:
            cv2.destroyWindow(title)
            self._analysers.shutdown(wait=False, cancel_futures=True)

    def _stream_until_stopped(self, title: str, layer: SubtitleLayer, frames: FrameBuffer) -> None:
        stream = WindowStream(self._hwnd, 1.0 / FPS)
        try:
            next_frame_at = 0.0
            while not self._stop.is_set():
                now = time.monotonic()
                if now >= next_frame_at and self._grab(stream, frames, now):
                    next_frame_at = now + 1.0 / FPS
                    self._show(title, layer, frames, now)
                cv2.waitKey(1)  # az OpenCV-ablak csak így frissül és mozgatható
                self._stop.wait(POLL_SECONDS)
        finally:
            stream.close()

    def _submit_analysis(self, image: np.ndarray) -> Future[FrameAnalysis]:
        with self._pending_lock:
            if self._pending >= MAX_PENDING_ANALYSES:
                return _skipped()
            self._pending += 1
        future = self._analysers.submit(analyse, image, self._config)
        future.add_done_callback(self._analysis_done)
        return future

    def _analysis_done(self, _future: Future[FrameAnalysis]) -> None:
        with self._pending_lock:
            self._pending -= 1

    def _grab(self, stream: WindowStream, frames: FrameBuffer, now: float) -> bool:
        with stream.frame() as frame_size:
            if frame_size is None:
                return False
            client = client_box_in_frame(self._hwnd)
            if client is None or not fits(client, *frame_size):
                return False
            image = stream.read_box(client)
            frames.push(now, CapturedFrame(image, self._submit_analysis(image)))
        return True

    def _show(self, title: str, layer: SubtitleLayer, frames: FrameBuffer, now: float) -> None:
        config = self._config
        found = frames.at_or_before(now - config.clean_output_delay_seconds)
        if found is None:
            return  # még nincs ilyen régi kép (épp most indult)
        frame_at, captured = found
        frame = captured.image
        layer.update_config(config)
        height, width = frame.shape[:2]
        analysis = _analysis_of(captured, config)
        shown = self.history.for_frame(frame_at, now, analysis.rows)
        # A fordított vagy takart sorok betűi kitörölve; ahol a sorkereső nem talált sort, ott elmosott takarás
        image, erased_rows = with_erased_rows(frame, analysis, shown)
        plan = layer.plan((width, height), shown, True, backdrop=Backdrop(image, Box(0, 0, width, height)))
        if plan is not None:
            image = draw_on_frame(image, without_covers_over(plan, erased_rows))
        cv2.imshow(title, image)


def _analysis_of(captured: CapturedFrame, config: AppConfig) -> FrameAnalysis:
    if captured.analysis is None:
        return analyse(captured.image, config)
    try:
        return captured.analysis.result(timeout=ANALYSIS_WAIT_SECONDS)
    except FutureTimeout:
        return FrameAnalysis(None)  # nem várunk rá: a játék elsőbb; ez a képkocka az idő szerint takar
    except (ValueError, cv2.error) as err:
        logger.warning("Clean picture row search failed: %s", err)
        return FrameAnalysis(None)  # ilyenkor az idő szerint döntünk, mint az élő réteg


def _show_title(name: str, title: str) -> None:
    """Az OpenCV-ablak látható címe ékezetekkel."""
    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(None, name)
    if hwnd and not user32.SetWindowTextW(hwnd, title):
        logger.debug("Clean window title could not be set")


def draw_on_frame(frame: np.ndarray, plan: LayerPlan) -> np.ndarray:
    """A terv kirajzolása a képkocka másolatára: csak a felirat körüli részen dolgozunk.

    A draw_plan minden lépésnél új képet ad; egész 1080p-s képen ez lépésenként ~3 ms másolás lenne."""
    height, width = frame.shape[:2]
    area = _plan_area(plan, width, height)
    if area is None:
        return frame
    crop = frame[area.top:area.bottom, area.left:area.right]
    # A takarás forrása a teljes képkocka, a kivágás koordinátáiban
    backdrop = Backdrop(frame, Box(-area.left, -area.top, width - area.left, height - area.top))
    output = frame.copy()
    output[area.top:area.bottom, area.left:area.right] = draw_plan(crop, _shifted(plan, area), backdrop)
    return output


def _plan_area(plan: LayerPlan, width: int, height: int) -> Box | None:
    boxes = [job.area if isinstance(job, CoverJob) else job.box for job in plan.covers]
    if plan.panel is not None:
        boxes.append(plan.panel.box)
    boxes += [Box(job.x, job.y, job.x + job.rendered.width, job.y + job.rendered.height) for job in plan.texts]
    if not boxes:
        return None
    area = Box(max(0, min(box.left for box in boxes)), max(0, min(box.top for box in boxes)),
               min(width, max(box.right for box in boxes)), min(height, max(box.bottom for box in boxes)))
    return area if area.width > 0 and area.height > 0 else None


def _moved(box: Box, origin: Box) -> Box:
    return Box(box.left - origin.left, box.top - origin.top, box.right - origin.left, box.bottom - origin.top)


def _shifted(plan: LayerPlan, origin: Box) -> LayerPlan:
    """A terv a kivágás koordinátáiban (a bal felső sarka origin)."""
    covers = tuple(replace(job, area=_moved(job.area, origin)) if isinstance(job, CoverJob)
                   else replace(job, box=_moved(job.box, origin)) for job in plan.covers)
    panel = replace(plan.panel, box=_moved(plan.panel.box, origin)) if plan.panel is not None else None
    texts = tuple(replace(job, x=job.x - origin.left, y=job.y - origin.top) for job in plan.texts)
    return replace(plan, covers=covers, panel=panel, texts=texts, outline=None)
