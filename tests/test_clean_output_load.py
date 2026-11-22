"""A tiszta kép nem lassíthatja a játékot: nem vár a késő elemzésre, nem halmoz fel elemzéseket, és a kimaradt
képkocka a szomszédja elemzését kapja."""

import threading
import time
from concurrent.futures import Future

import numpy as np

from gta_felirat import clean_output
from gta_felirat.clean_frame import FrameAnalysis
from gta_felirat.clean_history import FrameBuffer, FrameRow
from gta_felirat.clean_output import MAX_PENDING_ANALYSES, CapturedFrame, CleanOutput
from gta_felirat.config import AppConfig
from gta_felirat.ocr import Box
from gta_felirat.subtitle_layer import SubtitleLayer

SLOW_LIMIT_SECONDS = 0.5


def test_analysis_not_ready_in_time_is_not_waited_for():
    never_done: Future = Future()
    started = time.monotonic()

    analysis = clean_output._analysis_of(CapturedFrame(np.zeros((4, 4, 3), np.uint8), never_done), AppConfig())

    assert analysis.rows is None
    assert time.monotonic() - started < SLOW_LIMIT_SECONDS


def test_frames_beyond_the_backlog_are_not_analysed(monkeypatch):
    release = threading.Event()
    calls = []

    def slow_analyse(_image, _config):
        calls.append(1)
        release.wait(5.0)
        return FrameAnalysis([])

    monkeypatch.setattr(clean_output, "analyse", slow_analyse)
    output = CleanOutput(hwnd=0, config=AppConfig())
    output._analysers = clean_output.ThreadPoolExecutor(max_workers=clean_output.ANALYSER_THREADS)
    image = np.zeros((4, 4, 3), np.uint8)
    try:
        queued = [output._submit_analysis(image) for _ in range(MAX_PENDING_ANALYSES)]
        skipped = output._submit_analysis(image)

        assert skipped.done() and skipped.result().rows is None
        release.set()
        assert all(future.result(timeout=5.0).rows == [] for future in queued)
        assert output._submit_analysis(image).result(timeout=5.0).rows == []  # a sor kiürült: újra elemzünk
    finally:
        release.set()
        output._analysers.shutdown(wait=True)


def test_frame_left_without_analysis_reuses_the_previous_frames_rows(monkeypatch):
    shown_images = []
    monkeypatch.setattr(clean_output.cv2, "imshow", lambda _title, image: shown_images.append(image))
    row = Box(700, 915, 1220, 948)
    english = np.zeros((1080, 1920, 3), np.uint8)
    english[row.top:row.bottom, row.left:row.right] = 255  # az angol sor helye világos
    analysed: Future = Future()
    analysed.set_result(FrameAnalysis([FrameRow(row)]))
    frames = FrameBuffer(keep_seconds=2.0)
    frames.push(10.0, CapturedFrame(english, analysed))
    frames.push(10.033, CapturedFrame(english.copy(), clean_output._skipped()))  # túl sok volt a függő elemzés
    config = AppConfig()
    output = CleanOutput(hwnd=0, config=config)
    layer = SubtitleLayer(config, on_change=lambda _c: None)

    output._show("clean", layer, frames, now=11.0)
    output._show("clean", layer, frames, now=11.04)

    second = shown_images[1][row.top:row.bottom, row.left:row.right]
    assert not np.array_equal(second, english[row.top:row.bottom, row.left:row.right])
