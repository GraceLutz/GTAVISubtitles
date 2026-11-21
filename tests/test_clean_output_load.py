"""A tiszta kép nem lassíthatja a játékot: nem vár a késő elemzésre, és nem halmoz fel elemzéseket."""

import threading
import time
from concurrent.futures import Future

import numpy as np

from gta_felirat import clean_output
from gta_felirat.clean_frame import FrameAnalysis
from gta_felirat.clean_output import MAX_PENDING_ANALYSES, CapturedFrame, CleanOutput
from gta_felirat.config import AppConfig

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
