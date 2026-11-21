"""A RapidOCR két modellje külön-külön: szövegkeresés (detektálás) és szövegolvasás (felismerés).

A RapidOCR kész hívása minden olvasásnál mindkettőt lefuttatja, hat „pörgő” szállal: egy olvasás
~300 ms és ~1,4 processzor-másodperc volt, a program öt magot foglalt. Külön hívva a két modellt:
- a kereső kisebb képet kap (~25 ms),
- az olvasó csak a feliratszerű sorokat kapja meg, soronként (~25–65 ms),
- a szálak üresjáratban nem pörögnek (4 szál: ~0,25 processzor-másodperc olvasásonként).

Ehhez a rapidocr belső részeit használjuk, ezért a verziója rögzített (requirements.txt).
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager

import cv2
import numpy as np

OCR_THREADS = 4  # pörgés nélkül 2–4 szálon ugyanannyi a processzoridő, 4 szálon ~30%-kal gyorsabb
DET_MAX_SIDE_PX = 512  # a kereső ekkora képet kap legfeljebb; 384 alatt már kihagy sorokat
WARM_UP_TEXT = "Warm up: text."


@contextmanager
def _onnxruntime_without_spinning() -> Iterator[None]:
    """A rapidocr 3.10-ben nincs rá beállítás: a modellek létrehozása alatt kiegészítjük a munkamenet-opciókat.

    A pörgő munkaszálak várakozás közben is egy-egy teljes magot égetnek (mérve: 1430 → 230 ms/olvasás).
    """
    from rapidocr.inference_engine.onnxruntime.main import OrtInferSession

    original = OrtInferSession._init_sess_opts

    def quiet_options(cfg):
        options = original(cfg)
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        return options

    OrtInferSession._init_sess_opts = staticmethod(quiet_options)
    try:
        yield
    finally:
        OrtInferSession._init_sess_opts = staticmethod(original)


class OcrEngine:
    """Szövegkereső és szövegolvasó modell; mindkettő induláskor betöltve és bemelegítve."""

    def __init__(self) -> None:
        # Lassú import, ezért csak itt töltjük be
        from rapidocr import RapidOCR
        from rapidocr.ch_ppocr_rec import TextRecInput
        from rapidocr.utils.process_img import apply_vertical_padding, map_boxes_to_original

        engine = RapidOCR(params={
            "Global.log_level": "error",
            "Global.use_cls": False,
            "Det.limit_type": "max",
            "Det.limit_side_len": DET_MAX_SIDE_PX,
            "Rec.rec_batch_num": 1,  # a köteg minden képét a legszélesebbre tömné ki: soronként gyorsabb
            "EngineConfig.onnxruntime.intra_op_num_threads": OCR_THREADS,
        })
        with _onnxruntime_without_spinning():
            # A RapidOCR lustán tölt: szöveg nélküli próbaképnél az olvasó csak az első feliratnál töltődne be
            self._detector = engine._load_det_model()
            self._recogniser = engine._load_rec_model()
        self._flat_ratio, self._flat_min_height = engine.width_height_ratio, engine.min_height
        self._letterbox, self._map_boxes, self._rec_input = apply_vertical_padding, map_boxes_to_original, TextRecInput
        self._warm_up()

    def _warm_up(self) -> None:
        # Valódi betűk, hogy mindkét modell lefusson egyszer, még a játék előtt
        sample = np.zeros((48, 512, 3), np.uint8)
        cv2.putText(sample, WARM_UP_TEXT, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        self.detect(sample)
        self.recognise([sample])

    def detect(self, image: np.ndarray) -> list[np.ndarray]:
        """A képen talált szövegdarabok négyszögei (4 sarokpont, a kép saját koordinátáiban); olvasás nélkül."""
        record = {"preprocess": {"ratio_h": 1.0, "ratio_w": 1.0}}
        # A rapidocr a nagyon lapos képet keresés előtt függőlegesen kitömi: ezt megtartjuk
        padded, record = self._letterbox(image, record, self._flat_ratio, self._flat_min_height)
        found = self._detector(padded)
        if found.boxes is None:
            return []
        return list(self._map_boxes(found.boxes, record, image.shape[0], image.shape[1]))

    def recognise(self, patches: list[np.ndarray]) -> list[tuple[str, float]]:
        """(szöveg, bizonyosság) minden kivágott sorhoz; keresés nélkül."""
        if not patches:
            return []
        output = self._recogniser(self._rec_input(img=patches))
        return [(_plain_punctuation(text), float(score)) for text, score in zip(output.txts, output.scores)]


def _plain_punctuation(text: str) -> str:
    """A többnyelvű modell néha teljes szélességű írásjelet („Lu？”) vagy görbe idézőjelet ad: ASCII-re hajtjuk.

    Így ugyanaz a sor minden olvasásnál ugyanazt a szöveget adja (a fordítás gyorsítótára szöveg szerint keres).
    """
    return unicodedata.normalize("NFKC", text).translate(_STRAIGHT_QUOTES).strip()


_STRAIGHT_QUOTES = str.maketrans({"“": '"', "”": '"', "„": '"', "‘": "'", "’": "'"})
