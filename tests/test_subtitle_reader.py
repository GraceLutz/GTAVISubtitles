import numpy as np

from gta_felirat.config import Region
from gta_felirat.ocr import Box, SubtitleReader, region_to_box

WIDTH, HEIGHT = 1920, 1080
AREA = region_to_box(Region(), WIDTH, HEIGHT)
LINE = Box(780, 957, 1140, 994)
MIDDLE_WIDTH = 1074  # a kereső a kép szélességének középső 56%-át kapja (kerekítve)


def band_with(*boxes, level=255):
    """Sötét feliratsáv, a megadott dobozokban fehér „betűkkel” (4 képpontos vonások)."""
    band = np.full((AREA.height, AREA.width, 3), 25, np.uint8)
    for box in boxes:
        for x in range(box.left - AREA.left, box.right - AREA.left - 4, 8):
            band[box.top - AREA.top:box.bottom - AREA.top, x:x + 4] = level
    return band


def polygon(box, offset_x):
    """A kereső válasza: a doboz négy sarka a neki adott kép koordinátáiban."""
    left, right = box.left - AREA.left - offset_x, box.right - AREA.left - offset_x
    top, bottom = box.top - AREA.top, box.bottom - AREA.top
    return np.array([[left, top], [right, top], [right, bottom], [left, bottom]], np.float32)


class FakeEngine:
    """Megjegyzi, mekkora képet kapott a kereső és mit az olvasó; előre megadott dobozokat és szövegeket ad."""

    def __init__(self, texts):
        self.texts = texts  # doboz -> szöveg
        self.detected_widths = []
        self.recognised = []

    def detect(self, image):
        self.detected_widths.append(image.shape[1])
        offset_x = (AREA.width - image.shape[1]) // 2 if image.shape[1] < AREA.width else 0
        visible = [box for box in self.texts
                   if box.left - AREA.left >= offset_x and box.right - AREA.left <= offset_x + image.shape[1]]
        return [polygon(box, offset_x) for box in visible]

    def recognise(self, patches):
        self.recognised.append([patch.shape[:2] for patch in patches])
        return [(text, 0.99) for text in list(self.texts.values())[:len(patches)]]


def reader_with(engine):
    return SubtitleReader(Region(), 0.6, engine)


def test_reads_a_centred_white_line_from_the_middle_part_only():
    engine = FakeEngine({LINE: "How about you, how you doin'?"})

    result = reader_with(engine).read_band(band_with(LINE), AREA)

    assert result.text == "How about you, how you doin'?"
    assert result.lines[0].box == LINE
    assert engine.detected_widths == [MIDDLE_WIDTH]  # a kép középső 56%-a, nem a teljes sáv


def test_dark_band_is_not_searched_at_all():
    engine = FakeEngine({LINE: "x"})

    result = reader_with(engine).read_band(np.full((AREA.height, AREA.width, 3), 25, np.uint8), AREA)

    assert result.text == "" and engine.detected_widths == []


def test_non_white_piece_is_dropped_before_it_is_read():
    engine = FakeEngine({LINE: "How about you, how you doin'?"})
    plate = Box(900, 880, 1010, 920)  # a kereső megtalálja, de nem feliratszínű (a sávon itt nincs fehér)
    engine.texts = {plate: "BM472H", LINE: "How about you, how you doin'?"}

    rows = reader_with(engine).detect_rows(band_with(LINE), AREA)

    assert rows == [LINE]


def test_row_reaching_the_side_of_the_middle_part_makes_the_whole_band_searched():
    wide = Box(425, 957, 1495, 994)  # a középső kivágás széléig ér: lehet, hogy kilóg belőle
    engine = FakeEngine({wide: "A very long line."})

    reader_with(engine).detect_rows(band_with(wide), AREA)

    assert engine.detected_widths == [MIDDLE_WIDTH, AREA.width]


def test_known_rows_are_read_without_searching():
    engine = FakeEngine({LINE: "How about you, how you doin'?"})

    texts = reader_with(engine).read_rows(band_with(LINE), AREA, [LINE])

    assert texts == [("How about you, how you doin'?", 0.99)]
    assert engine.detected_widths == []
    assert engine.recognised == [[(LINE.height, LINE.width + 16)]]  # oldalt 8–8 képponttal bővebb kivágás


def test_low_score_text_is_ignored():
    class Unsure(FakeEngine):
        def recognise(self, patches):
            return [("Hw abt yu", 0.3) for _ in patches]

    result = reader_with(Unsure({LINE: "x"})).read_band(band_with(LINE), AREA)

    assert result.text == ""


def test_curly_quotes_and_full_width_punctuation_are_read_as_plain_ascii():
    from gta_felirat.ocr_engine import _plain_punctuation

    assert _plain_punctuation("your “little bird”？ ") == 'your "little bird"?'
    assert _plain_punctuation("It’s me.") == "It's me."


# --- a sáv a kép középső részén kívül -------------------------------------------

def test_band_beside_the_middle_part_is_searched_whole_instead_of_as_an_empty_crop():
    region = Region(left=0.80, top=0.70, right=0.98, bottom=0.98)  # a 0,22–0,78 középső részen kívül
    area = region_to_box(region, WIDTH, HEIGHT)
    engine = FakeEngine({})

    SubtitleReader(region, 0.6, engine).detect_rows(np.full((area.height, area.width, 3), 255, np.uint8), area)

    assert engine.detected_widths == [area.width]
