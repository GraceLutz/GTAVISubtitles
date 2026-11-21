import cv2
import numpy as np

from gta_felirat.clean_frame import FrameAnalysis, with_erased_rows, without_covers_over
from gta_felirat.clean_history import FrameRow
from gta_felirat.layer_plan import BoxJob, CoverJob, LayerPlan, Shown, ShownLine
from gta_felirat.ocr import Box
from gta_felirat.text_eraser import erase_row

WIDTH, HEIGHT = 1920, 1080
ROW_TOP, ROW_BOTTOM = 950, 990
FONT, SCALE = cv2.FONT_HERSHEY_SIMPLEX, 1.0
NAME, SPEECH, GAP = "Locals:", "Shit, what happened?", 14
NAME_WIDTH = cv2.getTextSize(NAME, FONT, SCALE, 2)[0][0]
SPEECH_WIDTH = cv2.getTextSize(SPEECH, FONT, SCALE, 2)[0][0]


def _ink_columns(name_left):
    canvas = np.zeros((HEIGHT, WIDTH), np.uint8)
    for text, x in ((NAME, name_left), (SPEECH, name_left + NAME_WIDTH + GAP)):
        cv2.putText(canvas, text, (x, ROW_BOTTOM - 10), FONT, SCALE, 255, 5, cv2.LINE_AA)
    columns = np.nonzero(canvas.max(axis=0))[0]
    return int(columns[0]), int(columns[-1]) + 1


# A sor a festett betűi szerint középre igazított, mint a játékban
_first_left = (WIDTH - NAME_WIDTH - GAP - SPEECH_WIDTH) // 2
_ink_left, _ink_right = _ink_columns(_first_left)
NAME_LEFT = _first_left + WIDTH // 2 - (_ink_left + _ink_right) // 2
SPEECH_LEFT = NAME_LEFT + NAME_WIDTH + GAP
INK_LEFT, INK_RIGHT = _ink_columns(NAME_LEFT)
CORE = Box(SPEECH_LEFT + 20, ROW_TOP, SPEECH_LEFT + SPEECH_WIDTH - 20, ROW_BOTTOM)  # a kereső csak a fehér magot fogja


def textured_background():
    rng = np.random.default_rng(7)
    noise = rng.integers(60, 110, (HEIGHT // 8, WIDTH // 8, 3), dtype=np.uint8)
    return cv2.GaussianBlur(cv2.resize(noise, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR), (9, 9), 0)


def with_subtitle(background):
    image = background.copy()
    for text, x, colour in ((NAME, NAME_LEFT, (80, 160, 230)), (SPEECH, SPEECH_LEFT, (255, 255, 255))):
        cv2.putText(image, text, (x, ROW_BOTTOM - 10), FONT, SCALE, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(image, text, (x, ROW_BOTTOM - 10), FONT, SCALE, colour, 2, cv2.LINE_AA)
    return image


def test_erasing_removes_the_whole_line_with_the_coloured_name():
    background = textured_background()
    image = with_subtitle(background)

    erased_box = erase_row(image, CORE)

    assert erased_box.left <= INK_LEFT and erased_box.right >= INK_RIGHT
    area = (slice(ROW_TOP, ROW_BOTTOM), slice(INK_LEFT, INK_RIGHT))
    before = np.abs(with_subtitle(background)[area].astype(int) - background[area]).mean()
    after = np.abs(image[area].astype(int) - background[area]).mean()
    assert after < before / 4


def test_erasing_leaves_the_picture_outside_the_line_alone():
    background = textured_background()
    image = with_subtitle(background)

    erased_box = erase_row(image, CORE)

    assert np.array_equal(image[:erased_box.top - 20], background[:erased_box.top - 20])


def test_only_rows_the_state_covers_are_erased():
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    erased = np.full_like(image, 9)
    subtitle, plate = Box(700, 950, 1220, 990), Box(880, 780, 980, 820)
    analysis = FrameAnalysis([FrameRow(subtitle), FrameRow(plate, sure=False)], erased)
    shown = Shown("Szia!", subtitle, (ShownLine("Szia!", Box(760, 952, 1160, 988)),))

    output, rows = with_erased_rows(image, analysis, shown)

    assert rows == [subtitle]
    assert output[970, 900, 0] == 9 and output[800, 900, 0] == 0


def test_erased_rows_get_no_blurred_cover_but_other_covers_stay():
    erased_row, other = Box(700, 950, 1220, 990), Box(700, 900, 1220, 940)
    plan = LayerPlan(covers=(CoverJob(erased_row, 4), BoxJob(other, 200)))

    assert without_covers_over(plan, [erased_row]).covers == (BoxJob(other, 200),)
