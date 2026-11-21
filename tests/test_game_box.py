import numpy as np

from gta_felirat.clean_frame import FrameAnalysis, with_neat_game_boxes
from gta_felirat.clean_history import FrameRow
from gta_felirat.config import AppConfig
from gta_felirat.frosted_cover import Backdrop
from gta_felirat.game_box import GameBox, find_game_box
from gta_felirat.layer_plan import Shown, ShownLine
from gta_felirat.ocr import Box
from gta_felirat.subtitle_layer import SubtitleLayer

WIDTH, HEIGHT = 1920, 1080
GAME_BOX = Box(744, 940, 1176, 1000)  # a játék doboza a szövegen túl kb. 16 képponttal
TEXT_ROW = Box(760, 952, 1160, 988)


def frame_with(background: int, box_color: int | None) -> np.ndarray:
    rng = np.random.default_rng(5)
    image = np.clip(rng.normal(background, 12, (HEIGHT, WIDTH, 3)), 0, 255).astype(np.uint8)
    if box_color is not None:
        image[GAME_BOX.top:GAME_BOX.bottom, GAME_BOX.left:GAME_BOX.right] = box_color
    for x in range(TEXT_ROW.left, TEXT_ROW.right, 14):  # fehér betűszárak
        image[TEXT_ROW.top + 4:TEXT_ROW.bottom - 4, x:x + 3] = 240
    return image


def test_dark_box_behind_the_line_is_found_with_its_edges_and_color():
    game_box = find_game_box(frame_with(170, 25), TEXT_ROW)

    assert game_box is not None
    assert abs(game_box.box.left - GAME_BOX.left) <= 2 and abs(game_box.box.right - GAME_BOX.right) <= 2
    assert abs(game_box.box.top - GAME_BOX.top) <= 2 and abs(game_box.box.bottom - GAME_BOX.bottom) <= 2
    assert max(game_box.color) <= 30


def test_line_without_a_box_has_none():
    assert find_game_box(frame_with(170, None), TEXT_ROW) is None


def test_dark_scene_has_no_box_edges_so_none():
    assert find_game_box(frame_with(20, 25), TEXT_ROW) is None


# --- az éles doboz a tiszta képen ---------------------------------------------------------------

BOX_COLOR = (40, 32, 34)
LONG_HU = "Haver, Raymond fogalma sincs, miről beszél, de tényleg semmi."


def neat(text: str | None) -> tuple[np.ndarray, list[Box]]:
    image = frame_with(170, None)
    layer = SubtitleLayer(AppConfig(), on_change=lambda _c: None)
    shown = None if text is None else Shown(text, TEXT_ROW, (ShownLine(text, TEXT_ROW),))
    plan = layer.plan((WIDTH, HEIGHT), shown, True, backdrop=Backdrop(image, Box(0, 0, WIDTH, HEIGHT)))
    analysis = FrameAnalysis([FrameRow(TEXT_ROW, True, GameBox(GAME_BOX, BOX_COLOR))], image)
    texts = [] if plan is None else [Box(job.x, job.y, job.x + job.rendered.width, job.y + job.rendered.height)
                                     for job in plan.texts]
    return with_neat_game_boxes(image.copy(), plan, analysis, [TEXT_ROW]), texts


def test_game_box_is_replaced_by_one_crisp_box_around_box_and_longer_translation():
    image, texts = neat(LONG_HU)
    text = texts[0]
    assert text.left < GAME_BOX.left  # a magyar szöveg hosszabb a doboznál

    painted = image[GAME_BOX.top:GAME_BOX.bottom, text.left - 4:text.right + 4]
    assert (painted == BOX_COLOR).all()
    assert not (image[GAME_BOX.top - 10, GAME_BOX.left:GAME_BOX.right] == BOX_COLOR).all(axis=1).any()


def test_no_box_is_drawn_without_translation_on_it():
    image, _texts = neat(None)

    assert not (image[GAME_BOX.top:GAME_BOX.bottom, GAME_BOX.left:GAME_BOX.right] == BOX_COLOR).all()
