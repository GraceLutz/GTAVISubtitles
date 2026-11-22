"""A sorkereső képkockánként ingadozik (biztos / bizonytalan / nem találja): a szomszédos képkockák döntenek."""

from gta_felirat.clean_frame import FrameAnalysis, steadied
from gta_felirat.clean_history import FrameRow
from gta_felirat.ocr import Box

LINE = Box(587, 942, 1333, 990)
PLATE = Box(880, 780, 980, 820)  # pl. rendszámtábla: sosem biztos sor


def frame_with(*rows: FrameRow) -> FrameAnalysis:
    return FrameAnalysis(list(rows))


def test_row_unsure_here_but_sure_on_neighbouring_frames_counts_as_sure():
    shaky = Box(583, 942, 1337, 996)

    analysis = steadied(frame_with(FrameRow(shaky, sure=False)), [frame_with(FrameRow(LINE))], [])

    assert analysis.rows == [FrameRow(shaky, sure=True)]


def test_unsure_row_that_is_never_sure_stays_unsure():
    plate = FrameRow(PLATE, sure=False)

    analysis = steadied(frame_with(plate), [frame_with(plate)], [frame_with(plate)])

    assert analysis.rows == [plate]


def test_line_missed_on_one_frame_between_two_finds_takes_the_previous_frame():
    previous = frame_with(FrameRow(LINE))

    assert steadied(frame_with(), [previous], [frame_with(FrameRow(LINE))]) is previous


def test_frame_after_the_subtitle_ended_stays_empty():
    empty = frame_with()

    assert steadied(empty, [frame_with(FrameRow(LINE))], [frame_with()]) is empty
