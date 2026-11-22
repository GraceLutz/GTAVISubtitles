"""A sorkereső képkockánként ingadozik (biztos / bizonytalan / nem találja): a szomszédos képkockák döntenek."""

import numpy as np

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


def centred_frame(*rows: FrameRow) -> FrameAnalysis:
    return FrameAnalysis(list(rows), np.zeros((1080, 1920, 3), np.uint8))


OBJECTIVE = Box(760, 1000, 1160, 1047)  # küldetésutasítás: pontosan középen, sosem biztos


def test_centred_unsure_row_seen_before_and_after_counts_as_sure():
    objective = FrameRow(OBJECTIVE, sure=False)

    analysis = steadied(centred_frame(objective), [frame_with(objective)], [frame_with(objective)])

    assert analysis.rows == [FrameRow(OBJECTIVE, sure=True)]


def test_centred_unsure_row_on_its_first_frame_counts_as_sure_when_later_frames_show_it():
    objective = FrameRow(OBJECTIVE, sure=False)

    analysis = steadied(centred_frame(objective), [frame_with()], [frame_with(objective), frame_with(objective)])

    assert analysis.rows == [FrameRow(OBJECTIVE, sure=True)]


def test_centred_unsure_row_on_a_single_neighbour_only_stays_unsure():
    objective = FrameRow(OBJECTIVE, sure=False)

    analysis = steadied(centred_frame(objective), [frame_with(objective)], [frame_with()])

    assert analysis.rows == [objective]


def test_steady_off_centre_unsure_row_stays_unsure():
    plate = FrameRow(PLATE, sure=False)

    analysis = steadied(centred_frame(plate), [frame_with(plate)], [frame_with(plate)])

    assert analysis.rows == [plate]


def test_line_missed_on_one_frame_between_two_finds_takes_the_previous_frame():
    previous = frame_with(FrameRow(LINE))

    assert steadied(frame_with(), [previous], [frame_with(FrameRow(LINE))]) is previous


def test_frame_after_the_subtitle_ended_stays_empty():
    empty = frame_with()

    assert steadied(empty, [frame_with(FrameRow(LINE))], [frame_with()]) is empty
