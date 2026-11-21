from gta_felirat.capture import FoundDevice
from gta_felirat.source_choice import (DeviceSource, NoSource, WindowSource, choose_source,
                                       devices_for_choice)

HAMA = FoundDevice("msmf", 0, (1920, 1080))
WEBCAM_BUSY = FoundDevice("dshow", 0, None)
CARD = FoundDevice("dshow", 1, (1920, 1080))


def test_running_veyra_is_used_first_even_with_remote_play_open():
    # Ha a Veyra fut, abban nézed a játékot; a Remote Play mellette nyitva maradhat
    assert choose_source(True, True, [HAMA]) == WindowSource("Veyra", by_start=True)


def test_remote_play_is_used_without_veyra():
    assert choose_source(True, False, [HAMA]) == WindowSource("PS Remote Play", by_start=False)


def test_single_capture_card_is_used_without_any_window():
    assert choose_source(False, False, [HAMA]) == DeviceSource(HAMA)


def test_of_several_devices_the_only_one_with_a_picture_is_used():
    assert choose_source(False, False, [WEBCAM_BUSY, CARD]) == DeviceSource(CARD)


def test_several_devices_with_a_picture_are_not_guessed():
    other = FoundDevice("dshow", 2, (1280, 720))

    assert choose_source(False, False, [CARD, other]) == NoSource((CARD, other))


def test_nothing_found_gives_no_source():
    assert choose_source(False, False, []) == NoSource(())


def test_devices_of_the_first_interface_that_has_any_are_used():
    # Ugyanaz a rögzítő a két felületen más sorszámon is lehet: a listákat nem keverjük
    assert devices_for_choice([[], [HAMA]]) == [HAMA]
    assert devices_for_choice([[CARD], [HAMA]]) == [CARD]
    assert devices_for_choice([[], []]) == []


def test_configured_webcam_is_not_taken_for_the_capture_card():
    webcam = FoundDevice("dshow", 0, (1280, 720))

    assert devices_for_choice([[webcam, CARD]], webcam_index=0) == [CARD]
    assert devices_for_choice([[webcam], [HAMA]], webcam_index=0) == []
