"""Indításkor (forrás megadása nélkül) a program maga keresi meg, honnan jön a játék képe.

Sorrend: a PS Remote Play ablaka, a Veyra ablaka (ez capture cardot vagy Remote Playt is mutathat),
végül egy capture card. Ha több capture card is van, nem találgatunk: a felhasználó választ (--device).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .capture import FoundDevice

REMOTE_PLAY_TITLE = "PS Remote Play"
# A Veyra címei: „Veyra — 采集卡 · LIVE”, „Veyra — PS5 Remote Play”, „Veyra — 本地实验版”. A gondolatjellel
# együtt keressük, hogy egy „Veyra-NRVideo …” című böngészőlapot ne vegyünk a játék képének.
VEYRA_TITLE_START = "Veyra —"


@dataclass(frozen=True)
class WindowSource:
    title: str
    by_start: bool  # a cím elejére is illeszt (a Veyra címe a forrástól függően változik)


@dataclass(frozen=True)
class DeviceSource:
    device: FoundDevice


@dataclass(frozen=True)
class NoSource:
    devices: tuple[FoundDevice, ...]  # ha több eszköz közül nem tudunk választani, ezeket mutatjuk meg


SourceChoice = WindowSource | DeviceSource | NoSource


def choose_source(has_remote_play: bool, has_veyra: bool, devices: Sequence[FoundDevice]) -> SourceChoice:
    if has_remote_play:
        return WindowSource(REMOTE_PLAY_TITLE, by_start=False)
    if has_veyra:
        return WindowSource(VEYRA_TITLE_START, by_start=True)
    device = _only_device(devices)
    return DeviceSource(device) if device is not None else NoSource(tuple(devices))


def _only_device(devices: Sequence[FoundDevice]) -> FoundDevice | None:
    """Az egyetlen eszköz; ha több van, az egyetlen, amelyik képet is ad (a többi pl. foglalt kamera)."""
    if len(devices) == 1:
        return devices[0]
    with_picture = [device for device in devices if device.size is not None]
    return with_picture[0] if len(with_picture) == 1 else None


def devices_for_choice(found_by_api: Iterable[Sequence[FoundDevice]],
                       webcam_index: int | None = None) -> list[FoundDevice]:
    """Az első felület eszközei, amelyen van egyáltalán eszköz (ugyanaz az eszköz a két felületen
    más sorszámon is szerepelhet, ezért a felületeket nem keverjük). A beállított webkamerát
    (webcam_device_index) kihagyjuk: az nem lehet a játék képe."""
    for devices in found_by_api:
        candidates = [device for device in devices if device.index != webcam_index]
        if candidates:
            return candidates
    return []
