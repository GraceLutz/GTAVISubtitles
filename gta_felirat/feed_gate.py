"""Megmondja, használható-e most a folyamatos rögzítés (band_feed) legfrissebb sávja.

A folyamatos rögzítés olyan ablakról is ad sávot, amelynek a tartalma feketén rögzül. A fekete sáv magában
nem hiba (sötét jelenet), de ha tartós, jobb a régi, olvasásonkénti rögzítésre váltani: az össze tudja vetni
az ablak képét a képernyővel (band_capture.BandCapture), és ha kell, figyelmeztetéssel a képernyőre vált.
Amint újra van kép a sávban, megint a folyamatos rögzítés megy.
"""

from __future__ import annotations

from .band_capture import BLACK_LEVEL
from .band_feed import SOURCE_NONE, BandFeed, CapturedBand

BLACK_BANDS_BEFORE_DOUBT = 60  # ennyi egymás utáni fekete sáv (~2 mp) után nem bízunk a folyamatos rögzítésben


class FeedGate:
    def __init__(self, feed: BandFeed) -> None:
        self._feed = feed
        self._black_bands = 0  # a rögzítő szál írja, a megjelenítési szál olvassa

    def on_band(self, captured: CapturedBand) -> None:
        """A rögzítő szálon, minden új sávnál (~0,1 ms)."""
        self._black_bands = 0 if captured.image.max() > BLACK_LEVEL else self._black_bands + 1

    def band(self) -> CapturedBand | None:
        """A használható legfrissebb sáv; None: most az olvasásonkénti rögzítés menjen."""
        if self._black_bands >= BLACK_BANDS_BEFORE_DOUBT:
            return None
        return self._feed.latest()

    @property
    def is_still(self) -> bool:
        """A rögzítés él, csak áll a kép (ilyenkor nem jön új képkocka): a teljes olvasás ráér ritkán futni."""
        return (self._feed.source != SOURCE_NONE and self._feed.latest() is None
                and self._black_bands < BLACK_BANDS_BEFORE_DOUBT)
