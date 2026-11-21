import numpy as np

from gta_felirat.band_feed import SOURCE_NONE, SOURCE_WGC, CapturedBand
from gta_felirat.feed_gate import BLACK_BANDS_BEFORE_DOUBT, FeedGate
from gta_felirat.ocr import Box

AREA = Box(0, 0, 40, 10)
FRAME_SIZE = (40, 10)


class FakeFeed:
    def __init__(self):
        self.band = None
        self.source = SOURCE_NONE

    def latest(self):
        return self.band


def capture(feed, gate, level, source=SOURCE_WGC):
    """Egy új sáv a rögzítő száltól: egyenletes fényességű kép."""
    sequence = 1 if feed.band is None else feed.band.sequence + 1
    image = np.full((AREA.height, AREA.width, 3), level, np.uint8)
    feed.band, feed.source = CapturedBand(image, AREA, FRAME_SIZE, 10.0, sequence, source), source
    gate.on_band(feed.band)
    return feed.band


def test_latest_band_of_a_working_feed_is_used():
    feed = FakeFeed()
    gate = FeedGate(feed)

    band = capture(feed, gate, level=120)

    assert gate.band() is band
    assert not gate.is_still


def test_feed_that_has_not_started_gives_no_band_and_is_not_a_still_picture():
    gate = FeedGate(FakeFeed())

    assert gate.band() is None
    assert not gate.is_still  # nincs folyamatos rögzítés: a régi út megy a rendes ütemében


def test_short_dark_scene_keeps_using_the_feed():
    feed = FakeFeed()
    gate = FeedGate(feed)

    for _ in range(BLACK_BANDS_BEFORE_DOUBT - 1):
        band = capture(feed, gate, level=0)

    assert gate.band() is band


def test_feed_that_stays_black_is_not_trusted_until_a_picture_returns():
    # Ha az ablak tartalma feketén rögzül, a régi út veszi észre (összeveti a képernyővel), nem a folyamatos rögzítés
    feed = FakeFeed()
    gate = FeedGate(feed)
    for _ in range(BLACK_BANDS_BEFORE_DOUBT):
        capture(feed, gate, level=0)

    assert gate.band() is None
    assert not gate.is_still  # nem álló kép: a régi útnak a rendes ütemében kell kiderítenie, mi a baj

    band = capture(feed, gate, level=120)

    assert gate.band() is band


def test_feed_whose_band_went_stale_shows_a_still_picture():
    # Álló képnél nem jön új képkocka: a rögzítés él, de a legfrissebb sávja már nem élő
    feed = FakeFeed()
    gate = FeedGate(feed)
    capture(feed, gate, level=120)

    feed.band = None  # a rögzítés szerint a sáv elavult

    assert gate.band() is None
    assert gate.is_still
